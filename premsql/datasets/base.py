import json
import os
import sqlite3
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from premsql.logger import setup_console_logger
from premsql.prompts import BASE_TEXT2SQL_PROMPT
from premsql.utils import (
    filter_options,
    get_accepted_filters,
    get_random_few_shot_prompts,
    tokenize_fn,
)

logger = setup_console_logger(name="[DATASET]")

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    logger.warn("Ensure transformers and torch. Install using: pip install torch transformers")


IGNORE_INDEX = -100


class Text2SQLBaseInstance:
    """
    `schema_source` controls how each row's schema text is built for the
    prompt:
      - "sqlite" (default): introspect the row's db_path SQLite file
        directly. This is upstream's original behavior, used by datasets
        that ship a queryable .sqlite file per db_id (Bird, Spider).
      - "metadata_file": look up the row's db_id inside one shared JSON
        file (data_schema_file) with a top-level "table_metadata" map.
        Used by datasets that ship a single combined schema file rather
        than per-db databases (WikiSQL).
      - "schema_path": read a per-row JSON file at row["schema_path"] with
        a "table_metadata" map. Used by datasets that ship one schema file
        per db_id alongside the database (Defog).
    """

    def __init__(self, dataset: list[dict], schema_source: str = "sqlite") -> None:
        assert "question" in dataset[0], "question is required"
        assert "SQL" in dataset[0], "sql is required"
        assert "db_path" in dataset[0], "db_path is required"
        assert "db_id" in dataset[0], "db_id is required"

        self.dataset = dataset
        self.schema_source = schema_source

    def __repr__(self) -> str:
        return str(json.dumps(self.dataset[:3], indent=4))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return dict(**self.dataset[idx])

    def schema_prompt(self, db_path: str) -> str:
        schemas = {}
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            if table_name == "sqlite_sequence":
                continue
            cursor.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            )
            create_table_sql = cursor.fetchone()
            if create_table_sql:
                schemas[table_name] = create_table_sql[0]
            else:
                schemas[table_name] = "Schema does not exist"

        schema_prompt = "\n".join(
            schemas[table[0]] for table in tables if table[0] != "sqlite_sequence"
        )
        return schema_prompt

    @staticmethod
    def to_prompt_schema(
        md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
    ) -> str:
        """
        Renders a {table_name: [{"column_name", "data_type", ...}, ...]}
        metadata map as CREATE TABLE-style DDL text, for datasets that ship
        column metadata as JSON instead of an actual queryable database.
        """
        md_create = ""
        table_names = list(md.keys())
        if seed:
            np.random.seed(seed)
            np.random.shuffle(table_names)
        for table in table_names:
            md_create += f"CREATE TABLE {table} (\n"
            columns = md[table]
            if seed:
                np.random.seed(seed)
                np.random.shuffle(columns)
            for i, column in enumerate(columns):
                col_name = column["column_name"]
                if " " in col_name:
                    col_name = f'"{col_name}"'
                dtype = column["data_type"]
                col_desc = column.get("column_description", "").replace("\n", " ")
                if col_desc:
                    col_desc = f" --{col_desc}"
                if i < len(columns) - 1:
                    md_create += f"  {col_name} {dtype},{col_desc}\n"
                else:
                    md_create += f"  {col_name} {dtype}{col_desc}\n"
            md_create += ");\n"
        return md_create

    @staticmethod
    def _load_table_metadata(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)["table_metadata"]

    def resolve_schema(self, blob: dict, data_schema_file: Optional[str] = None) -> str:
        if self.schema_source == "sqlite":
            return self.schema_prompt(blob["db_path"])
        if self.schema_source == "schema_path":
            md = self._load_table_metadata(blob["schema_path"])
            return self.to_prompt_schema(md)
        if self.schema_source == "metadata_file":
            assert data_schema_file, "data_schema_file is required for schema_source='metadata_file'"
            md = self._load_table_metadata(data_schema_file)
            db_id = blob["db_id"]
            return self.to_prompt_schema({db_id: md[db_id]})
        raise ValueError(f"Unknown schema_source: {self.schema_source}")

    def additional_prompt(self, prompt: Optional[str] = None):
        return "" if prompt is None else f"# Additional Knowledge:\n{prompt}"

    def add_few_shot_examples(self, db_id: str, k: int = 3) -> str:
        assert k > 0, "k should be greater than 0"
        db_fewshot_prompts_map = get_random_few_shot_prompts(
            dataset=self.dataset, num_few_shot=k
        )
        return db_fewshot_prompts_map[db_id]

    @staticmethod
    def _load_prompt_template(prompt_template: Optional[str]) -> str:
        if prompt_template is None:
            return BASE_TEXT2SQL_PROMPT
        # Callers may pass either raw template content or a path to a file
        # containing it. A template string always contains newlines, so
        # only attempt the filesystem check for short, single-line values
        # (avoids Path().exists() raising on an over-long "path").
        if "\n" not in prompt_template and len(prompt_template) < 260:
            path = Path(prompt_template)
            if path.exists():
                return path.read_text()
        return prompt_template

    def apply_prompt(
        self,
        num_fewshot: Optional[int] = None,
        prompt_template: Optional[str] = None,
        data_schema_file: Optional[str] = None,
    ):
        prompt_template_content = self._load_prompt_template(prompt_template)

        for blob in tqdm(self.dataset, total=len(self.dataset), desc="Applying prompt"):
            few_shot_prompt = (
                ""
                if num_fewshot is None
                else self.add_few_shot_examples(db_id=blob["db_id"], k=num_fewshot)
            )
            schemas = self.resolve_schema(blob, data_schema_file=data_schema_file)
            instructions = blob.get("instructions", "")

            final_prompt = prompt_template_content.format(
                db_type=blob.get("db_type", "SQLite"),
                schemas=schemas,
                table_metadata_string=schemas,
                additional_knowledge=(
                    ""
                    if "knowledge" not in blob
                    else self.additional_prompt(blob["knowledge"])
                ),
                instructions=instructions,
                few_shot_examples=few_shot_prompt,
                k_shot_prompt=few_shot_prompt,
                question=blob["question"],
                user_question=blob["question"],
            )
            blob["prompt"] = final_prompt
        return self.dataset


class SupervisedDatasetForTraining(torch.utils.data.Dataset):
    @classmethod
    def load_from_pth(cls, dataset_path: Union[str, Path]):
        dataset_path = str(dataset_path)
        dataset_dict = torch.load(dataset_path)

        assert "input_ids" in dataset_dict[0], "input_ids is required"
        assert "labels" in dataset_dict[0], "labels is required"
        assert "raw" in dataset_dict[0], "raw is required"

        return cls(
            dataset=dataset_dict,
            model_name_or_path=None,
            hf_token=None,
        )

    def __init__(
        self,
        dataset: dict,
        model_name_or_path: Optional[str] = None,
        tokenize: Optional[bool] = False,
        hf_token: Optional[str] = None,
    ):
        assert "prompt" in dataset[0], "key prompt is required"
        assert "SQL" in dataset[0], "key SQL is required"

        self.is_tokenized = False

        if model_name_or_path is not None and tokenize:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                padding_side="right",
                token=hf_token,
            )
            self.dataset = dataset

            if self.tokenizer.chat_template:
                for content in self.dataset:
                    content["prompt"] = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content["prompt"]}], tokenize=False
                    )
                logger.info("Casted dataset with model chat template")

            logger.info("Starting Tokenization ...")
            sources, targets = [], []
            for example in self.dataset:
                sources.append(example["prompt"])
                targets.append(f"{example['SQL']}{self.tokenizer.eos_token}")

            data_dict = self.preprocess(sources=sources, targets=targets)
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.is_tokenized = True

        elif "input_ids" in dataset[0] and "labels" in dataset[0]:
            self.dataset = dataset
            self.input_ids = dataset["input_ids"]
            self.labels = dataset["labels"]
            self.is_tokenized = True

        elif model_name_or_path is not None and not tokenize:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                padding_side="right",
                token=hf_token,
            )
            self.dataset = dataset
            if self.tokenizer.chat_template:
                for content in self.dataset:
                    content["prompt"] = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content["prompt"]}], tokenize=False
                    )
                logger.info("Casted dataset with model chat template")
        else:
            self.dataset = dataset

    def preprocess(self, sources: Sequence[str], targets: Sequence[str]):
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            tokenize_fn(strings, self.tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = deepcopy(input_ids)

        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if self.is_tokenized:
            return dict(
                input_ids=self.input_ids[idx],
                labels=self.labels[idx],
                raw=dict(**self.dataset[idx]),
            )
        else:
            return dict(**self.dataset[idx])

    def save_tokenized_dataset(self, path_to_save: Union[str, Path]):
        torch.save(self.dataset, str(path_to_save))
        logger.info("Dataset saved successfully in {}".format(path_to_save))


class Text2SQLBaseDataset(ABC):
    # Subclasses override these two to plug into a different data shape
    # without touching setup_dataset itself.
    VALID_SPLITS = ("train", "validation", "test")
    SCHEMA_SOURCE = "sqlite"  # "sqlite" | "schema_path" | "metadata_file"

    def __init__(
        self,
        split: str,
        dataset_path: Union[str, Path],
        database_folder_name: str,
        json_file_name: str,
        data_schema_file: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.database_folder_name = database_folder_name
        self.dataset = json.load(open(self.dataset_path / json_file_name, "r"))
        self.data_schema_file = data_schema_file
        assert split in self.VALID_SPLITS, ValueError(
            f"Split should be one of {self.VALID_SPLITS}"
        )
        self.split = split
        self.hf_token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)

    @property
    def raw_dataset(self):
        return self._text2sql_dataset.dataset

    @property
    def filter_availables(self):
        return get_accepted_filters(data=self.dataset)

    @abstractmethod
    def setup_dataset(
        self,
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        tokenize: Optional[bool] = False,
        prompt_template: Optional[str] = None,
    ):
        for content in self.dataset:
            content["db_path"] = str(
                self.dataset_path
                / f"{self.database_folder_name}"
                / content["db_id"]
                / f"{content['db_id']}.sqlite"
            )
            if self.SCHEMA_SOURCE == "schema_path":
                content["schema_path"] = str(
                    self.dataset_path
                    / f"{self.database_folder_name}"
                    / content["db_id"]
                    / f"{content['db_id']}.json"
                )

        if filter_by:
            self.dataset = filter_options(data=self.dataset, filter_by=filter_by)

        if num_rows:
            self.dataset = self.dataset[:num_rows]

        self.dataset = Text2SQLBaseInstance(
            dataset=self.dataset, schema_source=self.SCHEMA_SOURCE
        ).apply_prompt(
            num_fewshot=num_fewshot,
            prompt_template=prompt_template,
            data_schema_file=self.data_schema_file,
        )
        return SupervisedDatasetForTraining(
            dataset=self.dataset,
            model_name_or_path=model_name_or_path,
            hf_token=self.hf_token,
            tokenize=tokenize,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return dict(**self.dataset[idx])


class StandardDataset(Text2SQLBaseDataset):
    def __init__(
        self,
        split: str,
        dataset_path: Union[str, Path],
        database_folder_name: str,
        json_file_name: str,
        hf_token: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            dataset_path=dataset_path,
            database_folder_name=database_folder_name,
            json_file_name=json_file_name,
            hf_token=hf_token,
        )

    def setup_dataset(
        self,
        filter_by: tuple | None = None,
        num_rows: int | None = None,
        num_fewshot: int | None = None,
        model_name_or_path: str | None = None,
        prompt_template: str | None = None,
        tokenize: bool | None = False,
    ):
        logger.info("Setting up Dataset")
        return super().setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
            num_fewshot=num_fewshot,
        )
