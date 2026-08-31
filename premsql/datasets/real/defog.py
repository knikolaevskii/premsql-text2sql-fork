from pathlib import Path
from typing import Optional, Union

from premsql.datasets.base import Text2SQLBaseDataset
from premsql.logger import setup_console_logger

logger = setup_console_logger("[DEFOG-DATASET]")


class DefogDataset(Text2SQLBaseDataset):
    """
    Defog ships one schema JSON file per db_id alongside its database
    folder, so it uses schema_source "schema_path" (see
    Text2SQLBaseInstance) instead of the SQLite introspection Bird/Spider
    rely on.
    """

    VALID_SPLITS = ("test", "questions_gen")
    SCHEMA_SOURCE = "schema_path"

    def __init__(
        self,
        split: str,
        dataset_folder: Optional[Union[str, Path]] = "./data",
        hf_token: Optional[str] = None,
        force_download: Optional[bool] = False,
    ):
        dataset_folder = Path(dataset_folder)
        defog_folder = dataset_folder / "defog"
        if not defog_folder.exists():
            raise ValueError("Defog dataset not found")

        if split == "test":
            json_file_name = "test.json"
        elif split == "questions_gen":
            json_file_name = "questions_gen.json"
        else:
            raise ValueError("Split should be test or questions_gen")

        super().__init__(
            split=split,
            dataset_path=defog_folder,
            database_folder_name="database",
            json_file_name=json_file_name,
            hf_token=hf_token,
        )
        logger.info("Loaded Defog Dataset")

        # Defog's gold query field is named "query"; normalize it to "SQL"
        # so it's compatible with Text2SQLBaseInstance/the evaluator.
        for content in self.dataset:
            content["SQL"] = content["query"]

    def setup_dataset(
        self,
        filter_by: tuple | None = None,
        num_rows: int | None = None,
        num_fewshot: int | None = None,
        model_name_or_path: str | None = None,
        prompt_template: str | None = None,
        tokenize: bool | None = False,
    ):
        logger.info("Setting up Defog Dataset")
        return super().setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            num_fewshot=num_fewshot,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
        )
