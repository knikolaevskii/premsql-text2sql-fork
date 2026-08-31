from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download

from premsql.datasets.base import Text2SQLBaseDataset
from premsql.logger import setup_console_logger

logger = setup_console_logger("[BIRD-DATASET]")


class BirdDataset(Text2SQLBaseDataset):
    def __init__(
        self,
        split: str,
        dataset_folder: Optional[Union[str, Path]] = "./data",
        hf_token: Optional[str] = None,
        force_download: Optional[bool] = False,
        **kwargs
    ):
        dataset_folder = Path(dataset_folder)
        bird_folder = dataset_folder / "bird"
        # BirdBench stores each split in its own top-level folder
        # (bird/train/train_databases, bird/validation/dev_databases), so
        # fetch only the split being asked for. Downloading the whole repo
        # pulls the training databases too — well over 100GB — even when
        # only the validation split is wanted. The existence check is on
        # the split folder rather than the parent so that asking for a
        # second split later still downloads it.
        split_folder = bird_folder / split
        if not split_folder.exists() or force_download:
            bird_folder.mkdir(parents=True, exist_ok=True)

            # Download it from hf hub
            snapshot_download(
                repo_id="premai-io/birdbench",
                repo_type="dataset",
                local_dir=dataset_folder / "bird",
                force_download=force_download,
                allow_patterns=[f"{split}/*"],
            )

        dataset_path = bird_folder / split

        database_folder_name = kwargs.get("database_folder_name", None) or (
            "train_databases" if split == "train" else "dev_databases"
        )
        json_file_name = kwargs.get("json_file_name", None) or (
            "train.json" if split == "train" else "validation.json"
        )

        super().__init__(
            split=split,
            dataset_path=dataset_path,
            database_folder_name=database_folder_name,
            json_file_name=json_file_name,
            hf_token=hf_token,
        )
        logger.info("Loaded Bird Dataset")

    def setup_dataset(
        self,
        filter_by: tuple | None = None,
        num_rows: int | None = None,
        num_fewshot: int | None = None,
        model_name_or_path: str | None = None,
        prompt_template: str | None = None,
        tokenize: bool | None = False 
    ):
        logger.info("Setting up Bird Dataset")
        return super().setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
            num_fewshot=num_fewshot
        )