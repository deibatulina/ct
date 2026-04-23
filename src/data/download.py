from pathlib import Path

import config
from huggingface_hub import hf_hub_download

from src.data.dataset import maybe_login_to_huggingface
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def build_volume_repo_path(volume_name):
    """
    Строит путь к CT-файлу внутри хранилища Hugging Face.
    """
    parts = volume_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected volume name format: {volume_name}")

    split_prefix = parts[0]
    study_number = parts[1]
    subgroup = parts[2]

    if split_prefix == "train":
        root_folder = "train_preprocessed"
    elif split_prefix == "valid":
        root_folder = "valid_preprocessed"
    else:
        raise ValueError(f"Unsupported split prefix in volume name: {volume_name}")

    study_folder = f"{split_prefix}_{study_number}"
    subgroup_folder = f"{split_prefix}_{study_number}{subgroup}"

    return f"dataset/{root_folder}/{study_folder}/{subgroup_folder}/{volume_name}"


def delete_volume_from_hf_cache(volume_path):
    """
    Удаляет том из кэша HF по пути, который вернул hf_hub_download.
    """
    cache_path = Path(volume_path)
    blob_path = None

    try:
        if cache_path.is_symlink():
            blob_path = cache_path.resolve(strict=True)
    except FileNotFoundError:
        blob_path = None

    try:
        cache_path.unlink(missing_ok=True)
    except Exception as error:
        LOGGER.warning("Failed to remove cached snapshot entry %s: %s", cache_path, error)

    if blob_path is not None:
        try:
            blob_path.unlink(missing_ok=True)
        except Exception as error:
            LOGGER.warning("Failed to remove cached blob %s: %s", blob_path, error)


def delete_volumes_from_hf_cache(volume_paths):
    for volume_path in volume_paths.values():
        delete_volume_from_hf_cache(volume_path)


def download_selected_volumes(records):
    """
    Скачивает только те CT-файлы, которые нужны для итогового отобранного датасета.
    """
    requested_volumes = []
    seen_studies = set()

    for record in records:
        study_id = record["meta"]["study_id"]
        if study_id in seen_studies:
            continue
        seen_studies.add(study_id)
        requested_volumes.append(study_id)

    stats = {
        "requested_volumes": len(requested_volumes),
        "downloaded_volumes": 0,
        "cached_volumes": 0,
        "failed_volumes": 0,
    }
    available_studies = set()
    volume_paths = {}
    login_performed = False

    for index, volume_name in enumerate(requested_volumes, start=1):
        try:
            if not login_performed:
                maybe_login_to_huggingface()
                login_performed = True

            repo_path = build_volume_repo_path(volume_name)

            try:
                cached_file = hf_hub_download(
                    repo_id=config.DATASET_NAME,
                    filename=repo_path,
                    repo_type="dataset",
                    local_files_only=True,
                )
                stats["cached_volumes"] += 1
            except Exception:
                cached_file = hf_hub_download(
                    repo_id=config.DATASET_NAME,
                    filename=repo_path,
                    repo_type="dataset",
                )
                stats["downloaded_volumes"] += 1

            available_studies.add(volume_name)
            volume_paths[volume_name] = cached_file
        except Exception as error:
            stats["failed_volumes"] += 1
            LOGGER.warning("Failed to download %s: %s", volume_name, error)

        if index % 100 == 0:
            LOGGER.info(
                "Checked %s/%s selected volumes",
                index,
                len(requested_volumes),
            )

    return available_studies, volume_paths, stats
