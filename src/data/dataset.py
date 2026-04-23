import json
from pathlib import Path

import config
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.data.images import resolve_slice_cache_path


def maybe_login_to_huggingface():
    if not config.HF_ACCESS_TOKEN:
        return
    login(token=config.HF_ACCESS_TOKEN, add_to_git_credential=False)


def iterate_rows_from_json(path, limit=None):
    """
    Читает локальный JSON-файл с массивом записей.
    """
    if not path.exists():
        raise FileNotFoundError(f"RAW_DATASET_FILE does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list):
        raise ValueError("RAW_DATASET_FILE must contain a JSON array of dataset rows.")

    if limit is None:
        yield from rows
        return

    yield from rows[:limit]


def iterate_hf_rows(limit=None):
    """
    Построчно читает обе исходные части набора Hugging Face:
    сначала train, затем validation.
    """
    yielded_rows = 0

    for split_name in ["train", "validation"]:
        rows = load_dataset(
            config.DATASET_NAME,
            config.DATASET_CONFIG,
            split=split_name,
            streaming=True,
        )

        for row in rows:
            yield row
            yielded_rows += 1

            if limit is not None and yielded_rows >= limit:
                return


def load_dataset_rows(limit=None):
    """
    Возвращает итератор по строкам датасета.

    Для Hugging Face используется потоковое чтение, поэтому строки читаются по мере
    обработки, а не загружаются все сразу в память.
    """
    if config.RAW_DATASET_FILE is not None:
        return iterate_rows_from_json(config.RAW_DATASET_FILE, limit=limit)

    try:
        maybe_login_to_huggingface()
    except Exception as error:
        raise RuntimeError(
            "Failed to load dataset. "
            "Set RAW_DATASET_FILE in config.py or enable network access."
        ) from error

    return iterate_hf_rows(limit=limit)


def load_records_from_json(path):
    """
    Читает локальную часть подготовленного набора и возвращает список записей.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Dataset split must contain a JSON array: {path}")

    return records


ANATOMY_PROMPT_LABELS = {
    "lung": "lungs",
    "mediastinum": "mediastinum",
    "pleura": "pleura",
}


def build_prompt(anatomy, template):
    anatomy_phrase = ANATOMY_PROMPT_LABELS.get(anatomy, anatomy)
    return template.format(anatomy=anatomy, anatomy_phrase=anatomy_phrase)


def anatomy_to_id(anatomy):
    try:
        return config.ALLOWED_ANATOMIES.index(anatomy)
    except ValueError as error:
        raise ValueError(f"Unsupported anatomy label: {anatomy}") from error


def get_slice_positions(record, num_slices):
    slice_info = record.get("meta", {}).get("slice_info") or {}
    positions = slice_info.get("slice_positions")

    if positions is None:
        positions = list(config.SLICE_POSITIONS)

    if len(positions) != num_slices:
        raise ValueError(
            f"Record {record.get('id')} must contain {num_slices} slice positions, got {len(positions)}."
        )

    return [float(value) for value in positions]


def build_window_channel(slice_hu, window_min, window_max):
    clipped = np.clip(slice_hu, window_min, window_max)
    scaled = (clipped - window_min) / (window_max - window_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def build_multichannel_slice(slice_hu):
    channels = [build_window_channel(slice_hu, window_min, window_max) for window_min, window_max in config.MULTI_WINDOW_RANGES]
    return np.stack(channels, axis=-1)


def load_slice_cache(path):
    with np.load(path, allow_pickle=False) as payload:
        slices_hu = np.asarray(payload["slices_hu"])

    if slices_hu.ndim != 3:
        raise ValueError(f"Slice cache must have shape [num_slices, H, W], got {slices_hu.shape} for {path}")

    if slices_hu.dtype != np.int16:
        raise ValueError(f"Slice cache must store int16 HU slices, got {slices_hu.dtype} for {path}")

    return slices_hu


def resize_multichannel_slice(image_array, image_size):
    image = Image.fromarray(image_array, mode="RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def normalize_rgb_tensor(array):
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).to(dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def build_image_tensor_from_hu_slice(slice_hu, image_size):
    multichannel_slice = build_multichannel_slice(slice_hu)
    resized = resize_multichannel_slice(multichannel_slice, image_size=image_size)
    return normalize_rgb_tensor(resized)


class CTReportDataset(Dataset):
    def __init__(self, json_path, prompt_template, num_slices=8, image_size=224, max_records=None):
        self.json_path = Path(json_path)
        self.prompt_template = prompt_template
        self.num_slices = num_slices
        self.image_size = image_size
        self.records = load_records_from_json(self.json_path)

        if max_records is not None:
            self.records = self.records[:max_records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        slice_cache_path = record.get("slice_cache_path")
        if not slice_cache_path:
            raise ValueError(f"Record {record.get('id')} must contain slice_cache_path.")

        resolved_slice_cache_path = resolve_slice_cache_path(slice_cache_path)
        slices_hu = load_slice_cache(resolved_slice_cache_path)

        if slices_hu.shape[0] != self.num_slices:
            raise ValueError(
                f"Record {record.get('id')} must contain {self.num_slices} cached slices, got {slices_hu.shape[0]}."
            )

        images = []
        for slice_hu in slices_hu:
            image_tensor = build_image_tensor_from_hu_slice(slice_hu, image_size=self.image_size)
            if image_tensor.shape != (3, self.image_size, self.image_size):
                raise ValueError(
                    f"Unexpected tensor shape for {resolved_slice_cache_path}: {tuple(image_tensor.shape)}"
                )
            images.append(image_tensor)

        return {
            "id": record["id"],
            "images": torch.stack(images, dim=0),
            "text": record["text"],
            "anatomy": record["anatomy"],
            "anatomy_id": anatomy_to_id(record["anatomy"]),
            "slice_positions": torch.tensor(get_slice_positions(record, self.num_slices), dtype=torch.float32),
            "prompt": build_prompt(record["anatomy"], self.prompt_template),
            "meta": record.get("meta", {}),
        }
