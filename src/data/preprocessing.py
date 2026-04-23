import re
import time
from random import Random

import config
from src.data.dataset import load_dataset_rows
from src.data.download import delete_volumes_from_hf_cache, download_selected_volumes
from src.data.images import create_slice_cache_for_records, load_cached_slice_cache, resolve_slice_cache_path
from src.utils.io import ensure_directory, write_json
from src.utils.logger import get_logger
import numpy as np


LOGGER = get_logger(__name__)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    text = WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    if len(text) <= config.MAX_TEXT_LENGTH_CHARS:
        return text
    return text[: config.MAX_TEXT_LENGTH_CHARS - 3].rstrip() + "..."


def normalize_anatomy(anatomy):
    """
    Нормализует анатомическую метку.

    Правила:
    - пустые значения и NULL отбрасываются;
    - метки вида lung/lung схлопываются в lung;
    - вложенные метки вида mediastinum/aorta отбрасываются;
    - остаются только верхнеуровневые анатомические классы.
    """
    if anatomy is None:
        return None

    anatomy = WHITESPACE_RE.sub(" ", str(anatomy).strip().lower())
    if not anatomy or anatomy == "null":
        return None

    parts = [part.strip() for part in anatomy.split("/") if part.strip()]
    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    if len(set(parts)) == 1:
        return parts[0]

    return None


def validate_required_fields(row):
    missing_fields = [field for field in config.EXPECTED_FIELDS if field not in row]
    if missing_fields:
        raise KeyError(f"Missing required dataset fields: {missing_fields}")


def collect_records(rows):
    """
    Собирает итоговые записи с учётом анатомической метки.

    Одна итоговая запись = один том + одна верхнеуровневая анатомическая метка.
    Если для пары (Volumename, Anatomy) есть несколько предложений,
    они объединяются в один текст.
    """
    grouped = {}
    seen_rows = set()
    scanned_studies = set()
    stats = {
        "raw_rows": 0,
        "scanned_studies": 0,
        "kept_rows": 0,
        "duplicate_rows": 0,
        "excluded_empty_text": 0,
        "excluded_missing_study_id": 0,
        "excluded_null_anatomy": 0,
        "excluded_nested_anatomy": 0,
        "excluded_anatomy_filter": 0,
        "records_before_sampling": 0,
    }

    for row in rows:
        stats["raw_rows"] += 1
        validate_required_fields(row)

        volume_name = str(row["Volumename"]).strip()
        if volume_name:
            scanned_studies.add(volume_name)

        raw_anatomy = row.get("Anatomy")
        anatomy = normalize_anatomy(raw_anatomy)
        text = normalize_text(str(row.get("Sentence", "")))

        if config.LOG_EVERY_N_ROWS and stats["raw_rows"] % config.LOG_EVERY_N_ROWS == 0:
            LOGGER.info(
                "Processed %s rows, found %s unique studies, kept %s rows",
                stats["raw_rows"],
                len(scanned_studies),
                stats["kept_rows"],
            )

        if not volume_name:
            stats["excluded_missing_study_id"] += 1
            continue

        if config.REQUIRE_NON_EMPTY_TEXT and len(text) < config.MIN_SENTENCE_LENGTH_CHARS:
            stats["excluded_empty_text"] += 1
            continue

        if raw_anatomy is None or str(raw_anatomy).strip() == "" or str(raw_anatomy).strip().lower() == "null":
            stats["excluded_null_anatomy"] += 1
            continue

        if anatomy is None:
            stats["excluded_nested_anatomy"] += 1
            continue

        if anatomy not in config.ALLOWED_ANATOMIES:
            stats["excluded_anatomy_filter"] += 1
            continue

        row_key = (volume_name, anatomy, text)
        if row_key in seen_rows:
            stats["duplicate_rows"] += 1
            continue

        seen_rows.add(row_key)
        stats["kept_rows"] += 1

        group_key = (volume_name, anatomy)
        if group_key not in grouped:
            grouped[group_key] = {
                "volume_name": volume_name,
                "anatomy": anatomy,
                "sentences": [],
            }

        grouped[group_key]["sentences"].append(text)

    stats["scanned_studies"] = len(scanned_studies)

    records = []
    for group in grouped.values():
        sentences = group["sentences"]
        if config.DROP_DUPLICATE_SENTENCES:
            sentences = list(dict.fromkeys(sentences))

        text = config.JOIN_SENTENCES_WITH.join(sentences).strip()
        if not text:
            continue

        volume_name = group["volume_name"]
        anatomy = group["anatomy"]

        records.append(
            {
                "id": f"{volume_name}::{anatomy}",
                "text": text,
                "anatomy": anatomy,
                "meta": {
                    "study_id": volume_name,
                    "volume_name": volume_name,
                    "anatomy": anatomy,
                    "modality": "CT",
                    "report_type": "anatomy_conditioned",
                    "num_sentences": len(sentences),
                },
            }
        )

    stats["records_before_sampling"] = len(records)
    return records, stats


def order_records_for_sampling(records):
    """
    Формирует детерминированный diversity-first порядок записей.

    Сначала чередует разные target-text, чтобы не забивать начало набора
    шаблонными формулировками.
    """
    grouped_by_text = {}
    for record in records:
        text_key = record["text"].strip().lower()
        grouped_by_text.setdefault(text_key, []).append(record)

    randomizer = Random(config.SEED)
    text_keys = list(grouped_by_text.keys())
    randomizer.shuffle(text_keys)
    for text_key in text_keys:
        randomizer.shuffle(grouped_by_text[text_key])

    sampled = []
    while len(sampled) < config.DATASET_SIZE:
        progress_made = False
        for text_key in text_keys:
            bucket = grouped_by_text[text_key]
            if not bucket:
                continue
            sampled.append(bucket.pop())
            progress_made = True
            if len(sampled) >= config.DATASET_SIZE:
                break
        if not progress_made:
            break

    return sampled


def filter_records_with_downloaded_volumes(records, available_studies):
    filtered = []
    for record in records:
        if record["meta"]["study_id"] in available_studies:
            filtered.append(record)
    return filtered


def collect_missing_download_exclusions(records, available_studies):
    excluded = []
    seen_studies = set()

    for record in records:
        study_id = record["meta"]["study_id"]
        if study_id in seen_studies or study_id in available_studies:
            continue

        seen_studies.add(study_id)
        excluded.append(
            {
                "study_id": study_id,
                "stage": "download",
                "reason": "volume_unavailable",
            }
        )

    return excluded


def assign_split_labels(records):
    """
    Делит записи по study_id, чтобы один и тот же том
    не попадал одновременно в train, val и test.
    """
    total_ratio = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    study_ids = []
    for record in records:
        study_id = record["meta"]["study_id"]
        if study_id not in study_ids:
            study_ids.append(study_id)

    shuffled_ids = study_ids[:]
    Random(config.SEED).shuffle(shuffled_ids)

    total = len(shuffled_ids)
    train_cutoff = int(total * config.TRAIN_RATIO)
    val_cutoff = train_cutoff + int(total * config.VAL_RATIO)

    split_map = {}
    for index, study_id in enumerate(shuffled_ids):
        if index < train_cutoff:
            split_map[study_id] = "train"
        elif index < val_cutoff:
            split_map[study_id] = "val"
        else:
            split_map[study_id] = "test"

    return split_map


def apply_splits(records, split_map):
    for record in records:
        record["meta"]["split"] = split_map[record["meta"]["study_id"]]
    return records


def build_anatomy_stats(records):
    anatomy_counts = {}
    for record in records:
        anatomy = record["anatomy"]
        anatomy_counts[anatomy] = anatomy_counts.get(anatomy, 0) + 1
    return dict(sorted(anatomy_counts.items(), key=lambda item: (-item[1], item[0])))


def build_split_anatomy_stats(records):
    stats = {"train": {}, "val": {}, "test": {}}

    for record in records:
        split_name = record["meta"]["split"]
        anatomy = record["anatomy"]
        split_counts = stats.setdefault(split_name, {})
        split_counts[anatomy] = split_counts.get(anatomy, 0) + 1

    for split_name, counts in stats.items():
        stats[split_name] = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    return stats


def build_text_stats(records):
    if not records:
        return {
            "num_records": 0,
            "unique_target_texts": 0,
            "repeated_target_text_ratio": 0.0,
            "most_common_target_text_share": 0.0,
            "text_length_chars": {"min": 0, "mean": 0.0, "max": 0},
            "sentences_per_record": {"min": 0, "mean": 0.0, "max": 0},
        }

    text_lengths = [len(record["text"]) for record in records]
    sentence_counts = [record["meta"].get("num_sentences", 0) for record in records]
    text_counts = {}
    for record in records:
        text_key = record["text"].strip().lower()
        text_counts[text_key] = text_counts.get(text_key, 0) + 1

    unique_target_texts = len(text_counts)
    repeated_target_text_ratio = 0.0
    most_common_target_text_share = 0.0
    if records:
        repeated_target_text_ratio = round((len(records) - unique_target_texts) / len(records), 6)
        most_common_target_text_share = round(max(text_counts.values()) / len(records), 6)

    return {
        "num_records": len(records),
        "unique_target_texts": unique_target_texts,
        "repeated_target_text_ratio": repeated_target_text_ratio,
        "most_common_target_text_share": most_common_target_text_share,
        "text_length_chars": {
            "min": min(text_lengths),
            "mean": round(sum(text_lengths) / len(text_lengths), 3),
            "max": max(text_lengths),
        },
        "sentences_per_record": {
            "min": min(sentence_counts),
            "mean": round(sum(sentence_counts) / len(sentence_counts), 3),
            "max": max(sentence_counts),
        },
    }


def build_slice_selection_stats(records):
    stats = {
        "num_records": len(records),
        "lung_focused_selection_count": 0,
        "fallback_selection_count": 0,
        "full_volume_range_count": 0,
        "start_zero_count": 0,
        "end_last_slice_count": 0,
        "expanded_range_count": 0,
        "duplicate_slice_index_count": 0,
    }

    for record in records:
        slice_info = record.get("meta", {}).get("slice_info") or {}
        selection_strategy = slice_info.get("selection_strategy")
        lung_range_start = slice_info.get("lung_range_start")
        lung_range_end = slice_info.get("lung_range_end")
        lung_range_is_full_volume = bool(slice_info.get("lung_range_is_full_volume", False))
        resampled_shape = slice_info.get("resampled_shape") or []
        depth = None
        if len(resampled_shape) >= 3:
            depth = int(resampled_shape[2])

        if selection_strategy == "lung_focused_uniform":
            stats["lung_focused_selection_count"] += 1
        elif selection_strategy:
            stats["fallback_selection_count"] += 1

        if lung_range_is_full_volume:
            stats["full_volume_range_count"] += 1
        if lung_range_start == 0:
            stats["start_zero_count"] += 1
        if depth is not None and lung_range_end == depth - 1:
            stats["end_last_slice_count"] += 1
        if bool(slice_info.get("lung_range_was_expanded", False)):
            stats["expanded_range_count"] += 1
        if bool(slice_info.get("has_duplicate_slice_indices", False)):
            stats["duplicate_slice_index_count"] += 1

    return stats


def validate_record_slice_caches(records):
    validation = {
        "validated_records": 0,
        "validated_slice_cache_files": 0,
        "validated_cached_slices": 0,
        "invalid_records": [],
    }

    for record in records:
        slice_cache_path = record.get("slice_cache_path")
        problems = []

        if not slice_cache_path:
            problems.append("missing_slice_cache_path")
        else:
            slice_cache_path = str(slice_cache_path)
            try:
                resolved_slice_cache_path = resolve_slice_cache_path(slice_cache_path)
                with np.load(resolved_slice_cache_path, allow_pickle=False) as payload:
                    slices_hu = np.asarray(payload["slices_hu"])

                if slices_hu.ndim != 3:
                    problems.append(f"unexpected_cache_rank:{slice_cache_path}:{slices_hu.ndim}")
                else:
                    if slices_hu.shape[0] != len(config.SLICE_POSITIONS):
                        problems.append(
                            f"expected_{len(config.SLICE_POSITIONS)}_cached_slices_got_{slices_hu.shape[0]}"
                        )
                    if slices_hu.shape[1] != slices_hu.shape[2]:
                        problems.append(
                            f"cache_not_square:{slice_cache_path}:{slices_hu.shape[1]}x{slices_hu.shape[2]}"
                        )
                if slices_hu.dtype != np.int16:
                    problems.append(f"unexpected_cache_dtype:{slice_cache_path}:{slices_hu.dtype}")
            except FileNotFoundError:
                problems.append(f"missing_slice_cache:{slice_cache_path}")
            except Exception as error:
                problems.append(f"unreadable_slice_cache:{slice_cache_path}:{error}")

        if problems:
            validation["invalid_records"].append(
                {
                    "id": record["id"],
                    "study_id": record["meta"]["study_id"],
                    "problems": problems,
                }
            )
            continue

        validation["validated_records"] += 1
        validation["validated_slice_cache_files"] += 1
        validation["validated_cached_slices"] += len(config.SLICE_POSITIONS)

    validation["invalid_records_count"] = len(validation["invalid_records"])
    return validation


def build_studies_manifest(records):
    manifest = {}

    for record in records:
        study_id = record["meta"]["study_id"]
        study_entry = manifest.setdefault(
            study_id,
            {
                "study_id": study_id,
                "volume_name": record["meta"]["volume_name"],
                "split": record["meta"]["split"],
                "slice_cache_path": record["slice_cache_path"],
                "num_slices": record["meta"]["num_slices"],
                "slice_info": record["meta"].get("slice_info"),
                "anatomies": [],
                "record_ids": [],
            },
        )
        study_entry["anatomies"].append(record["anatomy"])
        study_entry["record_ids"].append(record["id"])

    manifest_rows = []
    for study_id, study_entry in manifest.items():
        study_entry["anatomies"] = sorted(set(study_entry["anatomies"]))
        study_entry["record_ids"] = sorted(study_entry["record_ids"])
        manifest_rows.append(study_entry)

    manifest_rows.sort(key=lambda item: item["study_id"])
    return manifest_rows


def split_into_chunks(items, chunk_size):
    if chunk_size <= 0:
        raise ValueError("download_batch_size must be a positive integer.")

    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def merge_counter_stats(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value
    return total


def attach_cached_slice_caches_to_records(records):
    """
    Подхватывает уже готовые .npz-кэши до скачивания томов,
    чтобы preprocess можно было безопасно продолжать после остановки.
    """
    cached_records = {}
    pending_records = []
    cached_studies = set()
    cache_by_study = {}
    excluded_studies = []
    stats = {
        "requested_slice_cache_studies": 0,
        "created_slice_cache_studies": 0,
        "cached_slice_cache_studies": 0,
        "failed_slice_cache_studies": 0,
    }

    for record in records:
        study_id = record["meta"]["study_id"]

        if study_id not in cache_by_study:
            cache_by_study[study_id] = load_cached_slice_cache(study_id)

        cached_path, cached_metadata = cache_by_study[study_id]
        if cached_path is None:
            pending_records.append(record)
            continue

        if cached_metadata and bool(cached_metadata.get("has_duplicate_slice_indices", False)):
            stats["failed_slice_cache_studies"] += 1
            excluded_studies.append(
                {
                    "study_id": study_id,
                    "stage": "cached_slice_cache",
                    "reason": "duplicate_slice_indices",
                }
            )
            continue

        updated_record = dict(record)
        updated_record["slice_cache_path"] = cached_path
        updated_record["meta"] = dict(record["meta"])
        updated_record["meta"]["num_slices"] = len(config.SLICE_POSITIONS)
        if cached_metadata is not None:
            updated_record["meta"]["slice_info"] = cached_metadata
        updated_record.pop("image_paths", None)
        updated_record.pop("image_path", None)

        cached_records[record["id"]] = updated_record
        cached_studies.add(study_id)

    stats["requested_slice_cache_studies"] = len({record["meta"]["study_id"] for record in records})
    stats["cached_slice_cache_studies"] = len(cached_studies)
    return cached_records, pending_records, stats, excluded_studies


def summarize_records(records, stats, download_stats, slice_cache_stats, excluded_studies, started_at):
    split_counts = {"train": 0, "val": 0, "test": 0}
    study_ids = set()

    for record in records:
        split_counts[record["meta"]["split"]] += 1
        study_ids.add(record["meta"]["study_id"])

    slice_cache_validation = validate_record_slice_caches(records)
    split_anatomy_counts = build_split_anatomy_stats(records)
    text_stats = build_text_stats(records)
    slice_selection_stats = build_slice_selection_stats(records)

    return {
        "status": "ok",
        "raw_rows": stats["raw_rows"],
        "scanned_studies": stats["scanned_studies"],
        "kept_rows": stats["kept_rows"],
        "duplicate_rows": stats["duplicate_rows"],
        "excluded_empty_text": stats["excluded_empty_text"],
        "excluded_missing_study_id": stats["excluded_missing_study_id"],
        "excluded_null_anatomy": stats["excluded_null_anatomy"],
        "excluded_nested_anatomy": stats["excluded_nested_anatomy"],
        "excluded_anatomy_filter": stats["excluded_anatomy_filter"],
        "records_before_sampling": stats["records_before_sampling"],
        "records_after_sampling": len(records),
        "unique_studies": len(study_ids),
        "split_counts": split_counts,
        "anatomy_counts": build_anatomy_stats(records),
        "split_anatomy_counts": split_anatomy_counts,
        "text_stats": text_stats,
        "slice_selection_stats": slice_selection_stats,
        "requested_volumes": download_stats["requested_volumes"],
        "downloaded_volumes": download_stats["downloaded_volumes"],
        "cached_volumes": download_stats["cached_volumes"],
        "failed_volumes": download_stats["failed_volumes"],
        "download_batch_size": download_stats["download_batch_size"],
        "download_chunks": download_stats["download_chunks"],
        "requested_slice_cache_studies": slice_cache_stats["requested_slice_cache_studies"],
        "created_slice_cache_studies": slice_cache_stats["created_slice_cache_studies"],
        "cached_slice_cache_studies": slice_cache_stats["cached_slice_cache_studies"],
        "failed_slice_cache_studies": slice_cache_stats["failed_slice_cache_studies"],
        "excluded_studies_count": len(excluded_studies),
        "slice_cache_validation": slice_cache_validation,
        "duration_seconds": round(time.time() - started_at, 3),
    }


def preprocess_dataset(limit=None, download_batch_size=None):
    """
    Полная предварительная подготовка:
    загружает строки датасета, оставляет только верхнеуровневые анатомические метки,
    агрегирует текст по (том, анатомическая метка), ограничивает размер итогового набора,
    скачивает только нужные CT-файлы, извлекает 8 HU-срезов в .npz-кэш и сохраняет train/val/test.
    """
    started_at = time.time()
    LOGGER.info("Loading raw dataset rows")
    rows = load_dataset_rows(limit=limit)

    LOGGER.info("Filtering and aggregating anatomy-conditioned records")
    records, stats = collect_records(rows)

    LOGGER.info("Sampling final dataset size")
    ordered_records = order_records_for_sampling(records)
    target_record_count = len(ordered_records)
    if config.DATASET_SIZE:
        target_record_count = min(config.DATASET_SIZE, len(ordered_records))

    if target_record_count == 0:
        records = []
    else:
        LOGGER.info(
            "Will keep up to %s valid record(s) and backfill from reserve candidates if needed",
            target_record_count,
        )

    batch_size = config.DOWNLOAD_BATCH_SIZE if download_batch_size is None else download_batch_size
    LOGGER.info(
        "Processing CT volumes in chunks of %s study(s)",
        batch_size,
    )
    excluded_studies = []
    download_stats = {
        "requested_volumes": 0,
        "downloaded_volumes": 0,
        "cached_volumes": 0,
        "failed_volumes": 0,
        "download_batch_size": batch_size,
        "download_chunks": 0,
    }
    slice_cache_stats = {
        "requested_slice_cache_studies": 0,
        "created_slice_cache_studies": 0,
        "cached_slice_cache_studies": 0,
        "failed_slice_cache_studies": 0,
    }
    selected_records = []

    for chunk_index, chunk_records in enumerate(split_into_chunks(ordered_records, batch_size), start=1):
        if len(selected_records) >= target_record_count:
            break

        chunk_volume_paths = {}
        batch_selected_records = []

        LOGGER.info(
            "Chunk %s: evaluating %s study candidate(s)",
            chunk_index,
            len(chunk_records),
        )

        cached_records, pending_records, cached_slice_cache_stats, cached_exclusions = attach_cached_slice_caches_to_records(
            chunk_records
        )
        merge_counter_stats(slice_cache_stats, cached_slice_cache_stats)
        excluded_studies.extend(cached_exclusions)

        if cached_slice_cache_stats["cached_slice_cache_studies"] > 0:
            LOGGER.info(
                "Chunk %s: reused %s cached slice-cache study(s)",
                chunk_index,
                cached_slice_cache_stats["cached_slice_cache_studies"],
            )

        for record in chunk_records:
            cached_record = cached_records.get(record["id"])
            if cached_record is not None:
                batch_selected_records.append(cached_record)

        if pending_records and len(selected_records) + len(batch_selected_records) < target_record_count:
            try:
                chunk_available_studies, chunk_volume_paths, chunk_download_stats = download_selected_volumes(pending_records)
                merge_counter_stats(download_stats, chunk_download_stats)
                download_stats["download_chunks"] = chunk_index

                chunk_exclusions = collect_missing_download_exclusions(pending_records, chunk_available_studies)
                excluded_studies.extend(chunk_exclusions)
                pending_records = filter_records_with_downloaded_volumes(pending_records, chunk_available_studies)

                processed_records, chunk_slice_cache_stats, chunk_slice_cache_exclusions = create_slice_cache_for_records(
                    pending_records,
                    chunk_volume_paths,
                )
                merge_counter_stats(slice_cache_stats, chunk_slice_cache_stats)
                excluded_studies.extend(chunk_slice_cache_exclusions)
                processed_records_map = {record["id"]: record for record in processed_records}

                for record in chunk_records:
                    if record["id"] in cached_records:
                        continue
                    processed_record = processed_records_map.get(record["id"])
                    if processed_record is not None:
                        batch_selected_records.append(processed_record)
            finally:
                delete_volumes_from_hf_cache(chunk_volume_paths)

        selected_records.extend(batch_selected_records)

    records = selected_records[:target_record_count]

    LOGGER.info("Assigning train/val/test splits")
    split_map = assign_split_labels(records)
    records = apply_splits(records, split_map)
    summary = summarize_records(records, stats, download_stats, slice_cache_stats, excluded_studies, started_at)

    LOGGER.info("Writing preprocessing artifacts")
    ensure_directory(config.PREPROCESSING_DIR)

    metadata_path = config.PREPROCESSING_DIR / "preprocessing_summary.json"
    excluded_studies_path = config.PREPROCESSING_EXCLUDED_STUDIES_PATH
    studies_manifest_path = config.PREPROCESSING_DIR / "studies_manifest.json"

    summary["artifacts"] = {
        "train": str(config.PREPROCESSING_DIR / "train.json"),
        "val": str(config.PREPROCESSING_DIR / "val.json"),
        "test": str(config.PREPROCESSING_DIR / "test.json"),
        "summary": str(metadata_path),
        "excluded_studies": str(excluded_studies_path),
        "studies_manifest": str(studies_manifest_path),
        "slices_dir": str(config.PREPROCESSING_SLICES_DIR),
    }

    split_records_map = {}
    for split_name in ("train", "val", "test"):
        split_records_map[split_name] = [record for record in records if record["meta"]["split"] == split_name]

    studies_manifest = build_studies_manifest(records)

    write_json(metadata_path, summary)
    write_json(excluded_studies_path, excluded_studies)
    write_json(studies_manifest_path, studies_manifest)

    for split_name, split_records in split_records_map.items():
        write_json(config.PREPROCESSING_DIR / f"{split_name}.json", split_records)
    return summary
