from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import Orientation, Spacing

import config
from src.utils.io import ensure_directory
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def volume_name_to_stem(volume_name):
    if volume_name.endswith(".nii.gz"):
        return volume_name[:-7]

    return Path(volume_name).stem


def build_slice_cache_path(volume_name):
    stem = volume_name_to_stem(volume_name)
    return config.PREPROCESSING_SLICES_DIR / f"{stem}.npz"


def to_project_relative_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_slice_cache_path(slice_cache_path):
    path = Path(slice_cache_path)
    if path.exists():
        return path

    if not path.is_absolute():
        project_relative = config.PROJECT_ROOT / path
        if project_relative.exists():
            return project_relative

    normalized = path.as_posix()
    marker = "data/preprocessing/slices/"
    if marker in normalized:
        relative_suffix = normalized.split(marker, 1)[1]
        remapped = config.PREPROCESSING_SLICES_DIR / relative_suffix
        if remapped.exists():
            return remapped

    raise FileNotFoundError(
        "Slice cache file does not exist and could not be remapped to the current project: "
        f"{slice_cache_path}"
    )


def _parse_optional_scalar(raw_value, cast):
    value = cast(np.asarray(raw_value).item())
    if value < 0:
        return None
    return value


def _parse_slice_cache_metadata(payload):
    slice_lung_fractions = payload.get("slice_lung_fractions")
    if slice_lung_fractions is None:
        slice_lung_fractions = []
    slice_lung_fractions_smoothed = payload.get("slice_lung_fractions_smoothed")
    if slice_lung_fractions_smoothed is None:
        slice_lung_fractions_smoothed = []

    return {
        "slice_axis": int(np.asarray(payload["slice_axis"]).item()),
        "slice_indices": [int(value) for value in np.asarray(payload["slice_indices"]).tolist()],
        "slice_positions": [float(value) for value in np.asarray(payload["slice_positions"]).tolist()],
        "selection_strategy": str(np.asarray(payload["selection_strategy"]).item()),
        "lung_range_start": _parse_optional_scalar(payload["lung_range_start"], int),
        "lung_range_end": _parse_optional_scalar(payload["lung_range_end"], int),
        "lung_range_length": int(np.asarray(payload.get("lung_range_length", 0)).item()),
        "lung_range_was_expanded": bool(np.asarray(payload.get("lung_range_was_expanded", False)).item()),
        "lung_range_is_full_volume": bool(np.asarray(payload.get("lung_range_is_full_volume", False)).item()),
        "slice_index_unique_count": int(np.asarray(payload.get("slice_index_unique_count", 0)).item()),
        "has_duplicate_slice_indices": bool(np.asarray(payload.get("has_duplicate_slice_indices", False)).item()),
        "slice_lung_fractions": [float(value) for value in np.asarray(slice_lung_fractions).tolist()],
        "slice_lung_fractions_smoothed": [
            float(value) for value in np.asarray(slice_lung_fractions_smoothed).tolist()
        ],
        "lung_fraction_threshold": float(np.asarray(payload.get("lung_fraction_threshold", 0.0)).item()),
        "original_shape": [int(value) for value in np.asarray(payload["original_shape"]).tolist()],
        "original_spacing": [float(value) for value in np.asarray(payload["original_spacing"]).tolist()],
        "canonical_shape": [int(value) for value in np.asarray(payload["canonical_shape"]).tolist()],
        "canonical_spacing": [float(value) for value in np.asarray(payload["canonical_spacing"]).tolist()],
        "resampled_shape": [int(value) for value in np.asarray(payload["resampled_shape"]).tolist()],
        "target_spacing": [float(value) for value in np.asarray(payload["target_spacing"]).tolist()],
        "padded_size": int(np.asarray(payload["padded_size"]).item()),
        "hu_clip_min": int(np.asarray(payload.get("hu_clip_min", config.HU_SANITIZE_MIN)).item()),
        "hu_clip_max": int(np.asarray(payload.get("hu_clip_max", config.HU_SANITIZE_MAX)).item()),
        "storage_format": str(np.asarray(payload["storage_format"]).item()),
        "storage_dtype": str(np.asarray(payload["storage_dtype"]).item()),
    }


def load_cached_slice_cache(volume_name):
    """
    Возвращает уже готовый .npz-кэш с 8 HU-срезами, если он существует и валиден.
    """
    cache_path = build_slice_cache_path(volume_name)

    if not cache_path.exists():
        return None, None

    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            slices_hu = np.asarray(payload["slices_hu"])
            if slices_hu.ndim != 3 or slices_hu.shape[0] != len(config.SLICE_POSITIONS):
                return None, None
            if slices_hu.dtype != np.int16:
                return None, None
            metadata = _parse_slice_cache_metadata(payload)
    except Exception:
        return None, None

    return to_project_relative_path(cache_path), metadata


def validate_spacing(spacing, volume_name):
    if len(spacing) != 3:
        raise ValueError(f"Expected 3 spacing values, got {spacing} for {volume_name}")

    normalized = []
    for value in spacing:
        value = float(value)
        if not np.isfinite(value) or value < config.MIN_VALID_VOXEL_SPACING:
            raise ValueError(f"Invalid voxel spacing {spacing} for {volume_name}")
        normalized.append(value)

    return tuple(normalized)


def extract_slice_indices(depth):
    indices = []

    for position in config.SLICE_POSITIONS:
        index = int(round((depth - 1) * position))
        index = max(0, min(depth - 1, index))
        indices.append(index)

    return indices


def estimate_lung_slice_mask(volume_hu):
    depth = volume_hu.shape[-1]
    slice_fractions = np.zeros(depth, dtype=np.float32)

    for slice_index in range(depth):
        slice_hu = volume_hu[:, :, slice_index]
        foreground_mask = slice_hu > config.LUNG_FOREGROUND_HU_THRESHOLD
        foreground_rows = np.flatnonzero(foreground_mask.any(axis=1))
        foreground_cols = np.flatnonzero(foreground_mask.any(axis=0))

        if foreground_rows.size == 0 or foreground_cols.size == 0:
            continue

        row_start = int(foreground_rows[0])
        row_end = int(foreground_rows[-1]) + 1
        col_start = int(foreground_cols[0])
        col_end = int(foreground_cols[-1]) + 1
        body_crop = slice_hu[row_start:row_end, col_start:col_end]
        if body_crop.size == 0:
            continue

        lung_like_mask = (body_crop >= config.LUNG_HU_MIN) & (body_crop <= config.LUNG_HU_MAX)
        slice_fractions[slice_index] = float(lung_like_mask.mean())

    return slice_fractions >= config.LUNG_MIN_FRACTION, slice_fractions


def smooth_slice_fractions(slice_fractions):
    window = max(1, int(config.LUNG_FRACTION_SMOOTHING_WINDOW))
    if window <= 1 or slice_fractions.size <= 1:
        return slice_fractions.astype(np.float32, copy=True)

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(slice_fractions, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32, copy=False)


def find_longest_true_run(mask):
    best_start = None
    best_end = None
    current_start = None

    for index, is_true in enumerate(mask.tolist()):
        if is_true and current_start is None:
            current_start = index
        elif not is_true and current_start is not None:
            current_end = index - 1
            if best_start is None or (current_end - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = current_end
            current_start = None

    if current_start is not None:
        current_end = len(mask) - 1
        if best_start is None or (current_end - current_start) > (best_end - best_start):
            best_start = current_start
            best_end = current_end

    return best_start, best_end


def find_peak_centered_mass_interval(slice_fractions, coverage):
    positive_fractions = np.clip(np.asarray(slice_fractions, dtype=np.float32), a_min=0.0, a_max=None)
    total_mass = float(positive_fractions.sum())
    if total_mass <= 0.0:
        return None, None

    target_mass = total_mass * float(np.clip(coverage, 0.0, 1.0))
    peak_index = int(np.argmax(positive_fractions))
    left_index = peak_index
    right_index = peak_index
    accumulated_mass = float(positive_fractions[peak_index])

    while accumulated_mass < target_mass and (left_index > 0 or right_index < len(positive_fractions) - 1):
        left_candidate = float(positive_fractions[left_index - 1]) if left_index > 0 else -1.0
        right_candidate = float(positive_fractions[right_index + 1]) if right_index < len(positive_fractions) - 1 else -1.0

        if right_candidate >= left_candidate and right_index < len(positive_fractions) - 1:
            right_index += 1
            accumulated_mass += float(positive_fractions[right_index])
        elif left_index > 0:
            left_index -= 1
            accumulated_mass += float(positive_fractions[left_index])
        else:
            break

    return left_index, right_index


def estimate_lung_range(volume_hu):
    depth = volume_hu.shape[-1]
    lung_positive_mask, slice_fractions = estimate_lung_slice_mask(volume_hu)
    positive_indices = np.flatnonzero(lung_positive_mask)

    if positive_indices.size == 0:
        return None, None, slice_fractions, slice_fractions, config.LUNG_MIN_FRACTION

    smoothed_fractions = smooth_slice_fractions(slice_fractions)
    peak_fraction = float(smoothed_fractions.max(initial=0.0))
    adaptive_threshold = max(
        float(config.LUNG_MIN_FRACTION),
        peak_fraction * float(config.LUNG_PEAK_RELATIVE_THRESHOLD),
    )
    adaptive_mask = smoothed_fractions >= adaptive_threshold
    adaptive_indices = np.flatnonzero(adaptive_mask)

    if adaptive_indices.size == 0:
        adaptive_mask = lung_positive_mask

    run_start, run_end = find_longest_true_run(adaptive_mask)
    if run_start is None or run_end is None:
        return None, None, slice_fractions, smoothed_fractions, adaptive_threshold

    mass_start, mass_end = find_peak_centered_mass_interval(
        smoothed_fractions,
        coverage=config.LUNG_PEAK_MASS_COVERAGE,
    )
    if mass_start is not None and mass_end is not None:
        refined_start = max(int(run_start), int(mass_start))
        refined_end = min(int(run_end), int(mass_end))
        if refined_start <= refined_end:
            run_start, run_end = refined_start, refined_end

    start_idx = max(0, int(run_start) - config.LUNG_RANGE_PADDING_SLICES)
    end_idx = min(depth - 1, int(run_end) + config.LUNG_RANGE_PADDING_SLICES)
    return start_idx, end_idx, slice_fractions, smoothed_fractions, adaptive_threshold


def expand_range_to_min_length(start_idx, end_idx, depth, min_length):
    desired_length = min(max(1, int(min_length)), depth)
    current_length = int(end_idx) - int(start_idx) + 1
    if current_length >= desired_length:
        return int(start_idx), int(end_idx), False

    deficit = desired_length - current_length
    expand_left = deficit // 2
    expand_right = deficit - expand_left
    start_idx = int(start_idx) - expand_left
    end_idx = int(end_idx) + expand_right

    if start_idx < 0:
        end_idx = min(depth - 1, end_idx - start_idx)
        start_idx = 0

    if end_idx > depth - 1:
        overflow = end_idx - (depth - 1)
        start_idx = max(0, start_idx - overflow)
        end_idx = depth - 1

    current_length = end_idx - start_idx + 1
    if current_length < desired_length and depth >= desired_length:
        end_idx = min(depth - 1, start_idx + desired_length - 1)
        start_idx = max(0, end_idx - desired_length + 1)

    return int(start_idx), int(end_idx), True


def select_slice_indices_from_range(start_idx, end_idx, num_slices):
    available_indices = np.arange(int(start_idx), int(end_idx) + 1, dtype=np.int32)
    if available_indices.size == 0:
        return []

    if available_indices.size < num_slices:
        positions = np.linspace(int(start_idx), int(end_idx), num=num_slices)
        return np.rint(positions).astype(int).tolist()

    candidate_positions = np.linspace(0, available_indices.size - 1, num=num_slices)
    selected_offsets = []
    last_offset = -1

    for position_index, raw_position in enumerate(candidate_positions):
        proposed_offset = int(round(float(raw_position)))
        min_allowed = last_offset + 1
        max_allowed = available_indices.size - (num_slices - position_index)
        clamped_offset = min(max(proposed_offset, min_allowed), max_allowed)
        selected_offsets.append(clamped_offset)
        last_offset = clamped_offset

    return available_indices[selected_offsets].astype(int).tolist()


def extract_lung_focused_slice_indices(volume_hu, num_slices=8):
    depth = volume_hu.shape[-1]
    if depth <= 0:
        raise ValueError("Volume depth must be positive.")

    lung_range_start, lung_range_end, slice_fractions, smoothed_fractions, adaptive_threshold = estimate_lung_range(
        volume_hu
    )
    if lung_range_start is None or lung_range_end is None or lung_range_end < lung_range_start:
        fallback_indices = extract_slice_indices(depth)
        fallback_positions = [round(index / max(depth - 1, 1), 6) for index in fallback_indices]
        return {
            "slice_indices": fallback_indices,
            "slice_positions": fallback_positions,
            "lung_range_start": None,
            "lung_range_end": None,
            "lung_range_length": 0,
            "lung_range_was_expanded": False,
            "lung_range_is_full_volume": False,
            "slice_index_unique_count": len(set(fallback_indices)),
            "has_duplicate_slice_indices": len(set(fallback_indices)) != len(fallback_indices),
            "slice_lung_fractions": [round(float(value), 6) for value in slice_fractions.tolist()],
            "slice_lung_fractions_smoothed": [round(float(value), 6) for value in smoothed_fractions.tolist()],
            "lung_fraction_threshold": round(float(adaptive_threshold), 6),
            "selection_strategy": "fallback_uniform_full_volume",
        }

    minimum_range_length = max(int(num_slices), int(config.LUNG_MIN_RANGE_SLICES))
    lung_range_start, lung_range_end, range_was_expanded = expand_range_to_min_length(
        lung_range_start,
        lung_range_end,
        depth,
        minimum_range_length,
    )
    slice_indices = select_slice_indices_from_range(lung_range_start, lung_range_end, num_slices=num_slices)
    slice_positions = [round(index / max(depth - 1, 1), 6) for index in slice_indices]
    unique_slice_count = len(set(slice_indices))

    return {
        "slice_indices": slice_indices,
        "slice_positions": slice_positions,
        "lung_range_start": lung_range_start,
        "lung_range_end": lung_range_end,
        "lung_range_length": int(lung_range_end - lung_range_start + 1),
        "lung_range_was_expanded": bool(range_was_expanded),
        "lung_range_is_full_volume": lung_range_start == 0 and lung_range_end == depth - 1,
        "slice_index_unique_count": unique_slice_count,
        "has_duplicate_slice_indices": unique_slice_count != len(slice_indices),
        "slice_lung_fractions": [round(float(value), 6) for value in slice_fractions.tolist()],
        "slice_lung_fractions_smoothed": [round(float(value), 6) for value in smoothed_fractions.tolist()],
        "lung_fraction_threshold": round(float(adaptive_threshold), 6),
        "selection_strategy": "lung_focused_uniform",
    }


def pad_slice_to_square(slice_hu):
    rotated = np.rot90(slice_hu)
    height, width = rotated.shape
    square_size = max(height, width)

    pad_height = square_size - height
    pad_width = square_size - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = np.pad(
        rotated,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=config.SLICE_PAD_HU_VALUE,
    )
    return padded, square_size


def convert_slice_to_int16(slice_hu):
    clipped = np.clip(np.rint(slice_hu), np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return clipped.astype(np.int16)


def sanitize_hu_volume(volume):
    sanitized = np.nan_to_num(
        volume,
        nan=float(config.SLICE_PAD_HU_VALUE),
        posinf=float(config.HU_SANITIZE_MAX),
        neginf=float(config.HU_SANITIZE_MIN),
    )
    sanitized = np.clip(sanitized, float(config.HU_SANITIZE_MIN), float(config.HU_SANITIZE_MAX))
    return sanitized.astype(np.float32, copy=False)


def load_and_prepare_volume(volume_path, volume_name):
    image = nib.load(str(volume_path))
    original_spacing = validate_spacing(image.header.get_zooms()[:3], volume_name)
    original_shape = tuple(int(value) for value in image.shape[:3])

    volume = image.get_fdata(dtype=np.float32)
    volume = np.asarray(volume)
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape} for {volume_name}")

    tensor = torch.as_tensor(volume[None], dtype=torch.float32)
    affine = np.asarray(image.affine, dtype=np.float64)

    prepared = MetaTensor(tensor, affine=affine)
    if config.USE_CANONICAL_ORIENTATION:
        prepared = Orientation(axcodes="RAS")(prepared)

    canonical_tensor = prepared.as_tensor() if hasattr(prepared, "as_tensor") else torch.as_tensor(prepared)
    canonical_volume = canonical_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    canonical_affine = np.asarray(prepared.affine, dtype=np.float64)
    canonical_spacing = validate_spacing(nib.affines.voxel_sizes(canonical_affine), volume_name)

    prepared = Spacing(
        pixdim=tuple(float(value) for value in config.TARGET_VOXEL_SPACING),
        mode="bilinear",
        padding_mode="border",
        diagonal=False,
        recompute_affine=True,
    )(prepared)

    prepared_tensor = prepared.as_tensor() if hasattr(prepared, "as_tensor") else torch.as_tensor(prepared)
    prepared_volume = prepared_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    prepared_volume = sanitize_hu_volume(prepared_volume)

    if prepared_volume.ndim != 3:
        raise ValueError(f"Expected transformed 3D volume, got shape {prepared_volume.shape} for {volume_name}")

    target_spacing = tuple(float(value) for value in config.TARGET_VOXEL_SPACING)

    return {
        "volume": prepared_volume,
        "original_shape": original_shape,
        "original_spacing": original_spacing,
        "canonical_shape": tuple(int(value) for value in canonical_volume.shape),
        "canonical_spacing": canonical_spacing,
        "resampled_shape": tuple(int(value) for value in prepared_volume.shape),
        "target_spacing": target_spacing,
    }


def extract_slice_cache(volume_path, volume_name):
    """
    Извлекает 8 lung-focused аксиальных HU-срезов из NIfTI-объёма
    и сохраняет их как один .npz-кэш на исследование.
    """
    cache_path = build_slice_cache_path(volume_name)

    cached_slice_cache_path, cached_metadata = load_cached_slice_cache(volume_name)
    if cached_slice_cache_path is not None:
        return cached_slice_cache_path, False, cached_metadata

    ensure_directory(config.PREPROCESSING_SLICES_DIR)

    prepared_volume = load_and_prepare_volume(volume_path, volume_name)
    volume = prepared_volume["volume"]
    depth = volume.shape[-1]
    if depth <= 0:
        raise ValueError(f"Volume has invalid depth for {volume_name}")

    slice_selection = extract_lung_focused_slice_indices(volume, num_slices=len(config.SLICE_POSITIONS))
    if slice_selection["has_duplicate_slice_indices"]:
        raise ValueError(
            "duplicate_slice_indices_after_range_expansion:"
            f"{slice_selection['slice_indices']}"
        )
    slice_arrays = []
    padded_size = None

    for slice_index in slice_selection["slice_indices"]:
        slice_hu = volume[:, :, slice_index]
        padded_slice_hu, current_padded_size = pad_slice_to_square(slice_hu)
        if padded_size is None:
            padded_size = current_padded_size
        elif padded_size != current_padded_size:
            raise ValueError(
                f"Inconsistent padded size across slices for {volume_name}: {padded_size} != {current_padded_size}"
            )
        slice_arrays.append(convert_slice_to_int16(padded_slice_hu))

    slices_hu = np.stack(slice_arrays, axis=0)
    if slices_hu.shape[0] != len(config.SLICE_POSITIONS):
        raise ValueError(
            f"Expected {len(config.SLICE_POSITIONS)} cached slices for {volume_name}, got {slices_hu.shape[0]}"
        )

    lung_range_start = -1 if slice_selection["lung_range_start"] is None else slice_selection["lung_range_start"]
    lung_range_end = -1 if slice_selection["lung_range_end"] is None else slice_selection["lung_range_end"]

    cache_payload = {
        "slice_axis": np.int32(2),
        "slice_indices": np.asarray(slice_selection["slice_indices"], dtype=np.int32),
        "slice_positions": np.asarray(slice_selection["slice_positions"], dtype=np.float32),
        "selection_strategy": np.asarray(slice_selection["selection_strategy"]),
        "lung_range_start": np.int32(lung_range_start),
        "lung_range_end": np.int32(lung_range_end),
        "lung_range_length": np.int32(slice_selection["lung_range_length"]),
        "lung_range_was_expanded": np.asarray(slice_selection["lung_range_was_expanded"]),
        "lung_range_is_full_volume": np.asarray(slice_selection["lung_range_is_full_volume"]),
        "slice_index_unique_count": np.int32(slice_selection["slice_index_unique_count"]),
        "has_duplicate_slice_indices": np.asarray(slice_selection["has_duplicate_slice_indices"]),
        "slice_lung_fractions": np.asarray(slice_selection["slice_lung_fractions"], dtype=np.float32),
        "slice_lung_fractions_smoothed": np.asarray(slice_selection["slice_lung_fractions_smoothed"], dtype=np.float32),
        "lung_fraction_threshold": np.float32(slice_selection["lung_fraction_threshold"]),
        "original_shape": np.asarray(prepared_volume["original_shape"], dtype=np.int32),
        "original_spacing": np.asarray(prepared_volume["original_spacing"], dtype=np.float32),
        "canonical_shape": np.asarray(prepared_volume["canonical_shape"], dtype=np.int32),
        "canonical_spacing": np.asarray(prepared_volume["canonical_spacing"], dtype=np.float32),
        "resampled_shape": np.asarray(prepared_volume["resampled_shape"], dtype=np.int32),
        "target_spacing": np.asarray(prepared_volume["target_spacing"], dtype=np.float32),
        "padded_size": np.int32(padded_size),
        "hu_clip_min": np.int32(config.HU_SANITIZE_MIN),
        "hu_clip_max": np.int32(config.HU_SANITIZE_MAX),
        "storage_format": np.asarray("npz_int16_hu"),
        "storage_dtype": np.asarray("int16"),
    }
    np.savez_compressed(cache_path, slices_hu=slices_hu, **cache_payload)

    return to_project_relative_path(cache_path), True, _parse_slice_cache_metadata(cache_payload)


def create_slice_cache_for_records(records, volume_paths):
    """
    Создаёт .npz-кэш HU-срезов для всех томов, вошедших в итоговый набор.

    Исходные NIfTI-файлы читаются прямо из локального кэша HF без сохранения
    копии в папку проекта.
    """
    requested_studies = []
    seen_studies = set()

    for record in records:
        study_id = record["meta"]["study_id"]
        if study_id in seen_studies:
            continue

        seen_studies.add(study_id)
        requested_studies.append(study_id)

    cache_paths_by_study = {}
    metadata_by_study = {}
    excluded_studies = []
    stats = {
        "requested_slice_cache_studies": len(requested_studies),
        "created_slice_cache_studies": 0,
        "cached_slice_cache_studies": 0,
        "failed_slice_cache_studies": 0,
    }

    for study_id in requested_studies:
        volume_path = volume_paths.get(study_id)

        if not volume_path:
            stats["failed_slice_cache_studies"] += 1
            LOGGER.warning("Volume path not found for slice-cache extraction: %s", study_id)
            excluded_studies.append(
                {
                    "study_id": study_id,
                    "stage": "slice_cache_extraction",
                    "reason": "missing_volume_path",
                }
            )
            continue

        try:
            slice_cache_path, created_now, slice_metadata = extract_slice_cache(volume_path, study_id)
            cache_paths_by_study[study_id] = slice_cache_path
            if slice_metadata is not None:
                metadata_by_study[study_id] = slice_metadata

            if created_now:
                stats["created_slice_cache_studies"] += 1
            else:
                stats["cached_slice_cache_studies"] += 1
        except Exception as error:
            stats["failed_slice_cache_studies"] += 1
            LOGGER.warning("Failed to extract slice cache for %s: %s", study_id, error)
            excluded_studies.append(
                {
                    "study_id": study_id,
                    "stage": "slice_cache_extraction",
                    "reason": str(error),
                }
            )

    filtered_records = []

    for record in records:
        study_id = record["meta"]["study_id"]
        if study_id not in cache_paths_by_study:
            continue

        updated_record = dict(record)
        updated_record["slice_cache_path"] = cache_paths_by_study[study_id]
        updated_record["meta"] = dict(record["meta"])
        updated_record["meta"]["num_slices"] = len(config.SLICE_POSITIONS)

        if study_id in metadata_by_study:
            updated_record["meta"]["slice_info"] = metadata_by_study[study_id]

        if "image_paths" in updated_record:
            del updated_record["image_paths"]
        if "image_path" in updated_record:
            del updated_record["image_path"]

        filtered_records.append(updated_record)

    return filtered_records, stats, excluded_studies
