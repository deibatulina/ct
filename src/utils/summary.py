def format_int(value):
    return f"{value:,}".replace(",", " ")



def format_top_anatomy(anatomy_counts, top_n=8):
    lines = []
    items = list(anatomy_counts.items())[:top_n]

    for anatomy, count in items:
        lines.append(f"  - {anatomy}: {format_int(count)}")

    return lines



def format_preprocessing_summary(summary):
    lines = []
    lines.append("Preprocessing Summary")
    lines.append("")
    lines.append("Итог")
    lines.append(f"  - статус: {summary['status']}")
    lines.append(f"  - время: {summary['duration_seconds']} сек")
    lines.append(f"  - просмотрено строк: {format_int(summary['raw_rows'])}")
    lines.append(f"  - найдено исследований: {format_int(summary['scanned_studies'])}")
    lines.append(f"  - записей до sampling: {format_int(summary['records_before_sampling'])}")
    lines.append(f"  - записей после sampling: {format_int(summary['records_after_sampling'])}")
    lines.append(f"  - уникальных исследований в финальном наборе: {format_int(summary['unique_studies'])}")

    lines.append("")
    lines.append("Фильтрация")
    lines.append(f"  - оставлено строк: {format_int(summary['kept_rows'])}")
    lines.append(f"  - дубликаты: {format_int(summary['duplicate_rows'])}")
    lines.append(f"  - исключено NULL anatomy: {format_int(summary['excluded_null_anatomy'])}")
    lines.append(f"  - исключено nested anatomy: {format_int(summary['excluded_nested_anatomy'])}")
    lines.append(f"  - исключено anatomy filter: {format_int(summary['excluded_anatomy_filter'])}")
    lines.append(f"  - исключено пустых текстов: {format_int(summary['excluded_empty_text'])}")
    lines.append(f"  - исключено пустых study_id: {format_int(summary['excluded_missing_study_id'])}")

    lines.append("")
    lines.append("Сплиты")
    lines.append(f"  - train: {format_int(summary['split_counts']['train'])}")
    lines.append(f"  - val: {format_int(summary['split_counts']['val'])}")
    lines.append(f"  - test: {format_int(summary['split_counts']['test'])}")

    lines.append("")
    lines.append("Volumes")
    lines.append(f"  - запрошено volumes: {format_int(summary['requested_volumes'])}")
    lines.append(f"  - размер чанка скачивания: {format_int(summary['download_batch_size'])}")
    lines.append(f"  - обработано чанков: {format_int(summary['download_chunks'])}")
    lines.append(f"  - скачано volumes: {format_int(summary['downloaded_volumes'])}")
    lines.append(f"  - уже было локально: {format_int(summary['cached_volumes'])}")
    lines.append(f"  - ошибок скачивания: {format_int(summary['failed_volumes'])}")

    lines.append("")
    lines.append("Slice Cache")
    lines.append(f"  - кэшей запрошено: {format_int(summary['requested_slice_cache_studies'])}")
    lines.append(f"  - создано кэшей: {format_int(summary['created_slice_cache_studies'])}")
    lines.append(f"  - уже было готово: {format_int(summary['cached_slice_cache_studies'])}")
    lines.append(f"  - ошибок извлечения: {format_int(summary['failed_slice_cache_studies'])}")
    lines.append(f"  - исключено исследований: {format_int(summary['excluded_studies_count'])}")
    lines.append(
        f"  - валидных записей кэша: {format_int(summary['slice_cache_validation']['validated_records'])}"
    )
    lines.append(
        f"  - проверено .npz-файлов: {format_int(summary['slice_cache_validation']['validated_slice_cache_files'])}"
    )
    lines.append(
        f"  - проверено кэшированных срезов: {format_int(summary['slice_cache_validation']['validated_cached_slices'])}"
    )
    lines.append(
        f"  - невалидных записей: {format_int(summary['slice_cache_validation']['invalid_records_count'])}"
    )

    lines.append("")
    lines.append("Slice Selection")
    lines.append(
        f"  - lung-focused выборов: {format_int(summary['slice_selection_stats']['lung_focused_selection_count'])}"
    )
    lines.append(
        f"  - fallback выборов: {format_int(summary['slice_selection_stats']['fallback_selection_count'])}"
    )
    lines.append(
        f"  - full-volume lung-range: {format_int(summary['slice_selection_stats']['full_volume_range_count'])}"
    )
    lines.append(
        f"  - lung_range_start == 0: {format_int(summary['slice_selection_stats']['start_zero_count'])}"
    )
    lines.append(
        f"  - lung_range_end == last_slice: {format_int(summary['slice_selection_stats']['end_last_slice_count'])}"
    )
    if "expanded_range_count" in summary["slice_selection_stats"]:
        lines.append(
            f"  - расширенных lung-range: {format_int(summary['slice_selection_stats']['expanded_range_count'])}"
        )
    if "duplicate_slice_index_count" in summary["slice_selection_stats"]:
        lines.append(
            "  - кейсов с дублирующимися slice indices: "
            f"{format_int(summary['slice_selection_stats']['duplicate_slice_index_count'])}"
        )

    lines.append("")
    lines.append("Тексты")
    lines.append(f"  - уникальных target-text: {format_int(summary['text_stats']['unique_target_texts'])}")
    lines.append(f"  - доля повторяющихся target-text: {summary['text_stats']['repeated_target_text_ratio']}")
    lines.append(f"  - доля самого частого target-text: {summary['text_stats']['most_common_target_text_share']}")
    lines.append(
        f"  - длина текста (символы): min={format_int(summary['text_stats']['text_length_chars']['min'])}, "
        f"mean={summary['text_stats']['text_length_chars']['mean']}, "
        f"max={format_int(summary['text_stats']['text_length_chars']['max'])}"
    )
    lines.append(
        f"  - предложений на запись: min={format_int(summary['text_stats']['sentences_per_record']['min'])}, "
        f"mean={summary['text_stats']['sentences_per_record']['mean']}, "
        f"max={format_int(summary['text_stats']['sentences_per_record']['max'])}"
    )

    if summary["anatomy_counts"]:
        lines.append("")
        lines.append("Топ Anatomy")
        lines.extend(format_top_anatomy(summary["anatomy_counts"]))

    if summary["split_anatomy_counts"]:
        lines.append("")
        lines.append("Anatomy По Сплитам")
        for split_name in ("train", "val", "test"):
            split_counts = summary["split_anatomy_counts"].get(split_name, {})
            if not split_counts:
                lines.append(f"  - {split_name}: нет записей")
                continue
            rendered = ", ".join(
                f"{anatomy}={format_int(count)}" for anatomy, count in split_counts.items()
            )
            lines.append(f"  - {split_name}: {rendered}")

    lines.append("")
    lines.append("Артефакты")
    lines.append(f"  - train: {summary['artifacts']['train']}")
    lines.append(f"  - val: {summary['artifacts']['val']}")
    lines.append(f"  - test: {summary['artifacts']['test']}")
    lines.append(f"  - summary: {summary['artifacts']['summary']}")
    if "excluded_studies" in summary["artifacts"]:
        lines.append(f"  - excluded studies: {summary['artifacts']['excluded_studies']}")
    lines.append(f"  - studies manifest: {summary['artifacts']['studies_manifest']}")
    lines.append(f"  - slices dir: {summary['artifacts']['slices_dir']}")

    return "\n".join(lines)
