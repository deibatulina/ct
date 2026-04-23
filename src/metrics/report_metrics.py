def tokenize_text(text):
    return [token for token in text.lower().split() if token]


def distinct_n(predictions, n):
    ngrams = []

    for prediction in predictions:
        tokens = tokenize_text(prediction)
        if len(tokens) < n:
            continue

        for index in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[index : index + n]))

    if not ngrams:
        return 0.0

    return round(len(set(ngrams)) / len(ngrams), 6)


def repetition_ratio(predictions):
    repeated_tokens = 0
    total_tokens = 0

    for prediction in predictions:
        tokens = tokenize_text(prediction)
        total_tokens += len(tokens)
        repeated_tokens += len(tokens) - len(set(tokens))

    if total_tokens == 0:
        return 0.0

    return round(repeated_tokens / total_tokens, 6)


def compute_report_metrics(predictions, too_short_min_tokens=3):
    lengths = [len(tokenize_text(prediction)) for prediction in predictions]

    if not lengths:
        return {
            "avg_generation_length": 0.0,
            "repetition_ratio": 0.0,
            "distinct1": 0.0,
            "distinct2": 0.0,
            "empty_output_pct": 0.0,
            "too_short_output_pct": 0.0,
            "unique_prediction_ratio": 0.0,
            "most_common_prediction_share": 0.0,
        }

    empty_count = sum(1 for prediction in predictions if not prediction.strip())
    too_short_count = sum(1 for length in lengths if length < too_short_min_tokens)
    normalized_predictions = [prediction.strip().lower() for prediction in predictions if prediction.strip()]
    unique_prediction_ratio = 0.0
    most_common_prediction_share = 0.0

    if normalized_predictions:
        counts = {}
        for prediction in normalized_predictions:
            counts[prediction] = counts.get(prediction, 0) + 1

        unique_prediction_ratio = round(len(counts) / len(predictions), 6)
        most_common_prediction_share = round(max(counts.values()) / len(predictions), 6)

    return {
        "avg_generation_length": round(sum(lengths) / len(lengths), 6),
        "repetition_ratio": repetition_ratio(predictions),
        "distinct1": distinct_n(predictions, 1),
        "distinct2": distinct_n(predictions, 2),
        "empty_output_pct": round(100.0 * empty_count / len(predictions), 6),
        "too_short_output_pct": round(100.0 * too_short_count / len(predictions), 6),
        "unique_prediction_ratio": unique_prediction_ratio,
        "most_common_prediction_share": most_common_prediction_share,
    }
