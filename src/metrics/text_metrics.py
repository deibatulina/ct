from pathlib import Path

import config


EVALUATE_CACHE_DIR = config.PROJECT_ROOT / ".cache" / "evaluate"


def average(values):
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def load_metric(evaluate_module, metric_name):
    EVALUATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return evaluate_module.load(metric_name, cache_dir=str(EVALUATE_CACHE_DIR))


def compute_text_metrics(predictions, references):
    """
    Считает BLEU, ROUGE и BERTScore через evaluate.
    """
    metrics = {
        "bleu": None,
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "bertscore": None,
    }

    try:
        import evaluate
    except ImportError:
        metrics["metric_errors"] = {"evaluate": "evaluate is not installed"}
        return metrics

    metric_errors = {}

    try:
        bleu_metric = load_metric(evaluate, "bleu")
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[reference] for reference in references],
        )
        metrics["bleu"] = round(float(bleu_result["bleu"]), 6)
    except Exception as error:
        metric_errors["bleu"] = str(error)

    try:
        rouge_metric = load_metric(evaluate, "rouge")
        rouge_result = rouge_metric.compute(
            predictions=predictions,
            references=references,
        )
        metrics["rouge1"] = round(float(rouge_result["rouge1"]), 6)
        metrics["rouge2"] = round(float(rouge_result["rouge2"]), 6)
        metrics["rougeL"] = round(float(rouge_result["rougeL"]), 6)
    except Exception as error:
        metric_errors["rouge"] = str(error)

    try:
        bertscore_metric = load_metric(evaluate, "bertscore")
        bertscore_result = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            lang="en",
        )
        metrics["bertscore"] = average([float(value) for value in bertscore_result["f1"]])
    except Exception as error:
        metric_errors["bertscore"] = str(error)

    if metric_errors:
        metrics["metric_errors"] = metric_errors

    return metrics
