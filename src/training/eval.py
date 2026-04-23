import torch

from src.metrics.report_metrics import compute_report_metrics
from src.metrics.text_metrics import compute_text_metrics
from src.training.generation import generate_batch_predictions
from src.training.losses import compute_autoregressive_lm_loss


def compute_grouped_prediction_metrics(predictions, group_labels):
    grouped = {}

    for prediction, group_label in zip(predictions, group_labels):
        grouped.setdefault(group_label, []).append(prediction)

    metrics = {}
    most_common_shares = []
    unique_ratios = []
    for group_label, group_predictions in grouped.items():
        group_metrics = compute_report_metrics(group_predictions)
        metrics[f"{group_label}_unique_prediction_ratio"] = group_metrics["unique_prediction_ratio"]
        metrics[f"{group_label}_most_common_prediction_share"] = group_metrics["most_common_prediction_share"]
        unique_ratios.append(group_metrics["unique_prediction_ratio"])
        most_common_shares.append(group_metrics["most_common_prediction_share"])

    if unique_ratios:
        metrics["group_mean_unique_prediction_ratio"] = round(sum(unique_ratios) / len(unique_ratios), 6)
    if most_common_shares:
        metrics["group_mean_most_common_prediction_share"] = round(sum(most_common_shares) / len(most_common_shares), 6)

    return metrics


def evaluate_model(
    model,
    dataloader,
    tokenizer,
    generation_config,
    max_input_length,
    device,
    label_smoothing=0.0,
    logger=None,
    split_name="eval",
):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    predictions = []
    references = []
    ids = []
    prompts = []
    anatomy_groups = []
    generated_lengths = []
    finished_by_eos = 0
    finished_by_limit = 0
    generated_sequences = 0
    total_batches_expected = len(dataloader)

    if logger is not None:
        logger.info("Starting %s: %s batch(es)", split_name, total_batches_expected)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        anatomy_ids = batch["anatomy_ids"].to(device)
        slice_positions = batch["slice_positions"].to(device)

        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                anatomy_ids=anatomy_ids,
                slice_positions=slice_positions,
            )
            loss = compute_autoregressive_lm_loss(
                outputs["logits"],
                outputs["labels"],
                label_smoothing=label_smoothing,
            )

        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        batch_predictions, generation_info = generate_batch_predictions(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            generation_config=generation_config,
            max_input_length=max_input_length,
            device=device,
            anatomy_ids=anatomy_ids,
            slice_positions=slice_positions,
        )
        predictions.extend(batch_predictions)
        references.extend(batch["texts"])
        ids.extend(batch["ids"])
        prompts.extend(batch["prompts"])
        anatomy_groups.extend(batch["anatomies"])
        generated_lengths.extend(generation_info["generated_lengths"])
        finished_by_eos += sum(1 for is_finished in generation_info["finished_by_eos"] if is_finished)
        finished_by_limit += sum(1 for is_finished in generation_info["finished_by_limit"] if is_finished)
        generated_sequences += len(generation_info["finished_by_eos"])

        if logger is not None:
            logger.info(
                "%s progress: batch %s/%s",
                split_name,
                batch_index,
                total_batches_expected,
            )

    mean_loss = 0.0
    if total_batches > 0:
        mean_loss = round(total_loss / total_batches, 6)

    metrics = {"loss": mean_loss}
    mean_prediction_length = 0.0
    eos_completion_rate = 0.0
    max_length_completion_rate = 0.0
    if generated_lengths:
        mean_prediction_length = round(sum(generated_lengths) / len(generated_lengths), 4)
    if generated_sequences > 0:
        eos_completion_rate = round(finished_by_eos / generated_sequences, 4)
        max_length_completion_rate = round(finished_by_limit / generated_sequences, 4)

    metrics["mean_prediction_length"] = mean_prediction_length
    metrics["eos_completion_rate"] = eos_completion_rate
    metrics["max_length_completion_rate"] = max_length_completion_rate
    metrics.update(compute_text_metrics(predictions, references))
    metrics.update(compute_report_metrics(predictions))
    metrics.update(compute_grouped_prediction_metrics(predictions, anatomy_groups))

    prediction_rows = []
    for sample_id, prompt, reference, prediction in zip(ids, prompts, references, predictions):
        prediction_rows.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
            }
        )

    if logger is not None:
        logger.info(
            "%s finished: loss=%s, mean_prediction_length=%s, eos_completion_rate=%s, max_length_completion_rate=%s",
            split_name,
            mean_loss,
            mean_prediction_length,
            eos_completion_rate,
            max_length_completion_rate,
        )

    return metrics, prediction_rows
