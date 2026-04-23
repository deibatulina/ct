import time
from collections import Counter
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

import config
from src.data.collators import CTBatchCollator
from src.data.dataset import CTReportDataset
from src.models.hybrid_model import HybridCTReportModel
from src.training.eval import evaluate_model
from src.training.generation import save_predictions, save_qualitative_examples
from src.training.losses import compute_autoregressive_lm_loss
from src.utils.config import load_yaml_config
from src.utils.io import ensure_directory, write_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return config.PROJECT_ROOT / path


def load_experiment_config(config_path):
    return load_yaml_config(config_path)


def build_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_dataloader(split_path, split_name, experiment_config, tokenizer, shuffle=False):
    runtime_config = experiment_config.get("runtime", {})
    data_config = experiment_config["data"]
    training_config = experiment_config.get("training", {})

    dataset = CTReportDataset(
        json_path=resolve_project_path(split_path),
        prompt_template=data_config["prompt_template"],
        num_slices=data_config["num_slices"],
        image_size=data_config["image_size"],
        max_records=runtime_config.get(f"max_{split_name}_records"),
    )

    collator = CTBatchCollator(
        tokenizer=tokenizer,
        max_text_length=data_config["max_text_length"],
        mask_prompt_tokens=True,
    )

    sampler = None
    sampling_strategy = training_config.get("sampling_strategy")
    if split_name == "train" and sampling_strategy == "anatomy_text_balanced":
        sampler = build_balanced_train_sampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=experiment_config["training"]["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collator,
    )


def build_balanced_train_sampler(dataset):
    anatomy_counts = Counter()
    text_counts = Counter()

    for record in dataset.records:
        anatomy = record["anatomy"]
        text = record["text"].strip().lower()
        anatomy_counts[anatomy] += 1
        text_counts[(anatomy, text)] += 1

    weights = []
    for record in dataset.records:
        anatomy = record["anatomy"]
        text = record["text"].strip().lower()
        anatomy_weight = 1.0 / anatomy_counts[anatomy]
        text_weight = 1.0 / text_counts[(anatomy, text)]
        # Blend anatomy-level and exact-text balancing to reduce template collapse
        # without making extremely rare reports dominate the epoch.
        sample_weight = (0.35 * anatomy_weight) + (0.65 * text_weight)
        weights.append(sample_weight)

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def select_metric(metrics, metric_name):
    metric_value = metrics.get(metric_name)
    if metric_value is not None:
        return metric_value

    if metric_name not in {"loss", "val_loss"} and metrics.get("loss") is not None:
        return -metrics["loss"]

    if metric_name == "val_loss":
        return -metrics.get("loss", 0.0)

    if metric_name == "loss":
        return -metrics.get("loss", 0.0)

    return None


def flatten_epoch_metrics(epoch, train_loss, val_metrics):
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
    }

    for key, value in val_metrics.items():
        row[f"val_{key}"] = value

    return row


def resolve_train_log_interval(num_batches):
    if num_batches <= 5:
        return 1
    if num_batches <= 20:
        return 2
    return 5


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    gradient_accumulation_steps,
    max_grad_norm,
    device,
    label_smoothing=0.0,
    logger=None,
):
    model.train()

    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad()
    total_batches = len(dataloader)
    log_interval = resolve_train_log_interval(total_batches)

    if logger is not None:
        logger.info("Starting train epoch: %s batch(es)", total_batches)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        anatomy_ids = batch["anatomy_ids"].to(device)
        slice_positions = batch["slice_positions"].to(device)

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
        step_count += 1

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(dataloader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if logger is not None and (
            batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches
        ):
            mean_loss_so_far = round(total_loss / step_count, 6)
            logger.info(
                "Train progress: batch %s/%s, mean_loss=%s",
                batch_index,
                total_batches,
                mean_loss_so_far,
            )

    if step_count == 0:
        return 0.0

    return round(total_loss / step_count, 6)


def build_output_paths(root_dir):
    checkpoints_dir = ensure_directory(root_dir / "checkpoints")
    predictions_dir = ensure_directory(root_dir / "predictions")
    logs_dir = ensure_directory(root_dir / "logs")
    examples_dir = ensure_directory(root_dir / "examples")

    return {
        "root_dir": str(root_dir),
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "launch_config": str(root_dir / "launch_config.json"),
        "epoch_logs": str(logs_dir / "epoch_logs.json"),
        "metrics": str(logs_dir / "metrics.json"),
        "val_predictions": str(predictions_dir / "val_predictions.json"),
        "test_predictions": str(predictions_dir / "test_predictions.json"),
        "qualitative_examples": str(examples_dir / "test_examples.json"),
        "summary": str(root_dir / "summary.json"),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, experiment_config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": experiment_config,
        },
        path,
    )


def count_training_steps(dataloader_length, num_epochs, gradient_accumulation_steps):
    if dataloader_length == 0:
        return 0

    steps_per_epoch = dataloader_length // gradient_accumulation_steps
    if dataloader_length % gradient_accumulation_steps != 0:
        steps_per_epoch += 1

    return steps_per_epoch * num_epochs


def load_best_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def collect_trainable_parameter_names(model):
    names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
    return names


def validate_trainable_setup(model):
    """
    Явно проверяет, что frozen-модули действительно не попали в обучение,
    а оптимизатор получает только разрешённые группы параметров.
    """
    visual_trainable = collect_trainable_parameter_names(model.visual_encoder)
    if visual_trainable:
        raise ValueError(f"Visual encoder must be frozen, got trainable params: {visual_trainable}")

    trainable_names = collect_trainable_parameter_names(model)
    if not trainable_names:
        raise ValueError("Model has no trainable parameters.")

    bridge_trainable = [name for name in trainable_names if name.startswith("bridge.")]
    if not bridge_trainable:
        raise ValueError("Bridge must remain trainable.")

    language_trainable = collect_trainable_parameter_names(model.language_model)
    if model.experiment_name == "mlp_lora":
        if not language_trainable:
            raise ValueError("LoRA parameters must remain trainable for mlp_lora.")

        invalid_language_names = [name for name in language_trainable if "lora_" not in name]
        if invalid_language_names:
            raise ValueError(
                "Base language-model weights must stay frozen for mlp_lora, got: "
                f"{invalid_language_names}"
            )
    elif language_trainable:
        raise ValueError(f"Language model must be frozen, got trainable params: {language_trainable}")

    invalid_names = [
        name
        for name in trainable_names
        if not name.startswith("bridge.") and "lora_" not in name
    ]
    if invalid_names:
        raise ValueError(
            "Only bridge parameters and LoRA adapters may remain trainable, "
            f"got: {invalid_names}"
        )

    return trainable_names


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_optimizer(model, training_config):
    learning_rate = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]

    if model.experiment_name != "mlp_lora":
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    bridge_parameters = []
    lora_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("bridge."):
            bridge_parameters.append(parameter)
            continue

        if "lora_" in name:
            lora_parameters.append(parameter)
            continue

        raise ValueError(f"Unsupported trainable parameter for optimizer: {name}")

    optimizer_groups = []
    if bridge_parameters:
        optimizer_groups.append(
            {
                "params": bridge_parameters,
                "lr": training_config.get("bridge_learning_rate", learning_rate),
            }
        )
    if lora_parameters:
        optimizer_groups.append(
            {
                "params": lora_parameters,
                "lr": training_config.get("lora_learning_rate", learning_rate),
            }
        )

    return AdamW(
        optimizer_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_experiment(config_path):
    """
    Запускает baseline, linear_aligner и mlp_lora поверх одного общего обучающего контура.
    """
    experiment_config = load_experiment_config(config_path=config_path)
    training_config = experiment_config["training"]
    data_config = experiment_config["data"]
    model_config = experiment_config["model"]
    generation_config = experiment_config["generation"]
    experiment_name = experiment_config["experiment_name"]

    set_seed(training_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = ensure_directory(resolve_project_path(experiment_config["outputs"]["root_dir"]))
    artifacts = build_output_paths(output_root)
    write_json(Path(artifacts["launch_config"]), experiment_config)

    tokenizer = build_tokenizer(model_config["language_model_name"])
    train_loader = build_dataloader(
        split_path=data_config["train_path"],
        split_name="train",
        experiment_config=experiment_config,
        tokenizer=tokenizer,
        shuffle=True,
    )
    val_loader = build_dataloader(
        split_path=data_config["val_path"],
        split_name="val",
        experiment_config=experiment_config,
        tokenizer=tokenizer,
        shuffle=False,
    )
    test_loader = build_dataloader(
        split_path=data_config["test_path"],
        split_name="test",
        experiment_config=experiment_config,
        tokenizer=tokenizer,
        shuffle=False,
    )

    model = HybridCTReportModel(
        experiment_name=experiment_name,
        model_config=model_config,
        lora_config=experiment_config.get("lora"),
    ).to(device)

    trainable_parameter_names = validate_trainable_setup(model)
    trainable_parameter_count = count_trainable_parameters(model)
    optimizer = build_optimizer(model, training_config)

    total_training_steps = count_training_steps(
        dataloader_length=len(train_loader),
        num_epochs=training_config["num_epochs"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    )
    warmup_steps = int(total_training_steps * training_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )

    selection_metric = training_config["early_stopping_metric"]
    best_metric_value = None
    epoch_logs = []
    best_val_metrics = None

    LOGGER.info("Running %s experiment on %s", experiment_name, device)
    LOGGER.info(
        "Dataloaders ready: train=%s batch(es), val=%s batch(es), test=%s batch(es)",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    LOGGER.info(
        "Trainable parameters: %s",
        ", ".join(trainable_parameter_names),
    )
    LOGGER.info("Trainable parameter count: %s", trainable_parameter_count)

    for epoch in range(1, training_config["num_epochs"] + 1):
        LOGGER.info("Epoch %s/%s started", epoch, training_config["num_epochs"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_started_at = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            max_grad_norm=training_config["max_grad_norm"],
            device=device,
            label_smoothing=training_config.get("label_smoothing", 0.0),
            logger=LOGGER,
        )

        val_metrics, val_predictions = evaluate_model(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_input_length=data_config["max_text_length"],
            device=device,
            label_smoothing=training_config.get("label_smoothing", 0.0),
            logger=LOGGER,
            split_name="val",
        )

        epoch_duration = round(time.time() - epoch_started_at, 4)
        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        epoch_row = flatten_epoch_metrics(epoch=epoch, train_loss=train_loss, val_metrics=val_metrics)
        epoch_row["epoch_time_seconds"] = epoch_duration
        epoch_row["peak_memory_mb"] = peak_memory_mb
        epoch_logs.append(epoch_row)
        write_json(Path(artifacts["epoch_logs"]), epoch_logs)
        save_predictions(Path(artifacts["val_predictions"]), val_predictions)

        LOGGER.info(
            "Epoch %s/%s finished: train_loss=%s, val_loss=%s, val_%s=%s",
            epoch,
            training_config["num_epochs"],
            train_loss,
            val_metrics.get("loss"),
            selection_metric,
            val_metrics.get(selection_metric),
        )
        LOGGER.info(
            "Epoch %s diagnostics: epoch_time_seconds=%s, peak_memory_mb=%s",
            epoch,
            epoch_duration,
            peak_memory_mb,
        )

        current_metric_value = select_metric(val_metrics, selection_metric)
        if current_metric_value is None:
            raise ValueError(f"Validation metric is missing: {selection_metric}")

        if val_metrics.get(selection_metric) is None and selection_metric not in {"loss", "val_loss"}:
            LOGGER.warning(
                "Validation metric %s is unavailable for epoch %s, falling back to loss for checkpoint selection.",
                selection_metric,
                epoch,
            )

        if best_metric_value is None or current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_val_metrics = val_metrics
            save_checkpoint(
                path=artifacts["best_checkpoint"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                experiment_config=experiment_config,
            )
            LOGGER.info(
                "New best checkpoint saved at epoch %s with %s=%s",
                epoch,
                selection_metric,
                current_metric_value,
            )

    LOGGER.info("Loading best checkpoint for final test evaluation")
    load_best_checkpoint(artifacts["best_checkpoint"], model, device)

    test_metrics, test_predictions = evaluate_model(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        generation_config=generation_config,
        max_input_length=data_config["max_text_length"],
        device=device,
        label_smoothing=training_config.get("label_smoothing", 0.0),
        logger=LOGGER,
        split_name="test",
    )
    save_predictions(Path(artifacts["test_predictions"]), test_predictions)
    save_qualitative_examples(Path(artifacts["qualitative_examples"]), test_predictions)
    write_json(
        Path(artifacts["metrics"]),
        {
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
    )
    LOGGER.info(
        "%s finished: test_loss=%s, test_rougeL=%s",
        experiment_name,
        test_metrics.get("loss"),
        test_metrics.get("rougeL"),
    )

    summary = {
        "status": "ok",
        "experiment_name": experiment_name,
        "device": str(device),
        "epochs_completed": training_config["num_epochs"],
        "selection_metric": selection_metric,
        "best_val_metric": best_metric_value,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "trainable_parameters": trainable_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
        "artifacts": artifacts,
        "config": experiment_config,
    }
    write_json(Path(artifacts["summary"]), summary)
    return summary


def run_baseline_experiment(config_path="configs/baseline.yaml"):
    """
    Запускает baseline-эксперимент от начала до конца поверх готовых частей подготовленного набора.
    """
    return run_experiment(config_path=config_path)


def run_linear_aligner_experiment(config_path="configs/linear_aligner.yaml"):
    """
    Запускает эксперимент linear_aligner поверх общего обучающего контура.
    """
    return run_experiment(config_path=config_path)


def run_mlp_lora_experiment(config_path="configs/mlp_lora.yaml"):
    """
    Запускает эксперимент mlp_lora поверх общего обучающего контура.
    """
    return run_experiment(config_path=config_path)
