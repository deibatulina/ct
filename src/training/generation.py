import torch

from src.utils.io import write_json


def apply_repetition_penalty(next_token_logits, generated_ids, generated_attention_mask, repetition_penalty):
    if repetition_penalty is None or repetition_penalty <= 1.0:
        return next_token_logits

    adjusted_logits = next_token_logits.clone()
    batch_size = generated_ids.size(0)

    for batch_index in range(batch_size):
        seen_token_ids = generated_ids[batch_index][generated_attention_mask[batch_index].bool()].unique()
        if seen_token_ids.numel() == 0:
            continue

        token_scores = adjusted_logits[batch_index, seen_token_ids]
        adjusted_logits[batch_index, seen_token_ids] = torch.where(
            token_scores < 0,
            token_scores * repetition_penalty,
            token_scores / repetition_penalty,
        )

    return adjusted_logits


def ban_repeated_ngrams(next_token_logits, generated_ids, generated_attention_mask, no_repeat_ngram_size):
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 1:
        return next_token_logits

    adjusted_logits = next_token_logits.clone()
    prefix_size = no_repeat_ngram_size - 1

    for batch_index in range(generated_ids.size(0)):
        sequence = generated_ids[batch_index][generated_attention_mask[batch_index].bool()].tolist()
        if len(sequence) < prefix_size:
            continue

        banned_tokens = set()
        if prefix_size == 0:
            banned_tokens.update(sequence)
        else:
            ngram_prefix = tuple(sequence[-prefix_size:])
            for start_index in range(len(sequence) - no_repeat_ngram_size + 1):
                if tuple(sequence[start_index : start_index + prefix_size]) == ngram_prefix:
                    banned_tokens.add(sequence[start_index + prefix_size])

        if banned_tokens:
            adjusted_logits[batch_index, list(banned_tokens)] = -torch.inf

    return adjusted_logits


def choose_next_tokens(next_token_logits, generated_ids, generated_attention_mask, generation_config):
    original_logits = next_token_logits
    next_token_logits = apply_repetition_penalty(
        next_token_logits=next_token_logits,
        generated_ids=generated_ids,
        generated_attention_mask=generated_attention_mask,
        repetition_penalty=generation_config.get("repetition_penalty", 1.0),
    )
    next_token_logits = ban_repeated_ngrams(
        next_token_logits=next_token_logits,
        generated_ids=generated_ids,
        generated_attention_mask=generated_attention_mask,
        no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 0),
    )
    invalid_rows = ~torch.isfinite(next_token_logits).any(dim=-1)
    if invalid_rows.any():
        next_token_logits[invalid_rows] = original_logits[invalid_rows]

    do_sample = generation_config["do_sample"]
    if do_sample:
        probabilities = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(1)

    return torch.argmax(next_token_logits, dim=-1)


def generate_batch_predictions(model, tokenizer, batch, generation_config, max_input_length, device, anatomy_ids=None, slice_positions=None):
    if generation_config.get("num_beams", 1) != 1:
        raise ValueError("Only num_beams=1 is supported in the baseline implementation.")

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        prompt_batch = tokenizer(
            batch["prompts"],
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    prompt_input_ids = prompt_batch["input_ids"].to(device)
    prompt_attention_mask = prompt_batch["attention_mask"].to(device)
    images = batch["images"].to(device)

    generated_ids = prompt_input_ids
    generated_attention_mask = prompt_attention_mask
    finished = torch.zeros(prompt_input_ids.size(0), dtype=torch.bool, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        visual_prefix, _, _ = model.encode_images(
            images=images,
            anatomy_ids=anatomy_ids,
            slice_positions=slice_positions,
        )

    for _ in range(generation_config["max_new_tokens"]):
        with torch.no_grad():
            multimodal_inputs = model.build_multimodal_inputs_from_prefix(
                visual_prefix=visual_prefix,
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
            )
            outputs = model.language_model(
                inputs_embeds=multimodal_inputs["inputs_embeds"],
                attention_mask=multimodal_inputs["attention_mask"],
                use_cache=False,
            )

        next_token_logits = outputs.logits[:, -1, :]
        active_sequences = ~finished
        next_tokens = choose_next_tokens(
            next_token_logits=next_token_logits,
            generated_ids=generated_ids,
            generated_attention_mask=generated_attention_mask,
            generation_config=generation_config,
        )
        next_tokens = torch.where(
            active_sequences,
            next_tokens,
            torch.full_like(next_tokens, pad_token_id),
        )

        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
        generated_attention_mask = torch.cat(
            [
                generated_attention_mask,
                active_sequences.unsqueeze(1).to(dtype=generated_attention_mask.dtype),
            ],
            dim=1,
        )

        if eos_token_id is not None:
            finished = finished | (active_sequences & next_tokens.eq(eos_token_id))
            if finished.all().item():
                break

    new_token_ids = generated_ids[:, prompt_input_ids.size(1) :]
    new_attention_mask = generated_attention_mask[:, prompt_input_ids.size(1) :]
    predictions = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    generation_info = {
        "generated_lengths": new_attention_mask.sum(dim=1).tolist(),
        "finished_by_eos": finished.tolist(),
        "finished_by_limit": (~finished).tolist(),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    return [prediction.strip() for prediction in predictions], generation_info


def save_predictions(path, predictions):
    write_json(path, predictions)


def save_qualitative_examples(path, predictions, limit=20):
    write_json(path, predictions[:limit])
