import torch


class CTBatchCollator:
    def __init__(self, tokenizer, max_text_length, mask_prompt_tokens=True):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.mask_prompt_tokens = mask_prompt_tokens
        self.eos_token = tokenizer.eos_token or ""

    def __call__(self, samples):
        if not samples:
            raise ValueError("Batch is empty.")

        images = torch.stack([sample["images"] for sample in samples], dim=0)
        ids = [sample["id"] for sample in samples]
        texts = [sample["text"] for sample in samples]
        prompts = [sample["prompt"] for sample in samples]
        anatomies = [sample["anatomy"] for sample in samples]
        anatomy_ids = torch.tensor([sample["anatomy_id"] for sample in samples], dtype=torch.long)
        slice_positions = torch.stack([sample["slice_positions"] for sample in samples], dim=0)
        metas = [sample["meta"] for sample in samples]

        full_texts = []
        for prompt, text in zip(prompts, texts):
            if text:
                full_texts.append(f"{prompt} {text}{self.eos_token}".strip())
            else:
                full_texts.append(f"{prompt}{self.eos_token}".strip())

        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        labels = encoded["input_ids"].clone()
        prompt_lengths = []

        for index, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_text_length,
                add_special_tokens=False,
            )["input_ids"]
            prompt_length = min(len(prompt_ids), labels.size(1))
            prompt_lengths.append(prompt_length)

            if self.mask_prompt_tokens and prompt_length > 0:
                labels[index, :prompt_length] = -100

        labels[encoded["attention_mask"] == 0] = -100

        return {
            "ids": ids,
            "images": images,
            "texts": texts,
            "prompts": prompts,
            "anatomies": anatomies,
            "anatomy_ids": anatomy_ids,
            "slice_positions": slice_positions,
            "metas": metas,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }
