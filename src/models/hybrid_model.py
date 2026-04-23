import torch
from torch import nn
from transformers import GPT2LMHeadModel

import config
from src.models.aligners import BaselineProjection, LinearAligner, MLPAligner
from src.models.lora_utils import apply_lora_to_language_model
from src.models.visual_encoder import VisualEncoder, freeze_module


class HybridCTReportModel(nn.Module):
    def __init__(self, experiment_name, model_config, lora_config=None):
        super().__init__()

        if experiment_name not in {"baseline", "linear_aligner", "mlp_lora"}:
            raise ValueError(f"Unsupported experiment: {experiment_name}")

        self.experiment_name = experiment_name
        self.aggregation = model_config["aggregation"]
        self.visual_prefix_tokens = model_config["visual_prefix_tokens"]
        self.projection_type = model_config["projection_type"]

        self.visual_encoder = VisualEncoder(
            encoder_name=model_config["visual_encoder_name"],
            frozen=model_config["freeze_visual_encoder"],
        )

        if self.experiment_name != self.projection_type:
            raise ValueError(
                "Experiment name and projection type must match: "
                f"{self.experiment_name} != {self.projection_type}"
            )

        if self.projection_type == "baseline":
            self.bridge = BaselineProjection(
                visual_feature_dim=model_config["visual_feature_dim"],
                visual_prefix_tokens=model_config["visual_prefix_tokens"],
                lm_hidden_dim=model_config["lm_hidden_dim"],
                num_anatomy_labels=model_config.get("num_anatomy_labels", len(config.ALLOWED_ANATOMIES)),
                use_anatomy_embedding=model_config.get("use_anatomy_embedding", True),
                use_slice_position_embedding=model_config.get("use_slice_position_embedding", True),
            )
        elif self.projection_type == "linear_aligner":
            self.bridge = LinearAligner(
                visual_feature_dim=model_config["visual_feature_dim"],
                lm_hidden_dim=model_config["lm_hidden_dim"],
                visual_prefix_tokens=model_config["visual_prefix_tokens"],
                num_anatomy_labels=model_config.get("num_anatomy_labels", len(config.ALLOWED_ANATOMIES)),
                use_anatomy_embedding=model_config.get("use_anatomy_embedding", True),
                use_slice_position_embedding=model_config.get("use_slice_position_embedding", True),
            )
        elif self.projection_type == "mlp_lora":
            self.bridge = MLPAligner(
                visual_feature_dim=model_config["visual_feature_dim"],
                mlp_hidden_dim=model_config["mlp_hidden_dim"],
                lm_hidden_dim=model_config["lm_hidden_dim"],
                visual_prefix_tokens=model_config["visual_prefix_tokens"],
                num_anatomy_labels=model_config.get("num_anatomy_labels", len(config.ALLOWED_ANATOMIES)),
                dropout=model_config.get("mlp_dropout", 0.1),
                use_anatomy_embedding=model_config.get("use_anatomy_embedding", True),
                use_slice_position_embedding=model_config.get("use_slice_position_embedding", True),
            )
        else:
            raise ValueError(f"Unsupported projection_type: {self.projection_type}")

        self.language_model = GPT2LMHeadModel.from_pretrained(model_config["language_model_name"])

        if model_config["freeze_language_model"]:
            freeze_module(self.language_model)

        if self.experiment_name == "mlp_lora":
            self.language_model = apply_lora_to_language_model(self.language_model, lora_config)

    def aggregate_slice_features(self, slice_features):
        if self.aggregation != "mean":
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        return slice_features.mean(dim=1)

    def build_prefix_source(self, slice_features, slice_positions=None):
        batch_size, num_slices, feature_dim = slice_features.shape

        if num_slices == self.visual_prefix_tokens:
            prefix_source = slice_features
            prefix_positions = slice_positions
        else:
            if num_slices % self.visual_prefix_tokens != 0:
                raise ValueError(
                    "Aligner expects num_slices to be divisible by visual_prefix_tokens, "
                    f"got {num_slices} and {self.visual_prefix_tokens}."
                )

            slices_per_token = num_slices // self.visual_prefix_tokens
            prefix_source = slice_features.view(
                batch_size,
                self.visual_prefix_tokens,
                slices_per_token,
                feature_dim,
            ).mean(dim=2)
            prefix_positions = None
            if slice_positions is not None:
                prefix_positions = slice_positions.view(
                    batch_size,
                    self.visual_prefix_tokens,
                    slices_per_token,
                ).mean(dim=2)

        return prefix_source, prefix_positions

    def encode_images(self, images, anatomy_ids=None, slice_positions=None):
        slice_features = self.visual_encoder(images)
        pooled_features = self.aggregate_slice_features(slice_features)

        if self.projection_type == "baseline":
            visual_prefix = self.bridge(
                slice_features,
                anatomy_ids=anatomy_ids,
                slice_positions=slice_positions,
            )
        elif self.projection_type in {"linear_aligner", "mlp_lora"}:
            prefix_source, prefix_positions = self.build_prefix_source(
                slice_features=slice_features,
                slice_positions=slice_positions,
            )
            visual_prefix = self.bridge(
                prefix_source,
                anatomy_ids=anatomy_ids,
                slice_positions=prefix_positions,
            )
        else:
            raise ValueError(f"Unsupported projection_type: {self.projection_type}")

        return visual_prefix, slice_features, pooled_features

    def build_multimodal_inputs_from_prefix(
        self,
        visual_prefix,
        input_ids,
        attention_mask,
        labels=None,
        slice_features=None,
        pooled_features=None,
    ):
        token_embeddings = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_prefix, token_embeddings], dim=1)

        prefix_attention = torch.ones(
            visual_prefix.size(0),
            visual_prefix.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), visual_prefix.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        return {
            "visual_prefix": visual_prefix,
            "slice_features": slice_features,
            "pooled_features": pooled_features,
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
            "labels": full_labels,
        }

    def build_multimodal_inputs(self, images, input_ids, attention_mask, labels=None, anatomy_ids=None, slice_positions=None):
        visual_prefix, slice_features, pooled_features = self.encode_images(
            images,
            anatomy_ids=anatomy_ids,
            slice_positions=slice_positions,
        )
        return self.build_multimodal_inputs_from_prefix(
            visual_prefix=visual_prefix,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            slice_features=slice_features,
            pooled_features=pooled_features,
        )

    def forward(self, images, input_ids, attention_mask, labels=None, anatomy_ids=None, slice_positions=None):
        batch = self.build_multimodal_inputs(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            anatomy_ids=anatomy_ids,
            slice_positions=slice_positions,
        )
        outputs = self.language_model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        batch["logits"] = outputs.logits
        return batch
