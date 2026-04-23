import torch
from torch import nn


class BaselineProjection(nn.Module):
    def __init__(
        self,
        visual_feature_dim,
        visual_prefix_tokens,
        lm_hidden_dim,
        num_anatomy_labels,
        use_anatomy_embedding=True,
        use_slice_position_embedding=True,
    ):
        super().__init__()
        self.visual_prefix_tokens = visual_prefix_tokens
        self.lm_hidden_dim = lm_hidden_dim
        self.use_anatomy_embedding = use_anatomy_embedding
        self.use_slice_position_embedding = use_slice_position_embedding
        self.projection = nn.Linear(
            visual_feature_dim,
            visual_prefix_tokens * lm_hidden_dim,
        )
        self.prefix_position_embeddings = nn.Parameter(torch.zeros(1, visual_prefix_tokens, lm_hidden_dim))
        nn.init.normal_(self.prefix_position_embeddings, std=0.02)

        if self.use_anatomy_embedding:
            self.anatomy_embedding = nn.Embedding(num_anatomy_labels, lm_hidden_dim)

        if self.use_slice_position_embedding:
            self.slice_position_projection = nn.Sequential(
                nn.Linear(1, visual_feature_dim),
                nn.Tanh(),
            )

    def forward(self, slice_features, anatomy_ids=None, slice_positions=None):
        if self.use_slice_position_embedding and slice_positions is not None:
            position_bias = self.slice_position_projection(slice_positions.unsqueeze(-1))
            slice_features = slice_features + position_bias

        pooled_features = slice_features.mean(dim=1)
        projected = self.projection(pooled_features)
        projected = projected.view(
            pooled_features.size(0),
            self.visual_prefix_tokens,
            self.lm_hidden_dim,
        )
        projected = projected + self.prefix_position_embeddings

        if self.use_anatomy_embedding and anatomy_ids is not None:
            projected = projected + self.anatomy_embedding(anatomy_ids).unsqueeze(1)

        return projected


class LinearAligner(nn.Module):
    def __init__(
        self,
        visual_feature_dim,
        lm_hidden_dim,
        visual_prefix_tokens,
        num_anatomy_labels,
        use_anatomy_embedding=True,
        use_slice_position_embedding=True,
    ):
        super().__init__()
        self.use_anatomy_embedding = use_anatomy_embedding
        self.use_slice_position_embedding = use_slice_position_embedding
        self.projection = nn.Linear(visual_feature_dim, lm_hidden_dim)
        self.prefix_position_embeddings = nn.Parameter(torch.zeros(1, visual_prefix_tokens, lm_hidden_dim))
        nn.init.normal_(self.prefix_position_embeddings, std=0.02)

        if self.use_anatomy_embedding:
            self.anatomy_embedding = nn.Embedding(num_anatomy_labels, lm_hidden_dim)

        if self.use_slice_position_embedding:
            self.slice_position_projection = nn.Sequential(
                nn.Linear(1, lm_hidden_dim),
                nn.Tanh(),
            )

    def forward(self, visual_features, anatomy_ids=None, slice_positions=None):
        projected = self.projection(visual_features)

        if self.use_slice_position_embedding and slice_positions is not None:
            projected = projected + self.slice_position_projection(slice_positions.unsqueeze(-1))

        projected = projected + self.prefix_position_embeddings[:, : projected.size(1), :]

        if self.use_anatomy_embedding and anatomy_ids is not None:
            projected = projected + self.anatomy_embedding(anatomy_ids).unsqueeze(1)

        return projected


class MLPAligner(nn.Module):
    def __init__(
        self,
        visual_feature_dim,
        mlp_hidden_dim,
        lm_hidden_dim,
        visual_prefix_tokens,
        num_anatomy_labels,
        dropout=0.1,
        use_anatomy_embedding=True,
        use_slice_position_embedding=True,
    ):
        super().__init__()
        self.use_anatomy_embedding = use_anatomy_embedding
        self.use_slice_position_embedding = use_slice_position_embedding
        self.projection = nn.Sequential(
            nn.Linear(visual_feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, lm_hidden_dim),
        )
        self.prefix_position_embeddings = nn.Parameter(torch.zeros(1, visual_prefix_tokens, lm_hidden_dim))
        nn.init.normal_(self.prefix_position_embeddings, std=0.02)

        if self.use_anatomy_embedding:
            self.anatomy_embedding = nn.Embedding(num_anatomy_labels, lm_hidden_dim)

        if self.use_slice_position_embedding:
            self.slice_position_projection = nn.Sequential(
                nn.Linear(1, lm_hidden_dim),
                nn.Tanh(),
            )

    def forward(self, visual_features, anatomy_ids=None, slice_positions=None):
        projected = self.projection(visual_features)

        if self.use_slice_position_embedding and slice_positions is not None:
            projected = projected + self.slice_position_projection(slice_positions.unsqueeze(-1))

        projected = projected + self.prefix_position_embeddings[:, : projected.size(1), :]

        if self.use_anatomy_embedding and anatomy_ids is not None:
            projected = projected + self.anatomy_embedding(anatomy_ids).unsqueeze(1)

        return projected
