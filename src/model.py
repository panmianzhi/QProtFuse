"""QProtFuse encoder: instruction-routed ESM groups and metadata projection."""

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LAYER_GROUPS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 29), (29, 33))


class AllLayerMLPWeighted(nn.Module):
    """Route an instruction over seven ESM groups, fuse, and project to 512D."""

    def __init__(self, text_dim=3072, layers=33, feature_dim=1280, output_dim=512, layer_temperature=2.0):
        super().__init__()
        if layers != 33:
            raise ValueError("QProtFuse expects ESM transformer blocks 1..33")
        self.layers = layers
        self.feature_dim = feature_dim
        if layer_temperature <= 0:
            raise ValueError("layer_temperature must be positive")
        self.layer_temperature = float(layer_temperature)
        self.text_norm = nn.LayerNorm(text_dim, elementwise_affine=False)
        self.weight_mlp = nn.Sequential(
            nn.Linear(text_dim, 256), nn.GELU(), nn.Linear(256, len(DEFAULT_LAYER_GROUPS))
        )
        # Zero logits initialize the router to uniform weights over groups.
        nn.init.zeros_(self.weight_mlp[-1].weight)
        nn.init.zeros_(self.weight_mlp[-1].bias)
        self.project = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def learned_weight_parameters(self):
        return list(self.weight_mlp.parameters())

    def group_weights(self, instruction):
        """Return the learned softmax distribution over seven ESM groups."""
        if instruction.ndim != 2:
            raise ValueError("instruction must have shape [batch, text_dim]")
        logits = self.weight_mlp(self.text_norm(instruction.float()))
        return (logits / self.layer_temperature).softmax(dim=-1)

    def layer_weights(self, instruction, features=None):
        """Expand each group probability evenly across its constituent ESM blocks."""
        del features  # Retained in the signature for evaluator compatibility.
        group_weights = self.group_weights(instruction)
        pieces = [
            group_weights[:, group_index:group_index + 1].expand(-1, stop - start) / (stop - start)
            for group_index, (start, stop) in enumerate(DEFAULT_LAYER_GROUPS)
        ]
        return torch.cat(pieces, dim=-1)

    def learned_group_weights(self, features, instruction):
        """Return router weights for aligned feature/instruction rows."""
        if features.shape[0] != instruction.shape[0]:
            raise ValueError("features and instruction batch sizes must match")
        return self.group_weights(instruction)

    def encode_with_weights(self, features, weights):
        if features.ndim != 3 or features.shape[1:] != (self.layers, self.feature_dim):
            raise ValueError(
                f"Expected [batch,{self.layers},{self.feature_dim}], got {tuple(features.shape)}"
            )
        if weights.shape != features.shape[:2]:
            raise ValueError(f"Expected weights {tuple(features.shape[:2])}, got {tuple(weights.shape)}")
        fused = torch.einsum("bl,bld->bd", weights, features.float())
        return F.normalize(self.project(fused), dim=-1, eps=1e-8)

    def forward(self, features, instruction=None):
        if instruction is None or instruction.ndim != 2 or instruction.shape[0] != features.shape[0]:
            raise ValueError("Each protein must have one aligned instruction embedding")
        return self.encode_with_weights(features, self.layer_weights(instruction))

    def score_query_conditioned(self, query_features, candidate_features, instruction=None, condition_ids=None):
        """Score candidates with each query's instruction-specific layer weights."""
        if instruction is None or instruction.shape[0] != query_features.shape[0]:
            raise ValueError("Each query must have one instruction embedding")
        query_encoded = self(query_features, instruction)
        if condition_ids is None:
            _, inverse = torch.unique(instruction, dim=0, sorted=True, return_inverse=True)
        else:
            if condition_ids.ndim != 1 or condition_ids.shape[0] != instruction.shape[0]:
                raise ValueError("condition_ids must align with query instructions")
            _, inverse = torch.unique(condition_ids, sorted=True, return_inverse=True)

        row_scores = [None] * query_features.shape[0]
        for group_id in inverse.unique(sorted=True).tolist():
            query_rows = torch.nonzero(inverse == group_id, as_tuple=True)[0]
            instruction_row = query_rows[0]
            candidate_instruction = instruction[instruction_row:instruction_row + 1].expand(
                candidate_features.shape[0], -1
            )
            candidate_encoded = self(candidate_features, candidate_instruction)
            group_scores = query_encoded[query_rows] @ candidate_encoded.T
            for local_index, query_row in enumerate(query_rows.tolist()):
                row_scores[query_row] = group_scores[local_index]
        return torch.stack(row_scores, dim=0)


class ProteinMetadataReranker(AllLayerMLPWeighted):
    """Instruction-conditioned protein encoder with a metadata-text projection."""

    def __init__(self, text_dim=3072, output_dim=512, layer_temperature=2.0):
        super().__init__(text_dim=text_dim, output_dim=output_dim, layer_temperature=layer_temperature)
        self.metadata_project = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def encode_metadata(self, metadata_embeddings):
        return F.normalize(self.metadata_project(metadata_embeddings.float()), dim=-1, eps=1e-8)

    def score_query_metadata(self, query_features, instruction, candidate_metadata):
        return self(query_features, instruction) @ self.encode_metadata(candidate_metadata).T
