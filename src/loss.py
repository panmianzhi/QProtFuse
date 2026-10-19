"""Masked pairwise SigLIP objective."""
import torch.nn.functional as F
import torch


def masked_metadata_siglip_loss(
    scores,
    query_positions,
    candidate_positions,
    query_metadata_ids,
    candidate_metadata_ids,
    query_instruction,
    candidate_instruction,
    query_metadata,
    candidate_metadata,
    temperature,
    bias,
    instruction_min=0.65,
    metadata_negative_max=0.70,
    negative_hardness=None,
    hard_negative_topk=0,
    hard_negative_weight=0.0,
    hard_negative_weighting="linear",
    hard_negative_temperature=0.1,
    balance_classes=False,
):
    """Return a configurable query-balanced SigLIP loss.

    Each retained pair contributes ``-log(sigmoid(y * (cosine / tau + bias)))``.
    Negatives can be restricted to the top-K by a frozen similarity. Linear
    weighting uses ``1 + weight * hardness``. Exponential weighting uses
    ``exp((hardness - row_max) / temperature)`` and normalizes the selected
    negative weights to mean one. With ``balance_classes``, positive and
    negative means receive equal weight within each query before retained
    queries are averaged.
    """
    if hard_negative_weighting not in {"linear", "exponential"}:
        raise ValueError("hard_negative_weighting must be linear or exponential")
    if hard_negative_temperature <= 0:
        raise ValueError("hard_negative_temperature must be positive")
    same_record = query_positions[:, None].eq(candidate_positions[None, :])
    same_task = (
        query_instruction.float() @ candidate_instruction.float().T
    ).gt(instruction_min)
    exact_metadata = query_metadata_ids[:, None].eq(candidate_metadata_ids[None, :])
    positive = same_task & exact_metadata & ~same_record
    negative = (
        same_task
        & (query_metadata.float() @ candidate_metadata.float().T).lt(metadata_negative_max)
        & ~exact_metadata
        & ~same_record
    )
    if negative_hardness is not None:
        if negative_hardness.shape != scores.shape:
            raise ValueError("negative_hardness must have the same shape as scores")
        if hard_negative_topk > 0:
            keep_count = min(hard_negative_topk, negative.shape[1])
            ranked = negative_hardness.float().masked_fill(~negative, -torch.inf)
            selected = torch.zeros_like(negative)
            selected.scatter_(1, ranked.topk(keep_count, dim=1).indices, True)
            negative = negative & selected
    eligible = positive | negative
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    keep = (positive_count > 0) & (negative_count > 0)
    if not bool(keep.any()):
        return None, {
            "queries": scores.shape[0],
            "kept": 0,
            "positive_pairs": int(positive.sum()),
            "negative_pairs": int(negative.sum()),
        }
    logits = scores[keep].float() / temperature + bias
    eligible_kept = eligible[keep]
    positive_kept = positive[keep]
    negative_kept = eligible_kept & ~positive_kept
    labels = torch.where(positive_kept, 1.0, -1.0)
    pair_loss = F.softplus(-labels * logits)
    if negative_hardness is not None and hard_negative_weighting == "exponential":
        hardness = negative_hardness[keep].float()
        selected_hardness = hardness.masked_fill(~negative_kept, -torch.inf)
        row_max = selected_hardness.max(dim=1, keepdim=True).values
        negative_weights = torch.exp(
            ((hardness - row_max) / hard_negative_temperature).clamp(min=-80.0, max=0.0)
        ).masked_fill(~negative_kept, 0.0)
        weight_sum = negative_weights.sum(dim=1, keepdim=True)
        negative_weights = negative_weights * (
            negative_kept.sum(dim=1, keepdim=True) / weight_sum
        )
    elif negative_hardness is not None and hard_negative_weight > 0:
        hardness = negative_hardness[keep].float().clamp(min=0.0, max=1.0)
        negative_weights = 1.0 + hard_negative_weight * hardness
    else:
        negative_weights = torch.ones_like(pair_loss)
    if balance_classes:
        positive_loss = pair_loss.masked_fill(~positive_kept, 0.0).sum(dim=1)
        positive_loss = positive_loss / positive_kept.sum(dim=1)
        weighted_negative = pair_loss * negative_weights
        negative_loss = weighted_negative.masked_fill(~negative_kept, 0.0).sum(dim=1)
        negative_normalizer = negative_weights.masked_fill(~negative_kept, 0.0).sum(dim=1)
        per_query = 0.5 * (positive_loss + negative_loss / negative_normalizer)
    else:
        pair_loss = torch.where(positive_kept, pair_loss, pair_loss * negative_weights)
        per_query = pair_loss.masked_fill(~eligible_kept, 0.0).sum(dim=1)
        per_query = per_query / eligible_kept.sum(dim=1)
    loss = per_query.mean()
    return loss, {
        "queries": scores.shape[0],
        "kept": int(keep.sum()),
        "positive_pairs": int(positive.sum()),
        "negative_pairs": int(negative.sum()),
        "eligible_pairs": int(eligible.sum()),
        "selected_negative_pairs": int(negative.sum()),
    }
