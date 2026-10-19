"""Train QProtFuse jointly on the Prot-Inst-OOD train split."""
import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from loss import (
    masked_metadata_siglip_loss,
)
from model import DEFAULT_LAYER_GROUPS, ProteinMetadataReranker

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("catalytic_activity", "domain_motif", "general_function", "protein_function")
DATASET = ROOT / "data/Prot-Inst-OOD"
MEAN_DB = ROOT / "data/esm2_650m_all_layers_mean_std/mean.sqlite"
METADATA_CACHE = ROOT / "data/protinst_ood_metadata_embeddings"
INSTRUCTION_CACHE = ROOT / "data/protinst_ood_instruction_embeddings"


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def load_embedding_catalog(directory, catalog_name, text_key):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / catalog_name).read_text())["records"]
    if manifest.get("status") != "complete" or manifest.get("model") != "text-embedding-3-large":
        raise ValueError(f"Invalid OpenAI embedding cache: {directory}")
    if not all(row["id"] == index for index, row in enumerate(records)):
        raise ValueError(f"Embedding IDs are not contiguous in {directory}")
    vectors = np.load(directory / "embeddings.npy")
    if vectors.shape != (len(records), 3072) or vectors.dtype != np.float32:
        raise ValueError(f"Embedding array mismatch in {directory}: {vectors.shape} {vectors.dtype}")
    lookup = {row[text_key].strip(): row["id"] for row in records}
    if len(lookup) != len(records):
        raise ValueError(f"Duplicate catalog strings in {directory}")
    return lookup, vectors


class ProtInstTrainData(Dataset):
    """Prot-Inst-OOD train rows with exact cache IDs; test rows are never retained."""

    def __init__(self):
        metadata_lookup, metadata_vectors = load_embedding_catalog(
            METADATA_CACHE, "metadata.json", "metadata"
        )
        instruction_lookup, instruction_vectors = load_embedding_catalog(
            INSTRUCTION_CACHE, "instructions.json", "instruction"
        )
        rows, skipped_empty = [], 0
        for task in TASKS:
            for source_row, row in enumerate(json.loads((DATASET / f"{task}_OOD.json").read_text())):
                if row["split"] != "train":
                    continue
                metadata = row.get("metadata")
                if not isinstance(metadata, str) or not metadata.strip():
                    skipped_empty += 1
                    continue
                metadata, instruction = metadata.strip(), row["instruction"].strip()
                if metadata not in metadata_lookup:
                    raise KeyError(f"Metadata cache miss: {task}:{source_row}")
                if instruction not in instruction_lookup:
                    raise KeyError(f"Instruction cache miss: {task}:{source_row}")
                digest = sequence_sha256(row["sequence"])
                rows.append((
                    task,
                    source_row,
                    digest,
                    metadata_lookup[metadata],
                    instruction_lookup[instruction],
                ))
        self.rows = rows
        self.metadata_ids = np.asarray([row[3] for row in rows], dtype=np.int64)
        self.instruction_ids = np.asarray([row[4] for row in rows], dtype=np.int64)
        self.task_ids = np.asarray([TASKS.index(row[0]) for row in rows], dtype=np.int64)
        self.metadata_vectors = metadata_vectors
        self.instruction_vectors = instruction_vectors
        self.skipped_empty_metadata = skipped_empty

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return int(index)


class MeanFeatureBank:
    """Load only one DDP rank's Prot-Inst-OOD mean-feature shard."""

    def preload(self, rows, positions, device):
        positions = np.asarray(positions, dtype=np.int64)
        wanted = defaultdict(list)
        for local_index, position in enumerate(positions):
            wanted[rows[int(position)][2]].append(local_index)
        host = np.empty((len(positions), 33, 1280), dtype=np.float16)
        found, loaded = set(), 0
        print(f"MEAN_BANK_PRELOAD_START rows={len(positions)} unique={len(wanted)}", flush=True)
        with readonly(MEAN_DB) as database:
            for digest, blob in database.execute("SELECT sequence_sha256, pooled FROM features"):
                local_indices = wanted.get(digest)
                if local_indices is None:
                    continue
                host[local_indices] = np.frombuffer(blob, dtype="<f2").reshape(34, 1280)[1:]
                found.add(digest)
                loaded += len(local_indices)
        if loaded != len(positions) or len(found) != len(wanted):
            missing = next(digest for digest in wanted if digest not in found)
            raise KeyError(f"Mean feature missing: {missing}; loaded={loaded}/{len(positions)}")
        tensor = torch.from_numpy(host).to(device=device, non_blocking=True)
        print(
            f"MEAN_BANK_GPU_READY rows={len(positions)} "
            f"gib={tensor.numel() * tensor.element_size() / 2**30:.2f}",
            flush=True,
        )
        return tensor


class PairAwareBatchSampler(Sampler):
    """Put exact-metadata, instruction-compatible pairs into every local batch."""

    def __init__(
        self,
        local_positions,
        metadata_ids,
        instruction_ids,
        instruction_compatible,
        batch_size,
        steps,
        seed,
        pair_fraction=0.5,
        task_ids=None,
        task_balanced=False,
        task_balance_strength=1.0,
    ):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("batch_size must be even and at least four")
        self.local_positions = np.asarray(local_positions, dtype=np.int64)
        self.metadata_ids = metadata_ids
        self.instruction_ids = instruction_ids
        self.instruction_compatible = instruction_compatible
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self.epoch = 0
        self.task_ids = task_ids
        self.task_balanced = bool(task_balanced)
        self.task_balance_strength = float(task_balance_strength)
        if not 0.0 <= self.task_balance_strength <= 1.0:
            raise ValueError("task_balance_strength must be between 0 and 1")
        groups = defaultdict(list)
        for position in self.local_positions:
            groups[int(metadata_ids[position])].append(int(position))
        self.groups = {}
        for metadata_id, members in groups.items():
            valid_anchors = []
            for anchor in members:
                compatible = [
                    candidate
                    for candidate in members
                    if candidate != anchor
                    and instruction_compatible[
                        instruction_ids[anchor], instruction_ids[candidate]
                    ]
                ]
                if compatible:
                    valid_anchors.append((anchor, compatible))
            if valid_anchors:
                self.groups[metadata_id] = valid_anchors
        if not self.groups:
            raise ValueError("No local metadata group has an instruction-compatible positive pair")
        self.group_ids = np.asarray(sorted(self.groups), dtype=np.int64)
        self.pairs_per_batch = max(1, int(batch_size * pair_fraction) // 2)
        if self.task_balanced:
            if task_ids is None:
                raise ValueError("task_ids are required for task-balanced sampling")
            self.task_values = np.asarray(sorted(np.unique(task_ids[self.local_positions])))
            self.task_positions = {
                int(task): self.local_positions[task_ids[self.local_positions] == task]
                for task in self.task_values
            }
            natural = np.asarray(
                [len(self.task_positions[int(task)]) for task in self.task_values],
                dtype=np.float64,
            )
            natural /= natural.sum()
            uniform = np.full(len(self.task_values), 1.0 / len(self.task_values))
            self.task_probabilities = (
                (1.0 - self.task_balance_strength) * natural
                + self.task_balance_strength * uniform
            )
            self.task_groups = {}
            self.task_group_ids = {}
            for task in self.task_values:
                task_group = {}
                for metadata_id, members in groups.items():
                    task_members = [
                        member for member in members if task_ids[member] == task
                    ]
                    valid_anchors = []
                    for anchor in task_members:
                        compatible = [
                            candidate
                            for candidate in task_members
                            if candidate != anchor
                            and instruction_compatible[
                                instruction_ids[anchor], instruction_ids[candidate]
                            ]
                        ]
                        if compatible:
                            valid_anchors.append((anchor, compatible))
                    if valid_anchors:
                        task_group[metadata_id] = valid_anchors
                if not task_group:
                    raise ValueError(f"Task {task} has no local compatible positive group")
                self.task_groups[int(task)] = task_group
                self.task_group_ids[int(task)] = np.asarray(tuple(task_group), dtype=np.int64)

    def _sample_tasks(self, count, rng):
        expected = self.task_probabilities * count
        quotas = np.floor(expected).astype(np.int64)
        remainder = count - int(quotas.sum())
        if remainder:
            residual = expected - quotas
            order = np.argsort(-(residual + rng.random(len(residual)) * 1e-12))
            quotas[order[:remainder]] += 1
        values = np.repeat(self.task_values, quotas)
        rng.shuffle(values)
        return values

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        filler_count = self.batch_size - 2 * self.pairs_per_batch
        for _ in range(self.steps):
            batch = []
            if self.task_balanced:
                for task in self._sample_tasks(self.pairs_per_batch, rng):
                    task_group = self.task_groups[int(task)]
                    group_ids = self.task_group_ids[int(task)]
                    metadata_id = group_ids[int(rng.integers(len(group_ids)))]
                    anchors = task_group[metadata_id]
                    anchor, compatible = anchors[int(rng.integers(len(anchors)))]
                    positive = compatible[int(rng.integers(len(compatible)))]
                    batch.extend((anchor, positive))
            else:
                selected_groups = rng.choice(
                    self.group_ids,
                    size=self.pairs_per_batch,
                    replace=len(self.group_ids) < self.pairs_per_batch,
                )
                for metadata_id in selected_groups:
                    anchors = self.groups[int(metadata_id)]
                    anchor, compatible = anchors[int(rng.integers(len(anchors)))]
                    positive = compatible[int(rng.integers(len(compatible)))]
                    batch.extend((anchor, positive))
            if filler_count:
                if self.task_balanced:
                    for task in self._sample_tasks(filler_count, rng):
                        positions = self.task_positions[int(task)]
                        batch.append(int(positions[int(rng.integers(len(positions)))]))
                else:
                    filler = rng.choice(
                        self.local_positions,
                        size=filler_count,
                        replace=len(self.local_positions) < filler_count,
                    )
                    batch.extend(int(value) for value in filler)
            rng.shuffle(batch)
            yield batch


def collate_positions(values):
    return np.asarray(values, dtype=np.int64)


def all_gather_plain(value, world_size):
    if world_size == 1 or not dist.is_initialized():
        return value
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        data,
        batch_size,
        learning_rate,
        temperature,
        siglip_bias,
        layer_temperature,
        layer_learning_rate,
        instruction_min,
        metadata_negative_max,
        balance_classes,
        hard_negative_topk,
        hard_negative_weight,
        hard_negative_weighting,
        hard_negative_temperature,
        output_dir,
        seed,
        task_balanced_sampling=False,
        task_balance_strength=1.0,
        alignment_target="joint",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        if alignment_target != "joint":
            raise ValueError("QProtFuse training requires both PP and PT objectives")
        self.model = ProteinMetadataReranker(layer_temperature=layer_temperature)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(temperature)))
        self.siglip_bias = torch.nn.Parameter(torch.tensor(float(siglip_bias)))
        self.data_ref = data
        self.local_positions = None
        self.position_to_local = None
        self.feature_bank = None
        self.metadata_bank = None
        self.instruction_bank = None
        self.instruction_compatible = None

    def setup(self, stage=None):
        rank, world = self.global_rank, self.trainer.world_size
        metadata_ids = self.data_ref.metadata_ids
        self.local_positions = np.asarray(
            [i for i, metadata_id in enumerate(metadata_ids) if metadata_id % world == rank],
            dtype=np.int64,
        )
        self.position_to_local = {
            int(position): local for local, position in enumerate(self.local_positions)
        }
        self.feature_bank = MeanFeatureBank().preload(
            self.data_ref.rows, self.local_positions, self.device
        )
        metadata = torch.from_numpy(self.data_ref.metadata_vectors).to(
            device=self.device, dtype=torch.float16
        )
        instruction = torch.from_numpy(self.data_ref.instruction_vectors).to(
            device=self.device, dtype=torch.float32
        )
        self.metadata_bank = torch.nn.functional.normalize(metadata.float(), dim=-1).half()
        self.instruction_bank = torch.nn.functional.normalize(instruction, dim=-1)
        self.instruction_compatible = (
            self.instruction_bank @ self.instruction_bank.T
        ).gt(self.hparams.instruction_min).cpu().numpy()
        print(
            f"TRAIN_SHARD_READY rank={rank}/{world} rows={len(self.local_positions)} "
            f"global_rows={len(self.data_ref)}",
            flush=True,
        )

    def train_dataloader(self):
        steps = math.ceil(
            len(self.data_ref) / (self.trainer.world_size * self.hparams.batch_size)
        )
        sampler = PairAwareBatchSampler(
            self.local_positions,
            self.data_ref.metadata_ids,
            self.data_ref.instruction_ids,
            self.instruction_compatible,
            self.hparams.batch_size,
            steps,
            self.hparams.seed + self.global_rank * 100003,
            task_ids=self.data_ref.task_ids,
            task_balanced=self.hparams.task_balanced_sampling,
            task_balance_strength=self.hparams.task_balance_strength,
        )
        if self.hparams.task_balanced_sampling:
            print(
                "TASK_BALANCED_SAMPLER_READY "
                f"rank={self.global_rank}/{self.trainer.world_size} "
                f"positions={[len(sampler.task_positions[int(task)]) for task in sampler.task_values]} "
                f"positive_groups={[len(sampler.task_groups[int(task)]) for task in sampler.task_values]} "
                f"strength={sampler.task_balance_strength:.3f} "
                f"probabilities={[round(float(value), 6) for value in sampler.task_probabilities]}",
                flush=True,
            )
        return DataLoader(
            self.data_ref,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_positions,
        )

    def configure_optimizers(self):
        layer_parameters = self.model.learned_weight_parameters()
        layer_parameter_ids = {id(parameter) for parameter in layer_parameters}
        projection_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in layer_parameter_ids
        ]
        groups = [
            {"params": projection_parameters, "weight_decay": 1e-4},
            {"params": [self.log_temperature, self.siglip_bias], "weight_decay": 0.0},
        ]
        if layer_parameters:
            groups.append({
                "params": layer_parameters,
                "lr": self.hparams.layer_learning_rate,
                "weight_decay": 1e-4,
            })
        return torch.optim.AdamW(groups, lr=self.hparams.learning_rate)

    def training_step(self, positions, batch_index):
        positions = np.asarray(positions, dtype=np.int64)
        device = self.device
        local_indices = torch.as_tensor(
            [self.position_to_local[int(position)] for position in positions], device=device
        )
        position_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
        metadata_ids = torch.as_tensor(
            self.data_ref.metadata_ids[positions], device=device, dtype=torch.long
        )
        instruction_ids = torch.as_tensor(
            self.data_ref.instruction_ids[positions], device=device, dtype=torch.long
        )
        instruction = self.instruction_bank[instruction_ids]
        local_features = self.feature_bank[local_indices]

        world, rank = self.trainer.world_size, self.global_rank
        global_features = all_gather_plain(local_features, world)
        global_positions = all_gather_plain(position_tensor, world)
        global_metadata_ids = all_gather_plain(metadata_ids, world)
        global_instruction_ids = all_gather_plain(instruction_ids, world)
        protein_scores = self.model.score_query_conditioned(
            local_features, global_features, instruction, instruction_ids
        )
        metadata_scores = self.model.score_query_metadata(
            local_features, instruction, self.metadata_bank[global_metadata_ids]
        )
        query_reference = F.normalize(local_features.float().mean(dim=1), dim=-1)
        candidate_reference = F.normalize(global_features.float().mean(dim=1), dim=-1)
        negative_hardness = query_reference @ candidate_reference.T
        temperature = self.log_temperature.exp()
        def retrieval_loss(scores):
            return masked_metadata_siglip_loss(
                scores,
                position_tensor,
                global_positions,
                metadata_ids,
                global_metadata_ids,
                instruction,
                self.instruction_bank[global_instruction_ids],
                self.metadata_bank[metadata_ids],
                self.metadata_bank[global_metadata_ids],
                temperature=temperature,
                bias=self.siglip_bias,
                instruction_min=self.hparams.instruction_min,
                metadata_negative_max=self.hparams.metadata_negative_max,
                negative_hardness=negative_hardness,
                hard_negative_topk=self.hparams.hard_negative_topk,
                hard_negative_weight=self.hparams.hard_negative_weight,
                hard_negative_weighting=self.hparams.hard_negative_weighting,
                hard_negative_temperature=self.hparams.hard_negative_temperature,
                balance_classes=self.hparams.balance_classes,
            )

        protein_loss, stats = retrieval_loss(protein_scores)
        metadata_loss, metadata_stats = retrieval_loss(metadata_scores)
        if protein_loss is None or metadata_loss is None:
            raise RuntimeError(
                f"Rank {rank} produced no valid joint query: {stats}, {metadata_stats}"
            )
        retrieval_objective = protein_loss + metadata_loss
        loss = retrieval_objective
        local_batch_size = int(position_tensor.numel())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=local_batch_size,
        )
        for name, value in (
            ("train_protein_loss", protein_loss),
            ("train_metadata_loss", metadata_loss),
        ):
            self.log(
                name, value, on_step=True, on_epoch=True,
                sync_dist=True, batch_size=local_batch_size,
            )
        self.log(
            "temperature", temperature, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=local_batch_size,
        )
        self.log(
            "siglip_bias", self.siglip_bias, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=local_batch_size,
        )
        self.log(
            "train_kept_fraction",
            stats["kept"] / stats["queries"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=local_batch_size,
        )
        if batch_index == 0:
            components = (
                f" protein={float(protein_loss.detach()):.6f} "
                f"metadata={float(metadata_loss.detach()):.6f}"
            )
            print(
                f"TRAIN_STEP_COMPLETE rank={rank} epoch={self.current_epoch} "
                f"loss={float(loss.detach()):.6f}{components} stats={stats}",
                flush=True,
            )
        return loss

    def on_train_epoch_end(self):
        sample_count = min(512, len(self.local_positions))
        sample_positions = self.local_positions[:sample_count]
        sample_instruction_ids = torch.as_tensor(
            self.data_ref.instruction_ids[sample_positions], device=self.device
        )
        weights = self.model.layer_weights(
            self.instruction_bank[sample_instruction_ids]
        ).detach().float()
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        diagnostic = {
            "epoch": int(self.current_epoch),
            "instructions": int(weights.shape[0]),
            "mean_layer_entropy": float(entropy.mean()),
            "normalized_layer_entropy": float(entropy.mean() / math.log(weights.shape[1])),
            "mean_max_layer_weight": float(weights.max(dim=-1).values.mean()),
            "layer_argmax_counts": torch.bincount(
                weights.argmax(dim=-1), minlength=weights.shape[1]
            ).cpu().tolist(),
        }
        grouped = torch.stack(
            [weights[:, start:stop].sum(dim=-1) for start, stop in DEFAULT_LAYER_GROUPS],
            dim=-1,
        )
        diagnostic["mean_max_group_weight"] = float(grouped.max(dim=-1).values.mean())
        diagnostic["group_argmax_counts"] = torch.bincount(
            grouped.argmax(dim=-1), minlength=len(DEFAULT_LAYER_GROUPS)
        ).cpu().tolist()
        if self.global_rank == 0:
            path = ROOT / self.hparams.output_dir / "layer_diagnostics.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(diagnostic) + "\n")
            print(f"LAYER_DIAGNOSTIC {json.dumps(diagnostic)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--siglip-bias", type=float, default=-10.0)
    parser.add_argument("--layer-temperature", type=float, default=2.0)
    parser.add_argument("--layer-learning-rate", type=float, default=1e-4)
    parser.add_argument("--instruction-min", type=float, default=0.65)
    parser.add_argument("--metadata-negative-max", type=float, default=0.70)
    parser.add_argument(
        "--balance-classes", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--hard-negative-topk", type=int, default=0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument(
        "--hard-negative-weighting", choices=("linear", "exponential"), default="linear"
    )
    parser.add_argument("--hard-negative-temperature", type=float, default=0.1)
    parser.add_argument(
        "--task-balanced-sampling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--task-balance-strength", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    ddp_child = "LOCAL_RANK" in os.environ
    if not ddp_child:
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        for _ in range(120):
            if args.output.exists():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("DDP child did not observe output directory")
    pl.seed_everything(args.seed, workers=True)
    data = ProtInstTrainData()
    if not ddp_child:
        config = {
            "framework": "pytorch_lightning",
            "dataset": "Prot-Inst-OOD train split only",
            "tasks": TASKS,
            "records": len(data),
            "skipped_empty_metadata": data.skipped_empty_metadata,
            "training_data_policy": "all eligible records in the official train split; no held-out validation split",
            "architecture": "instruction embedding -> MLP -> softmax(logits / 2) over seven ESM groups; weighted ESM feature sum -> 512D projection",
            "layer_weight_initialization": "zero-initialized final MLP layer; uniform group weights at initialization",
            "layer_groups_zero_based_half_open": DEFAULT_LAYER_GROUPS,
            "metadata_projector": "LayerNorm(3072) -> Linear(3072,1024) -> GELU -> Linear(1024,512) -> LayerNorm(512) -> L2",
            "layer_temperature": args.layer_temperature,
            "feature_source": str(MEAN_DB.relative_to(ROOT)),
            "feature_layout": "ESM2-650M blocks 1..33, residue mean pooling only; no std",
            "instruction_embedding": {
                "source": str(INSTRUCTION_CACHE.relative_to(ROOT)),
                "model": "text-embedding-3-large",
                "dimension": 3072,
            },
            "metadata_embedding": {
                "source": str(METADATA_CACHE.relative_to(ROOT)),
                "model": "text-embedding-3-large",
                "dimension": 3072,
            },
            "positive": "exact stripped metadata equality within global DDP batch",
            "negative": f"metadata cosine < {args.metadata_negative_max}",
            "pair_mask": f"instruction cosine > {args.instruction_min}",
            "retrieval_instruction_cosine_min": 0.65,
            "objective": "1.0 * protein-protein masked SigLIP + 1.0 * protein-metadata masked SigLIP",
            "loss_weights": {"protein_protein": 1.0, "protein_metadata": 1.0},
            "balance_positive_negative_within_query": args.balance_classes,
            "hard_negative_topk_by_fixed_all_layer_mean_cosine": args.hard_negative_topk,
            "hard_negative_weight": args.hard_negative_weight,
            "hard_negative_weighting": args.hard_negative_weighting,
            "hard_negative_temperature": args.hard_negative_temperature,
            "task_balanced_sampling": args.task_balanced_sampling,
            "task_balance_strength": args.task_balance_strength,
            "sampling_task_weights": (
                (
                    f"(1-{args.task_balance_strength}) * local natural frequency + "
                    f"{args.task_balance_strength} * uniform"
                ) if args.task_balanced_sampling
                else "natural local-shard distribution"
            ),
            "candidate_conditioning": "protein features use the query instruction's MLP-generated group weights; metadata projection is instruction-independent",
            "temperature": {"learnable": True, "initial_value": args.temperature},
            "siglip_bias": {"learnable": True, "initial_value": args.siglip_bias},
            "query_reduction": (
                "equal positive/negative means per query, then mean retained queries"
                if args.balance_classes
                else "mean eligible pairs per query, then mean retained queries"
            ),
            "batch_size_per_rank": args.batch_size,
            "devices": args.devices,
            "ddp_topology": {
                "nodes": 1,
                "processes_per_node": args.devices,
                "world_size": args.devices,
            },
            "global_candidate_batch": args.batch_size * args.devices,
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "layer_learning_rate": args.layer_learning_rate,
            "validation": False,
            "checkpoint_every_epochs": 5,
            "checkpoint_contents": "trainable model weights and reconstruction hyperparameters only",
            "frozen_provenance": {
                "esm": "ESM2-650M precomputed mean.sqlite",
                "openai": "cached text-embedding-3-large arrays; no API calls",
            },
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    module = RetrievalModule(
        data,
        args.batch_size,
        args.learning_rate,
        args.temperature,
        args.siglip_bias,
        args.layer_temperature,
        args.layer_learning_rate,
        args.instruction_min,
        args.metadata_negative_max,
        args.balance_classes,
        args.hard_negative_topk,
        args.hard_negative_weight,
        args.hard_negative_weighting,
        args.hard_negative_temperature,
        Path(os.path.relpath(args.output.resolve(), ROOT)).as_posix(),
        args.seed,
        args.task_balanced_sampling,
        args.task_balance_strength,
        "joint",
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output),
        filename="epoch-{epoch:02d}",
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
        monitor=None,
        save_weights_only=True,
    )
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        logger=False,
        precision="bf16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        default_root_dir=str(args.output),
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
