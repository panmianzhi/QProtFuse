"""QProtFuse test-to-train retrieval: pp, pt, or rerank of saved PP results.

All modes share data loading, instruction filtering, encoding, and metrics.
PP searches train records; PT searches their unique metadata; rerank only
reorders a saved PP candidate set, keeping its original order for score ties.
"""
import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import ProteinMetadataReranker


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("catalytic_activity", "domain_motif", "general_function", "protein_function")
DATASET = ROOT / "data/Prot-Inst-OOD"
MEAN_DB = ROOT / "data/esm2_650m_all_layers_mean_std/mean.sqlite"
INSTRUCTION_CACHE = ROOT / "data/protinst_ood_instruction_embeddings"
METADATA_CACHE = ROOT / "data/protinst_ood_metadata_embeddings"


def load_records(dataset_dir):
    references, queries = [], []
    for task in TASKS:
        rows = json.loads((dataset_dir / f"{task}_OOD.json").read_text())
        for source_row, row in enumerate(rows):
            target = references if row["split"] == "train" else queries
            prefix = "t" if row["split"] == "train" else "q"
            target.append({
                "id": f"{prefix}{len(target):06d}",
                "task": task,
                "source_row": source_row,
                "instruction": row["instruction"].strip(),
                "sequence_sha256": hashlib.sha256(row["sequence"].encode("ascii")).hexdigest(),
                "metadata": row.get("metadata"),
            })
    if not references or not queries:
        raise ValueError("Prot-Inst-OOD reference or query split is empty")
    return references, queries


def load_mean_features(hashes, device):
    """Load blocks 1..33; retain record duplicates to preserve the PP protocol."""
    wanted = defaultdict(list)
    for index, digest in enumerate(hashes):
        wanted[digest].append(index)
    host = np.empty((len(hashes), 33, 1280), dtype=np.float16)
    loaded = 0
    uri = MEAN_DB.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        for digest, blob in database.execute("SELECT sequence_sha256, pooled FROM features"):
            positions = wanted.get(digest)
            if positions is not None:
                # Row 0 is the ESM embedding output, not a transformer block.
                host[positions] = np.frombuffer(blob, dtype="<f2").reshape(34, 1280)[1:]
                loaded += len(positions)
    if loaded != len(hashes):
        raise KeyError(f"Mean features missing; loaded={loaded}/{len(hashes)}")
    print(f"MEAN_FEATURES_READY records={loaded} unique={len(wanted)}", flush=True)
    return torch.from_numpy(host).to(device=device, non_blocking=True)


def load_text_cache(directory, records_file, text_key):
    """Read and validate a precomputed OpenAI embedding cache; no API calls."""
    manifest = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / records_file).read_text())["records"]
    if manifest.get("status") != "complete" or manifest.get("model") != "text-embedding-3-large":
        raise ValueError(f"Expected complete text-embedding-3-large cache: {directory}")
    if any(row["id"] != index for index, row in enumerate(records)):
        raise ValueError(f"Cache IDs are not contiguous: {directory}")
    lookup = {row[text_key].strip(): row["id"] for row in records}
    if len(lookup) != len(records):
        raise ValueError(f"Duplicate stripped texts in cache: {directory}")
    # Load into memory to support filesystems without memory mapping.
    vectors = np.load(directory / "embeddings.npy")
    if vectors.shape != (len(records), 3072) or vectors.dtype != np.float32:
        raise ValueError(f"Embedding shape/dtype mismatch: {directory}")
    return lookup, vectors


def load_checkpoint(checkpoint, device):
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = loaded.get("hyper_parameters", {})
    if hp.get("alignment_target") != "joint":
        raise ValueError(f"Expected a joint PP+PT checkpoint: {checkpoint}")
    model = ProteinMetadataReranker(
        layer_temperature=hp.get("layer_temperature", 2.0)
    ).to(device).eval()
    state = {key.removeprefix("model."): value
             for key, value in loaded["state_dict"].items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model


def stripped_metadata(row):
    value = row.get("metadata")
    return value.strip() if isinstance(value, str) else None


def encode_metadata(model, references, device, batch_size):
    lookup, vectors = load_text_cache(METADATA_CACHE, "metadata.json", "metadata")
    texts = sorted({stripped_metadata(row) for row in references if stripped_metadata(row)},
                   key=lookup.__getitem__)
    if not texts:
        raise ValueError("No nonempty training metadata")
    cache_ids = np.asarray([lookup[text] for text in texts], dtype=np.int64)
    encoded = []
    for start in range(0, len(texts), batch_size):
        source = torch.from_numpy(vectors[cache_ids[start:start + batch_size]]).to(device)
        # Match training: normalize frozen text embeddings before the text MLP.
        encoded.append(model.encode_metadata(F.normalize(source.float(), dim=-1)))
    return {
        "texts": texts,
        "cache_ids": cache_ids,
        "index": {text: index for index, text in enumerate(texts)},
        "encoded": torch.cat(encoded, dim=0),
    }


def load_protein_predictions(directory, queries, references):
    predictions = {}
    reference_ids = {row["id"] for row in references}
    with (directory / "predictions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in predictions:
                raise ValueError(f"Duplicate PP query: {row['id']}")
            candidates = row.get("topk", [])
            if not candidates or any(value not in reference_ids for value in candidates):
                raise ValueError(f"Missing or invalid PP candidates for {row['id']}")
            predictions[row["id"]] = row
    if set(predictions) != {row["id"] for row in queries}:
        raise ValueError("PP results must contain exactly the current test queries")
    return predictions


def protein_topk(model, query_encoded, query_rows, references, ref_features,
                 allowed, instruction, args):
    """Stream eligible train proteins through the encoder and merge chunk top-k."""
    device = query_encoded.device
    top_scores = torch.full((len(query_rows), args.topk), -torch.inf, device=device)
    top_ids = torch.full((len(query_rows), args.topk), -1, device=device, dtype=torch.long)
    # Fixed record-chunk boundaries keep filtering from changing the chunk
    # layout used to resolve top-k ties for repeated protein features.
    for start in range(0, len(references), args.candidate_chunk):
        ids = torch.arange(start, min(start + args.candidate_chunk, len(references)), device=device)
        ids = ids[allowed[ids]]
        if not len(ids):
            continue
        encoded = model(ref_features[ids], instruction.expand(len(ids), -1))
        scores = query_encoded @ encoded.T
        if args.exclude_identical_sequences:
            hashes = [references[index]["sequence_sha256"] for index in ids.tolist()]
            identical = torch.tensor(
                [[query["sequence_sha256"] == digest for digest in hashes] for query in query_rows],
                device=device,
            )
            scores.masked_fill_(identical, -torch.inf)
        values, local_ids = scores.topk(min(args.topk, len(ids)), dim=1)
        merged_scores = torch.cat((top_scores, values), dim=1)
        merged_ids = torch.cat((top_ids, ids[local_ids]), dim=1)
        top_scores, order = merged_scores.topk(args.topk, dim=1)
        top_ids = merged_ids.gather(1, order)
    return [
        [index for index, score in zip(ids, scores) if index >= 0 and np.isfinite(score)]
        for ids, scores in zip(top_ids.cpu().tolist(), top_scores.cpu().tolist())
    ]


def metadata_topk(query_encoded, candidate_ids, catalog, args):
    """Score each unique eligible metadata once, regardless of protein duplicates."""
    candidates = catalog["encoded"][candidate_ids]
    picked, similarities = [], []
    for start in range(0, len(query_encoded), args.query_batch_size):
        scores = query_encoded[start:start + args.query_batch_size] @ candidates.T
        values, local_ids = scores.topk(min(args.topk, len(candidate_ids)), dim=1)
        picked.extend(candidate_ids[local_ids].cpu().tolist())
        similarities.extend(values.cpu().tolist())
    return picked, similarities


def rerank_candidates(query_encoded, initial_ids, reference_index, references, catalog, allowed):
    """Stable sort by PT score: repeated metadata retain their original PP order."""
    indices = [reference_index[value] for value in initial_ids]
    if not bool(allowed[indices].all()):
        raise ValueError("Input PP candidates violate the requested instruction filter")
    metadata_ids = [catalog["index"].get(stripped_metadata(references[index])) for index in indices]
    scores = [-float("inf")] * len(indices)
    valid = [index for index, metadata_id in enumerate(metadata_ids) if metadata_id is not None]
    if valid:
        values = (query_encoded @ catalog["encoded"][[metadata_ids[index] for index in valid]].T)
        for index, value in zip(valid, values.cpu().tolist()):
            scores[index] = value
    order = sorted(range(len(indices)), key=lambda index: scores[index], reverse=True)
    return [indices[index] for index in order], scores, order


def summarize_results(predictions):
    metrics = {}
    for task in TASKS:
        ranks = [row["first_hit_rank_within_topk"] for row in predictions if row["task"] == task]
        if not ranks:
            raise ValueError(f"No test queries for task {task}")
        metrics[task] = {
            "n": len(ranks),
            "R@1": sum(rank == 1 for rank in ranks) / len(ranks),
            "R@10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
            "MRR@10": sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / len(ranks),
        }
    metrics["macro"] = {key: sum(metrics[task][key] for task in TASKS) / len(TASKS)
                        for key in ("R@1", "R@10", "MRR@10")}
    return metrics


def evaluate(model, references, queries, query_features, instructions, instruction_index,
             args, ref_features=None, catalog=None, protein_predictions=None):
    device = query_features.device
    normalized = F.normalize(instructions.float(), dim=-1)
    compatibility = normalized @ normalized.T
    reference_instructions = torch.tensor(
        [instruction_index[row["instruction"]] for row in references], device=device
    )
    reference_index = {row["id"]: index for index, row in enumerate(references)}
    # Map record-level eligibility to unique metadata only after filtering.
    if args.mode == "pt":
        record_metadata = torch.tensor(
            [catalog["index"].get(stripped_metadata(row), -1) for row in references], device=device
        )
    by_instruction = defaultdict(list)
    for index, row in enumerate(queries):
        by_instruction[row["instruction"]].append(index)

    predictions = []
    for text, positions in by_instruction.items():
        instruction_id = instruction_index[text]
        instruction = instructions[instruction_id:instruction_id + 1]
        # One shared rule for all modes: cosine(query instruction, train instruction) >= threshold.
        allowed = compatibility[instruction_id, reference_instructions].ge(args.instruction_cosine_min)
        candidate_ids = torch.nonzero(allowed, as_tuple=True)[0]
        if not len(candidate_ids):
            raise ValueError(f"No compatible train records for {text!r}")
        encoded_batches = []
        for start in range(0, len(positions), args.query_batch_size):
            indices = positions[start:start + args.query_batch_size]
            encoded_batches.append(model(query_features[indices], instruction.expand(len(indices), -1)))
        query_encoded = torch.cat(encoded_batches, dim=0)
        query_rows = [queries[index] for index in positions]

        if args.mode == "pp":
            picked = protein_topk(model, query_encoded, query_rows, references,
                                  ref_features, allowed, instruction, args)
        elif args.mode == "pt":
            # A metadata is eligible if ANY of its associated train records is eligible.
            metadata_ids = record_metadata[candidate_ids]
            metadata_ids = metadata_ids[metadata_ids >= 0].unique(sorted=True)
            if not len(metadata_ids):
                raise ValueError(f"No compatible training metadata for {text!r}")
            picked, similarities = metadata_topk(query_encoded, metadata_ids, catalog, args)

        for local_index, query in enumerate(query_rows):
            prediction = {"id": query["id"], "task": query["task"]}
            if args.mode == "pt":
                indices = picked[local_index]
                retrieved = [catalog["texts"][index] for index in indices]
                correct = stripped_metadata(query)
                prediction.update({
                    "topk_metadata": retrieved,
                    "topk_metadata_ids": [int(catalog["cache_ids"][index]) for index in indices],
                    "topk_cosine": similarities[local_index],
                })
            else:
                if args.mode == "rerank":
                    initial_ids = protein_predictions[query["id"]]["topk"]
                    indices, scores, order = rerank_candidates(
                        query_encoded[local_index], initial_ids, reference_index,
                        references, catalog, allowed,
                    )
                    prediction.update({
                        "protein_topk_before_rerank": initial_ids,
                        "protein_text_alignment_scores_before_rerank": scores,
                        "candidate_order_after_rerank": order,
                    })
                else:
                    indices = picked[local_index]
                prediction["topk"] = [references[index]["id"] for index in indices]
                # PP compares record metadata verbatim; PT and rerank compare
                # text after trimming leading/trailing whitespace.
                retrieved = [(references[index]["metadata"] if args.mode == "pp"
                              else stripped_metadata(references[index])) for index in indices]
                correct = query["metadata"] if args.mode == "pp" else stripped_metadata(query)
            prediction["first_hit_rank_within_topk"] = next(
                (rank for rank, value in enumerate(retrieved, 1)
                 if value == correct and (args.mode != "rerank" or value is not None)), None
            )
            predictions.append(prediction)
    return summarize_results(predictions), predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pp", "pt", "rerank"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction-cosine-min", type=float, default=0.65)
    parser.add_argument("--topk", type=int, default=10, help="PP/PT top-k; rerank retains the input candidate set")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--candidate-chunk", type=int, default=1024, help="PP candidate encoding batch size")
    parser.add_argument("--metadata-batch-size", type=int, default=2048)
    parser.add_argument("--exclude-identical-sequences", action="store_true", help="PP only")
    parser.add_argument("--protein-results", type=Path, help="Required for rerank: saved PP output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if min(args.topk, args.query_batch_size, args.candidate_chunk, args.metadata_batch_size) < 1:
        parser.error("top-k and batch sizes must be positive")
    if not -1 <= args.instruction_cosine_min <= 1:
        parser.error("instruction threshold must be in [-1, 1]")
    if args.mode == "rerank" and (args.protein_results is None or not args.protein_results.is_dir()):
        parser.error("rerank requires --protein-results pointing to saved PP results")
    if args.mode != "rerank" and args.protein_results is not None:
        parser.error("--protein-results is only used by rerank")
    if args.mode != "pp" and args.exclude_identical_sequences:
        parser.error("--exclude-identical-sequences is only used by PP")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    references, queries = load_records(args.dataset_dir)
    # Only PP needs train-protein features; PT/rerank load test-protein features only.
    feature_rows = references + queries if args.mode == "pp" else queries
    features = load_mean_features([row["sequence_sha256"] for row in feature_rows], device)
    ref_features = features[:len(references)] if args.mode == "pp" else None
    query_features = features[len(references):] if args.mode == "pp" else features
    instruction_texts = sorted({row["instruction"] for row in references + queries})
    lookup, vectors = load_text_cache(INSTRUCTION_CACHE, "instructions.json", "instruction")
    instructions = torch.from_numpy(vectors[[lookup[text] for text in instruction_texts]]).to(device)
    instruction_index = {text: index for index, text in enumerate(instruction_texts)}
    catalog = (encode_metadata(model, references, device, args.metadata_batch_size)
               if args.mode in ("pt", "rerank") else None)
    protein_predictions = (load_protein_predictions(args.protein_results, queries, references)
                           if args.mode == "rerank" else None)
    metrics, predictions = evaluate(
        model, references, queries, query_features, instructions, instruction_index,
        args, ref_features, catalog, protein_predictions,
    )
    manifest = {
        "mode": args.mode,
        "path_base": "repository_root",
        "checkpoint": Path(os.path.relpath(args.checkpoint.resolve(), ROOT)).as_posix(),
        "dataset_dir": Path(os.path.relpath(args.dataset_dir.resolve(), ROOT)).as_posix(),
        "model_config": {"architecture": "instruction-MLP seven-group ESM fusion",
                         "alignment_target": "joint", "layer_temperature": model.layer_temperature},
        "references": len(references), "queries": len(queries),
        "reference_policy": "all official train records from all four tasks; no sequence deduplication",
        "query_policy": "all official test records from all four tasks",
        "instruction_cosine_min": args.instruction_cosine_min,
        "instruction_filter": "train record eligible iff cosine(query instruction, train instruction) >= threshold",
        "instruction_embedding_cache": str(INSTRUCTION_CACHE.relative_to(ROOT)),
        "feature_source": str(MEAN_DB.relative_to(ROOT)),
        "feature_layout": "ESM2-650M blocks 1..33, residue mean pooling only; no std",
        "topk": (len(next(iter(protein_predictions.values()))["topk"])
                 if args.mode == "rerank" else args.topk),
        "success": ("exact raw metadata match" if args.mode == "pp" else "exact stripped metadata match"),
        "ranking_score": "cosine similarity of normalized 512D embeddings",
    }
    if args.mode == "pp":
        manifest.update({"candidate_chunk": args.candidate_chunk,
                         "exclude_identical_sequences": args.exclude_identical_sequences})
    else:
        manifest.update({"unique_reference_metadata": len(catalog["texts"]),
                         "metadata_embedding_cache": str(METADATA_CACHE.relative_to(ROOT))})
        if args.mode == "pt":
            manifest["candidate_policy"] = "unique metadata associated with at least one eligible train record"
        else:
            manifest.update({
                "protein_retrieval_results": Path(os.path.relpath(args.protein_results.resolve(), ROOT)).as_posix(),
                "instruction_filter_applied_before_reranking": True,
                "reranking": "retain PP candidate set; stable sort by PT cosine, preserving original PP order for ties",
            })
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output / "predictions.jsonl").open("x") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
