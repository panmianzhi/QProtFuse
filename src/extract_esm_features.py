"""Build the mean-only ESM2 feature database consumed by training and retrieval."""
import argparse
import hashlib
import json
import os
import sqlite3
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, EsmModel


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("catalytic_activity", "domain_motif", "general_function", "protein_function")
DEFAULT_OUTPUT = ROOT / "data/esm2_650m_all_layers_mean_std"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def load_sequences(dataset_dir):
    """Deduplicate train AND test sequences using the readers' exact hash rule."""
    sequences, sources = {}, {}
    for task in TASKS:
        path = dataset_dir / f"{task}_OOD.json"
        sources[relative_path(path)] = file_sha256(path)
        for row in json.loads(path.read_text()):
            sequence = row["sequence"]
            if not sequence or not sequence.isascii() or any(char.isspace() for char in sequence):
                raise ValueError(f"Expected a nonempty ASCII sequence without whitespace in {path.name}")
            # Do not uppercase or strip: training/retrieval hash the original string.
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            previous = sequences.setdefault(digest, sequence)
            if previous != sequence:
                raise ValueError("Sequence hash collision")
    if not sequences:
        raise ValueError("No protein sequences found")
    return sorted(sequences.items()), sources


def sequence_windows(length, size, overlap):
    spans, start = [], 0
    while True:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            return spans
        start += size - overlap


def pooled_forward(model, inputs, weights):
    """Pool layers through hooks, without retaining all residue hidden states.

    HF layer 0 is the input to the first transformer block. The final layer is
    captured AFTER the encoder LayerNorm, matching HF hidden-state indices.
    Only weighted sums are computed; no second moments or std features are kept.
    """
    pooled, handles = {}, []
    depth = model.config.num_hidden_layers

    def capture(level, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            pooled[level] = torch.bmm(weights[:, None, :], hidden.float()).squeeze(1)

    try:
        handles.append(model.encoder.layer[0].register_forward_pre_hook(
            lambda module, args: capture(0, args[0])))
        for level in range(1, depth):
            handles.append(model.encoder.layer[level - 1].register_forward_hook(
                lambda module, args, output, level=level: capture(level, output[0])))
        handles.append(model.encoder.emb_layer_norm_after.register_forward_hook(
            lambda module, args, output: capture(depth, output)))
        model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
        return torch.stack([pooled[level] for level in range(depth + 1)], dim=1)
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def encode(records, model, tokenizer, device, args):
    tasks, totals, window_counts = [], [], []
    for index, (_, sequence) in enumerate(records):
        spans = sequence_windows(len(sequence), args.window, args.overlap)
        coverage = np.zeros(len(sequence), dtype=np.int32)
        for start, end in spans:
            coverage[start:end] += 1
        totals.append(np.zeros((model.config.num_hidden_layers + 1, model.config.hidden_size),
                               dtype=np.float32))
        window_counts.append(len(spans))
        for start, end in spans:
            # A residue appearing in n windows contributes 1/n from each window.
            tasks.append((index, sequence[start:end], 1.0 / coverage[start:end]))
    tasks.sort(key=lambda task: len(task[1]))
    cursor, batch_size, token_budget = 0, args.batch_size, args.token_budget
    while cursor < len(tasks):
        batch, longest = [], 0
        while cursor < len(tasks) and len(batch) < batch_size:
            task = tasks[cursor]
            next_longest = max(longest, len(task[1]) + 2)
            if batch and next_longest * (len(batch) + 1) > token_budget:
                break
            batch.append(task)
            longest = next_longest
            cursor += 1
        tokens = tokenizer([item[1] for item in batch], padding=True,
                           return_special_tokens_mask=True, return_tensors="pt")
        residues = tokens["attention_mask"].bool() & ~tokens.pop("special_tokens_mask").bool()
        if residues.sum(dim=1).tolist() != [len(item[1]) for item in batch]:
            raise ValueError("Tokenizer must preserve one token per amino acid")
        weights = torch.zeros_like(tokens["attention_mask"], dtype=torch.float32)
        for row, (_, _, weight) in enumerate(batch):
            weights[row, residues[row]] = torch.as_tensor(weight, dtype=torch.float32)
        inputs = {key: value.to(device) for key, value in tokens.items()}
        weights = weights.to(device)
        try:
            pooled = pooled_forward(model, inputs, weights).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            cursor -= len(batch)
            batch_size = max(1, len(batch) // 2)
            token_budget = min(token_budget, batch_size * longest)
            del inputs, weights
            torch.cuda.empty_cache()
            warnings.warn(f"CUDA OOM: retrying with batch_size={batch_size}, token_budget={token_budget}")
            continue
        for row, (index, _, _) in enumerate(batch):
            totals[index] += pooled[row]
        del inputs, weights, pooled
    return [total / len(sequence) for total, (_, sequence) in zip(totals, records)], window_counts


def connect_store(path, signature):
    """One writer, transactional commits, and strict provenance checks on resume."""
    database = sqlite3.connect(path)
    try:
        database.execute("PRAGMA journal_mode=DELETE")
        database.execute("PRAGMA synchronous=FULL")
        database.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS features (
                sequence_sha256 TEXT PRIMARY KEY,
                length INTEGER NOT NULL,
                windows INTEGER NOT NULL,
                pooled BLOB NOT NULL
            );
        """)
        encoded = json.dumps(signature, sort_keys=True)
        previous = database.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if previous is not None and previous[0] != encoded:
            raise ValueError("Existing feature provenance differs; use a new output directory")
        if previous is None:
            if database.execute("SELECT count(*) FROM features").fetchone()[0]:
                raise ValueError("Cannot resume a populated database without feature provenance")
            with database:
                database.execute("INSERT INTO metadata VALUES ('signature', ?)", (encoded,))
        return database
    except Exception:
        database.close()
        raise


def write_manifest(directory, manifest):
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/Prot-Inst-OOD")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for mean.sqlite and manifest.json")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="Hugging Face model ID or local model directory")
    parser.add_argument("--revision", default="main", help="Hugging Face model revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--window", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--commit-every", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.window <= 1022 or not 0 <= args.overlap < args.window:
        parser.error("Require 1 <= window <= 1022 and 0 <= overlap < window")
    if min(args.batch_size, args.commit_every) < 1 or args.token_budget < args.window + 2:
        parser.error("Batch/commit sizes must be positive; token budget must fit one window plus special tokens")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu --compute-dtype float32 for CPU extraction")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use a single process: concurrent writers to mean.sqlite are not supported")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    records, sources = load_sequences(args.dataset_dir)
    print(f"INPUT_READY unique_sequences={len(records)}", flush=True)
    pretrained = {"revision": args.revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **pretrained)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False,
                                     torch_dtype=getattr(torch, args.compute_dtype), **pretrained)
    if model.config.hidden_size != 1280 or model.config.num_hidden_layers != 33:
        raise ValueError("Expected ESM2-650M: hidden_size=1280 and num_hidden_layers=33")
    local_model = Path(args.model)
    signature = {
        "format_version": 1,
        "sources": sources,
        "model": relative_path(local_model) if local_model.is_dir() else args.model,
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "checkpoint_files": ({path.name: file_sha256(path) for path in sorted(local_model.iterdir())
                              if path.is_file() and path.suffix in (".json", ".txt", ".bin", ".safetensors")}
                             if local_model.is_dir() else {}),
        "layers": list(range(34)), "hidden_size": 1280,
        "layer_semantics": "HF hidden-state indices; 0=embedding output, 33=final normalized output",
        "pooling": "residue mean; CLS/EOS/padding excluded; overlapping windows weighted by reciprocal coverage",
        "statistic": "mean", "storage_dtype": "<f2",
        "compute_dtype": args.compute_dtype,
        "window": args.window, "overlap": args.overlap,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    database = connect_store(args.output / "mean.sqlite", signature)
    try:
        completed = {row[0] for row in database.execute("SELECT sequence_sha256 FROM features")}
        if not completed.issubset({digest for digest, _ in records}):
            raise ValueError("Database contains sequences outside the requested dataset")
        manifest = {
            "status": "in_progress", "records": len(completed), "expected_records": len(records),
            "mean_database": "mean.sqlite", "key": "sequence_sha256",
            "shape": [34, 1280], "storage_dtype": "float16",
        }
        write_manifest(args.output, manifest)
        if len(completed) < len(records):
            model.requires_grad_(False).eval().to(args.device)
        pending, written, started = [], 0, time.monotonic()

        def flush():
            nonlocal written
            if not pending:
                return
            values, window_counts = encode(pending, model, tokenizer, args.device, args)
            with database:
                for (digest, sequence), value, windows in zip(pending, values, window_counts):
                    stored = value.astype("<f2")
                    if stored.shape != (34, 1280) or not np.isfinite(stored).all():
                        raise ValueError(f"Invalid feature or float16 overflow for {digest}")
                    database.execute("INSERT INTO features VALUES (?, ?, ?, ?)",
                                     (digest, len(sequence), windows, stored.tobytes(order="C")))
            written += len(pending)
            pending.clear()
            print(f"FEATURES_COMMITTED new={written} total={len(completed) + written}/{len(records)} "
                  f"elapsed_s={time.monotonic() - started:.1f}", flush=True)

        for record in records:
            if record[0] in completed:
                continue
            pending.append(record)
            if len(pending) >= args.commit_every:
                flush()
        flush()
        actual = database.execute("SELECT count(*) FROM features").fetchone()[0]
        if actual != len(records):
            raise ValueError(f"Expected {len(records)} features, found {actual}")
        with database:
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('last_run', ?)",
                             (json.dumps({"status": "complete", "records": actual}),))
        manifest.update(status="complete", records=actual)
        write_manifest(args.output, manifest)
        print(f"EXTRACTION_COMPLETE records={actual} output={relative_path(args.output)}", flush=True)
    finally:
        database.close()


if __name__ == "__main__":
    main()
