# QProtFuse

QProtFuse learns instruction-specific weights to fuse protein language model
features across layers. It jointly trains protein-to-protein and
protein-to-metadata retrieval objectives and supports two-stage retrieval with
metadata-based reranking.

## Repository layout

```text
src/
  model.py        Instruction router and protein/metadata encoders
  loss.py         Masked SigLIP loss and hard-negative weighting
  extract_esm_features.py  Build the ESM residue-mean feature database
  train.py        Joint training on Prot-Inst-OOD
  retrieve.py     PP, PT, and reranking evaluation
data/             Download and preparation instructions
train.sh          Training launcher
retrieve.sh       Launcher for all three retrieval modes
requirements.txt  Python dependencies
```

## Installation

Use Python 3.9–3.12 and a CUDA-capable GPU with bfloat16 support. Create an
environment and install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime. The dependency
versions in `requirements.txt` match the environment used for training.

## Data

Dataset files and precomputed embeddings are distributed separately from the
code. Only preparation READMEs are tracked under `data/`; downloaded and
generated files are ignored by Git.

- [Prot-Inst-OOD](data/Prot-Inst-OOD/README.md): download the dataset from the
  official [Hugging Face repository](https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD).
- [ESM features](data/esm2_650m_all_layers_mean_std/README.md): generate locally,
  or use the Google Drive download link once available.
- [Instruction embeddings](data/protinst_ood_instruction_embeddings/README.md)
  and [metadata embeddings](data/protinst_ood_metadata_embeddings/README.md):
  Google Drive download links will be added to their respective READMEs.

After downloading or generating the artifacts, the directory layout is:

```text
data/
  Prot-Inst-OOD/
    catalytic_activity_OOD.json
    domain_motif_OOD.json
    general_function_OOD.json
    protein_function_OOD.json
  esm2_650m_all_layers_mean_std/
    mean.sqlite
    manifest.json
  protinst_ood_instruction_embeddings/
    instructions.json
    embeddings.npy
    manifest.json
  protinst_ood_metadata_embeddings/
    metadata.json
    embeddings.npy
    manifest.json
```

The ESM2-650M database stores residue-mean features as float16 arrays of shape
`[34, 1280]`, keyed by the SHA-256 hash of each protein sequence. Row 0 is the
embedding output; QProtFuse uses transformer blocks 1–33. Standard-deviation
features are not needed.

Instruction and metadata embeddings are precomputed float32, 3072-dimensional
`text-embedding-3-large` vectors. Training and retrieval read these caches and
do not make API requests. Dataset attribution is provided in
[the dataset README](data/Prot-Inst-OOD/README.md).

### Generate ESM features

From the repository root, run:

```bash
python src/extract_esm_features.py
```

The script loads `facebook/esm2_t33_650M_UR50D` from Hugging Face and extracts
every unique sequence across the four tasks' train and test splits. It writes
`mean.sqlite` and `manifest.json` into `data/esm2_650m_all_layers_mean_std/`,
which is the location used by training and retrieval. Only mean features are
computed. The resulting database contains the sequences needed for this
dataset, rather than an additional external protein collection.

Extraction uses bfloat16 model weights with float32 pooling and float16 storage.
CLS, EOS, and padding tokens are excluded. Sequences longer than 1022 residues
are processed in windows with 128-residue overlap; reciprocal-coverage weights
ensure each residue contributes equally to the protein mean. The final ESM
layer is captured after its encoder LayerNorm.

Use `--batch-size` and `--token-budget` to control GPU memory usage. CUDA OOMs
automatically reduce the window batch size when possible. The script uses one
process and one GPU; launch it with `python`, not `torchrun`. Committed records
are reused on a repeated invocation with the same dataset, model, and extraction
settings. Different provenance is rejected rather than mixing features.

To use a local model directory without downloading weights:

```bash
python src/extract_esm_features.py \
  --model checkpoints/esm2_t33_650M_UR50D --local-files-only
```

The SQLite schema is:

```sql
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE features (
    sequence_sha256 TEXT PRIMARY KEY,
    length INTEGER NOT NULL,
    windows INTEGER NOT NULL,
    pooled BLOB NOT NULL
);
```

Each `pooled` blob is a C-order, little-endian float16 array of shape
`[34, 1280]`. Sequence hashes are calculated from the original ASCII sequence
without normalization, matching the training and retrieval readers.

## Method

Protein features are averaged within seven contiguous groups of ESM blocks.
The instruction router applies
`LayerNorm → Linear(3072, 256) → GELU → Linear(256, 7)` to the instruction
embedding. A softmax with temperature 2 produces the group weights. The weighted
protein feature is projected to a normalized 512-dimensional vector; a separate
MLP maps metadata embeddings to the same space.

Training sums the protein-to-protein and protein-to-metadata masked SigLIP
losses with equal weights. Positive pairs have matching metadata and instruction
cosine similarity greater than 0.65. Negative pairs must also pass the
instruction filter and have metadata cosine similarity below 0.70. The top 128
hard negatives are selected using frozen all-layer-mean ESM similarity and
receive linear weights. Both objectives exclude a record paired with itself.

## Training

Run commands from the repository root, using a fresh output directory:

```bash
bash train.sh runs/qprotfuse 20260929
```

The launcher uses all eligible records in the official Prot-Inst-OOD train split,
without a held-out validation split. Default settings are 30 epochs, four GPUs
on one node, batch size 256 per GPU, and bfloat16 mixed precision. AdamW uses
learning rate `1e-3` for the projection and similarity parameters and `1e-4` for
the instruction router. Edit `--devices` and `--batch-size` in `train.sh` to
adapt to your hardware; these settings determine the global batch size.

Each run saves `config.json`, layer-weight diagnostics, checkpoints every five
epochs, and `last.ckpt`. After 30 epochs, `last.ckpt` corresponds to epoch 29
under zero-based indexing. Checkpoints contain trainable weights and
reconstruction hyperparameters; frozen feature banks remain separate. They do
not contain the optimizer state needed to resume an interrupted run exactly.

## Retrieval

Queries come from the official test split and candidates from the train split
across all four tasks. All modes apply instruction cosine filtering at
`>= 0.65` and report per-task and macro-averaged MRR@10, Recall@1, and Recall@10.

Run all three modes sequentially:

```bash
bash retrieve.sh runs/qprotfuse/last.ckpt runs/qprotfuse/retrieval
```

This creates `pp/`, `pt/`, and `rerank/` output directories, individual logs,
`summary.json` with per-task and macro metrics, and `summary.md` with a macro
comparison table. The output directory must be new.

To run the modes individually:

```bash
# Protein-to-protein: search eligible training records.
python src/retrieve.py --mode pp \
  --checkpoint runs/qprotfuse/last.ckpt \
  --output runs/qprotfuse/retrieval-pp \
  --topk 10 --instruction-cosine-min 0.65

# Protein-to-metadata: search unique metadata from eligible training records.
python src/retrieve.py --mode pt \
  --checkpoint runs/qprotfuse/last.ckpt \
  --output runs/qprotfuse/retrieval-pt \
  --topk 10 --instruction-cosine-min 0.65

# Reranking: reorder the saved PP top-10 using protein-to-metadata scores.
python src/retrieve.py --mode rerank \
  --checkpoint runs/qprotfuse/last.ckpt \
  --protein-results runs/qprotfuse/retrieval-pp \
  --output runs/qprotfuse/retrieval-rerank \
  --instruction-cosine-min 0.65
```

PP retains individual training records, including repeated sequences. PT
deduplicates metadata: a text is eligible if any associated training record
passes the instruction filter. Reranking retains the PP candidate set and sorts
by protein-to-metadata similarity; equal scores preserve the PP order.

Each mode writes `manifest.json`, `metrics.json`, and `predictions.jsonl`.
Paths recorded in retrieval manifests are relative to the repository root.
PP evaluates hits using raw metadata equality; PT and rerank trim leading and
trailing whitespace before comparison.

Use `--candidate-chunk` to control PP candidate encoding batches,
`--query-batch-size` for query batches, and `--metadata-batch-size` for metadata
encoding batches. The optional PP flag `--exclude-identical-sequences` excludes
training proteins identical to the query sequence.
