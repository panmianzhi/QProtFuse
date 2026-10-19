# QProtFuse

Code for training QProtFuse and evaluating protein-to-protein retrieval,
protein-to-metadata retrieval, and two-stage reranking on Prot-Inst-OOD.

## Installation

Use Python 3.9–3.12 and a CUDA-capable GPU with bfloat16 support.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime. Run the following
commands from the repository root.

## Data preparation

1. Download [Prot-Inst-OOD](https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD)
   into `data/Prot-Inst-OOD/`. See its [README](data/Prot-Inst-OOD/README.md)
   for the required filenames.
2. Download both precomputed text embedding folders from
   [Google Drive](https://drive.google.com/drive/folders/1Rln_d6gRqoss67S57zjGLA6eYU5xDmrI?usp=sharing)
   and place them under `data/` as shown below.
3. Generate the ESM features locally:

```bash
python src/extract_esm_features.py
```

The generated `mean.sqlite` is approximately **26–27 GB** (about **25 GiB**)
for the full dataset. Allow at least **35 GB** of free disk space for the
database and model weights. Extraction uses one GPU and supports resuming with
the same settings. ESM features are not provided as a download.

```text
data/
  Prot-Inst-OOD/                        # Four *_OOD.json files
  esm2_650m_all_layers_mean_std/         # Generated mean.sqlite and manifest.json
  protinst_ood_instruction_embeddings/  # instructions.json, embeddings.npy, manifest.json
  protinst_ood_metadata_embeddings/     # metadata.json, embeddings.npy, manifest.json
```

Each data directory contains a README with preparation instructions. Actual data
and embeddings are ignored by Git.

## Training

```bash
bash train.sh runs/qprotfuse 20260929
```

The launcher defaults to 30 epochs and four GPUs. Adjust `--devices` and
`--batch-size` in `train.sh` for your hardware. Use a new output directory;
the trained weights are saved as `runs/qprotfuse/last.ckpt`.

## Retrieval evaluation

Run all three retrieval modes using the same checkpoint:

```bash
bash retrieve.sh runs/qprotfuse/last.ckpt runs/qprotfuse/retrieval
```

Results are saved under `pp/`, `pt/`, and `rerank/`, with per-task and macro
metrics in `summary.json` and a comparison table in `summary.md`.
For individual modes and additional options, run
`python src/retrieve.py --help`.
