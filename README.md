# QProtFuse

Code for training QProtFuse and evaluating protein-to-protein retrieval,
protein-to-metadata retrieval, and two-stage reranking on Prot-Inst-OOD.

## Data preparation

We precompute protein and text representations for efficient training and
retrieval. These representations can alternatively be computed on the fly using
the corresponding encoders; the provided scripts use precomputed caches.

Download and preparation instructions are in the READMEs for
[Prot-Inst-OOD](data/Prot-Inst-OOD/README.md),
[ESM features](data/esm2_650m_all_layers_mean_std/README.md),
[instruction embeddings](data/protinst_ood_instruction_embeddings/README.md), and
[metadata embeddings](data/protinst_ood_metadata_embeddings/README.md).

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
