# ESM features

Generate these features locally after downloading Prot-Inst-OOD. No feature
download is provided.

From the repository root, run:

```bash
python src/extract_esm_features.py
```

This creates `mean.sqlite` and `manifest.json` in this directory.
The full dataset has 298,925 unique sequences; the resulting database is
approximately **26–27 GB** (about **25 GiB**). Allow at least **35 GB** of free
disk space for the database and ESM2-650M model weights.

Extraction uses one GPU and computes mean features only. Repeat the same
command to resume an interrupted run. Use `--batch-size` and `--token-budget`
to adjust GPU memory usage, or `--model checkpoints/esm2_t33_650M_UR50D
--local-files-only` to use a local model. Run the script with `--help` for
all options.
