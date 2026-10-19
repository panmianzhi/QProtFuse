# ESM2 residue-mean features

This directory holds the feature database used by QProtFuse:

```text
mean.sqlite
manifest.json
```

The feature files are not included in Git. A Google Drive download link will
be added here when the artifacts are available.

Alternatively, after downloading Prot-Inst-OOD, generate the database from the
repository root:

```bash
python src/extract_esm_features.py
```

The extractor uses `facebook/esm2_t33_650M_UR50D` and processes unique protein
sequences from all four tasks' train and test splits. Each `features.pooled`
blob is a little-endian float16 array of shape `[34, 1280]`: row 0 is the
embedding output and rows 1–33 are transformer features. Rows are keyed by
the SHA-256 hash of the original ASCII sequence. No standard-deviation
features are required or generated.

See the root README for extraction settings and the SQLite schema. Put files
directly in this directory, without an additional nested archive directory.
