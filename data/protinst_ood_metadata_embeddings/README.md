# Precomputed metadata embeddings

A Google Drive download link will be added here when the artifacts are
available. The actual embeddings are not included in Git.

Download and extract these files directly into this directory:

```text
metadata.json
embeddings.npy
manifest.json
```

The vectors were generated with `text-embedding-3-large` and stored as a
float32 NumPy array of shape `[N, 3072]`. `metadata.json` contains unique
metadata records; each contiguous `id` identifies the corresponding row in
`embeddings.npy`. The manifest specifies the model and completion status.

Training, protein-to-metadata retrieval, and reranking use this cache without
API requests. Keep the catalog, array, and manifest together from the same
download, preserving the text-to-row mapping.
