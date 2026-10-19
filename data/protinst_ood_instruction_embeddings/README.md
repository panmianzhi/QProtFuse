# Precomputed instruction embeddings

A Google Drive download link will be added here when the artifacts are
available. The actual embeddings are not included in Git.

Download and extract these files directly into this directory:

```text
instructions.json
embeddings.npy
manifest.json
```

The vectors were generated with `text-embedding-3-large` and stored as a
float32 NumPy array of shape `[N, 3072]`. `instructions.json` contains the
instruction records; each contiguous `id` identifies the corresponding row
in `embeddings.npy`. The manifest specifies the model and completion status.

Training and retrieval read this cache directly and make no API requests.
Keep the catalog, array, and manifest together from the same download.
