# Metadata embeddings

These embeddings were precomputed using OpenAI's `text-embedding-3-large`
model, with 3,072 dimensions per metadata text.

Download the metadata embeddings from the shared
[Google Drive folder](https://drive.google.com/drive/folders/1Rln_d6gRqoss67S57zjGLA6eYU5xDmrI?usp=sharing).

Place the following files directly in this directory:

```text
metadata.json
embeddings.npy
manifest.json
```

Keep the three files from the same download together. Training and retrieval
read this cache without making API requests.
