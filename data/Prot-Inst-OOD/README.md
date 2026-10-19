# Prot-Inst-OOD

Download the dataset from the official Hugging Face repository:
[TimeRune/Prot-Inst-OOD](https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD).
The dataset was introduced in
[Rethinking Text-based Protein Understanding: Retrieval or LLM?](https://arxiv.org/abs/2505.20354)
and is distributed by the [RAPM authors](https://github.com/IDEA-XL/RAPM).

Run this command from the QProtFuse repository root after installing the dependencies:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TimeRune/Prot-Inst-OOD",
    repo_type="dataset",
    allow_patterns=["*_OOD.json"],
    local_dir="data/Prot-Inst-OOD",
)
PY
```

The directory must then contain:

```text
catalytic_activity_OOD.json
domain_motif_OOD.json
general_function_OOD.json
protein_function_OOD.json
```

Use the `split` field inside each JSON record for train/test membership. Preserve
the supplied records and split labels. The dataset files are not stored in this
Git repository; only these instructions are tracked. Refer to the upstream
dataset card for its MIT license and citation.
