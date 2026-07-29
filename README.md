---
title: Semantic Product Search
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Semantic Product Search

A semantic (meaning-based, not keyword-based) product search demo. Encodes
product metadata with `sentence-transformers/all-MiniLM-L6-v2` and serves
nearest-neighbor search over a FAISS index.

This Space runs entirely on CPU — no GPU is required or used, even though
it's hosted on a ZeroGPU-tier Space (free-tier Gradio Spaces currently only
offer ZeroGPU as the compute option; this app simply never requests a GPU,
so it runs on the Space's base CPU allocation at no GPU-quota cost).

## Repo layout
```
app.py            - Gradio UI + API endpoint
inference.py       - SemanticSearch class: loads index, serves search()
model_utils.py     - shared embedding model loader (HF Hub or local cache)
build_index.py     - offline script to (re)build the FAISS index — run locally, not on the Space
requirements.txt
data/
    my_data.csv
    faiss_index.index
    product_mapping.pkl
```

## API
Programmatic access via `gradio_client`:
```python
from gradio_client import Client
client = Client("your-username/your-space-name")
result = client.predict("red running shoes", 10, api_name="/search")
# -> {"results": {"0": "<image_url>", "1": "<image_url>", ...}}
```
