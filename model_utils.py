"""
Shared embedding model loader used by both build_index.py and inference.py.

Behavior:
- If a local copy of the model already exists at `<directory>/<model_name_folder>`,
  load it from disk (fast, no network call).
- Otherwise, download it from the Hugging Face Hub by name and cache it locally
  inside `directory` for next time.

This means the code works identically whether you're running it fresh on a new
machine (e.g. a HF Space, which has no local model yet) or on a machine where
you've already downloaded the model once.
"""

import os
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(directory: str, model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence embedding model.

    Args:
        directory: base directory where a local model cache may live
                   (e.g. "<directory>/all-MiniLM-L6-v2").
        model_name: Hugging Face Hub model id to fall back to / download.

    Returns:
        A loaded SentenceTransformer instance.
    """
    local_folder_name = model_name.split("/")[-1]
    local_path = os.path.join(directory, local_folder_name)

    if os.path.isdir(local_path) and os.listdir(local_path):
        print(f"[model] loading local model from {local_path}")
        return SentenceTransformer(local_path)

    print(f"[model] local model not found at {local_path}, downloading '{model_name}' from Hugging Face Hub")
    model = SentenceTransformer(model_name)

    # cache it locally so future runs (and Spaces cold-starts backed by a
    # persistent volume) don't need to re-download
    os.makedirs(local_path, exist_ok=True)
    model.save(local_path)
    print(f"[model] cached model to {local_path}")

    return model
