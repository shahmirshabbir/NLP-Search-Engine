"""
inference.py
Fast-loading search interface. Import `SemanticSearch` from this module in
your Gradio/Streamlit app or API layer. Does NOT do any training — it expects
build_index.py to have already been run at least once against the target
`directory`.
"""

import os
import spaces

import torch
from sentence_transformers import SentenceTransformer

import faiss
import joblib
import pandas as pd


# Module-level reference so the standalone @spaces.GPU function can reach
# the model without needing `self` (and the rest of the heavy class) to be
# pickled across the ZeroGPU worker boundary.
_MODEL = None
IMAGE_BASE_URL = "https://huggingface.co/datasets/shahmirshabir/Products-Catalog/resolve/main"

@spaces.GPU
def _encode_on_gpu(query: str):
    """Runs inside a real ZeroGPU-attached worker process. Keep this
    function's args simple/picklable — no `self`, no faiss index, no df."""
    _MODEL.to("cuda")
    return _MODEL.encode([query], convert_to_numpy=True).astype("float32")


class SemanticSearch:
    def __init__(self, directory: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        global _MODEL

        self.directory = directory
        self.data_path = os.path.join(directory, "my_data.csv")
        self.index_path = os.path.join(directory, "faiss_index.index")
        self.mapping_path = os.path.join(directory, "product_mapping.pkl")

        for path in (self.data_path, self.index_path, self.mapping_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run build_index.py first to create the index artifacts."
                )

        print("[inference] loading model...")
        self.model = SentenceTransformer(model_name)
        _MODEL = self.model  # expose to the standalone GPU function

        self.index = faiss.read_index(self.index_path)
        self.train_data = pd.read_csv(self.data_path)
        self.product_mapping = joblib.load(self.mapping_path)

        print(f"[inference] ready — {len(self.train_data)} products, {self.index.ntotal} vectors")
    @staticmethod
    def _build_image_url(ext_id) -> str:
        """images/{first_digit_of_id}/{id}.jpg on the Products-Catalog dataset repo."""
        id_str = str(ext_id)
        fdr_id = id_str[0]
        return f"{IMAGE_BASE_URL}/{fdr_id}/{id_str}.jpg"
        
    def search(self, query: str, k: int = 10, candidate_pool: int = None) -> list:
        """
        Returns up to k external ProductIDs, ranked by similarity, excluding
        inactive products.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = _encode_on_gpu(query)

        fetch_k = candidate_pool or k
        seen = set()
        result_ids = []
        attempts = 0
        max_attempts = 5

        while len(result_ids) < k and attempts < max_attempts:
            fetch_k = min(fetch_k, self.index.ntotal)
            if fetch_k == 0:
                break

            _, indices = self.index.search(query_vector, fetch_k)

            for i in indices[0]:
                if i == -1 or i in seen:
                    continue
                seen.add(i)
                row = self.train_data.iloc[i]
                if row.get("Status", "active") != "inactive":
                    result_ids.append(row["ProductID"])
                if len(result_ids) >= k:
                    break

            if fetch_k >= self.index.ntotal:
                break
            fetch_k *= 2
            attempts += 1

        product_ids = result_ids[:k]
        urls = []

        for pid in product_ids:
            urls.append(self._build_image_url(pid))
            
        return urls