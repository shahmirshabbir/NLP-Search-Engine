"""
inference.py

Fast-loading search interface. Import `SemanticSearch` from this module in
your Gradio/Streamlit app or API layer. Does NOT do any training — it expects
build_index.py to have already been run at least once against the target
`directory`.

Example:
    from inference import SemanticSearch
    engine = SemanticSearch(directory="./data")
    results = engine.search("red running shoes for men", k=10)
    # -> ['P10234', 'P10981', ...]  (external ProductIDs)
    urls = engine.search_image_urls("red running shoes for men", k=10)
    # -> ['https://huggingface.co/datasets/.../1/P10234.jpg', ...]
"""

import os
import re

import faiss
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from model_utils import load_embedding_model

# Public dataset repo that hosts product images, keyed by <first_char_of_id>/<id>.jpg
IMAGE_BASE_URL = "https://huggingface.co/datasets/shahmirshabir/Products-Catalog/resolve/main"


def _ensure_nltk_resources():
    for resource in ["stopwords", "wordnet"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[nltk] warning: could not download '{resource}': {e}")


class SemanticSearch:
    def __init__(self, directory: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
        self.model = load_embedding_model(directory, model_name)

        print("[inference] loading index + data...")
        self.index = faiss.read_index(self.index_path)
        self.train_data = pd.read_csv(self.data_path)
        self.product_mapping = joblib.load(self.mapping_path)

        _ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        print(f"[inference] ready — {len(self.train_data)} products, {self.index.ntotal} vectors")

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 10, candidate_pool: int = None) -> list:
        """
        Returns up to k external ProductIDs, ranked by similarity, excluding
        inactive products. Always returns ProductIDs (never raw text),
        regardless of how many results were found on the first pass.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")

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
                break  # can't fetch more than the whole index
            fetch_k *= 2
            attempts += 1

        return result_ids[:k]

    def search_with_metadata(self, query: str, k: int = 10) -> list:
        """Same as search(), but returns full product rows instead of just IDs.
        Useful for a frontend that wants to display name/price/description directly."""
        ids = self.search(query, k)
        subset = self.train_data[self.train_data["ProductID"].isin(ids)]
        # preserve rank order from `ids`
        subset = subset.set_index("ProductID").loc[ids].reset_index()
        return subset.to_dict(orient="records")

    @staticmethod
    def _build_image_url(ext_id) -> str:
        """images/{first_digit_of_id}/{id}.jpg on the Products-Catalog dataset repo."""
        id_str = str(ext_id)
        fdr_id = id_str[0]
        return f"{IMAGE_BASE_URL}/{fdr_id}/{id_str}.jpg"

    def search_image_urls(self, query: str, k: int = 10) -> list:
        """
        Search for the k nearest products to the query text.

        Returns:
            list[str]: image URLs, e.g.
                "https://huggingface.co/datasets/shahmirshabir/Products-Catalog/resolve/main/9/9001.jpg"
        """
        ids = self.search(query, k)
        return [self._build_image_url(pid) for pid in ids]

    # ------------------------------------------------------------------ #
    # mutation (optional — keep out of a read-only portfolio demo if you
    # want a simpler/safer Space; useful if you want to show this off live)
    # ------------------------------------------------------------------ #
    def _preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(words)

    def _combine_attributes(self, row) -> str:
        attributes = [
            row.get("ProductID"), row.get("name"), row.get("age"),
            row.get("gender"), row.get("price"), row.get("usage"),
            row.get("description"), row.get("category"),
        ]
        combined_text = " ".join(
            str(a).strip() for a in attributes if a is not None and str(a).strip()
        )
        return " ".join(dict.fromkeys(combined_text.split()))

    def add_product(self, product: dict) -> str:
        """
        Add a single product. No DataFrame construction needed on the caller's side.

            engine.add_product({
                "ProductID": "P20001",
                "name": "Men's Trail Running Shoes",
                "gender": "Male",
                "price": 1899,
                "usage": "Running",
                "description": "Lightweight breathable trail running shoe",
                "category": "Footwear",
            })

        Internally this still does one encode() call (batch of size 1) —
        there's no extra loop cost versus add_products, just no upfront
        DataFrame-building step on your side.
        """
        self.add_products([product])
        return product.get("ProductID")

    def add_products(self, new_products) -> None:
        """
        Add multiple products in a single batched encode() call — this is the
        efficient path for bulk inserts (one pass over the data, one model call,
        not one call per product).

        Accepts EITHER:
          - a list of dicts: [{"ProductID": "P1", "name": "...", ...}, {...}]
          - a pandas DataFrame with the same columns

        You do NOT need to pre-build a DataFrame yourself — a plain list of
        dicts is converted internally in one O(n) step, same cost as if you'd
        built the DataFrame by hand.
        """
        if isinstance(new_products, pd.DataFrame):
            new_products_df = new_products.copy()
        else:
            new_products_df = pd.DataFrame(list(new_products))

        if new_products_df.empty:
            return

        if "Status" not in new_products_df.columns:
            new_products_df["Status"] = "active"

        new_products_df["combined_text"] = new_products_df.apply(self._combine_attributes, axis=1)
        new_products_df["clean"] = new_products_df["combined_text"].apply(self._preprocess)

        embeddings = self.model.encode(
            new_products_df["clean"].tolist(), convert_to_numpy=True
        ).astype("float32")

        start_idx = len(self.train_data)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        self.train_data = pd.concat([self.train_data, new_products_df], ignore_index=True)
        self.train_data.to_csv(self.data_path, index=False)

        for offset, pid in enumerate(new_products_df["ProductID"]):
            self.product_mapping[pid] = start_idx + offset
        joblib.dump(self.product_mapping, self.mapping_path)

    def delete(self, product_id):
        """Soft-delete: marks the product inactive so it's excluded from future search results."""
        if product_id not in self.product_mapping:
            raise KeyError(f"Unknown ProductID: {product_id}")
        row_idx = self.product_mapping[product_id]
        self.train_data.loc[row_idx, "Status"] = "inactive"
        self.train_data.to_csv(self.data_path, index=False)


# if __name__ == "__main__":
#     # quick manual smoke test
#     engine = SemanticSearch(directory="./data")
#     q = input("Search query: ")
#     for pid in engine.search(q, k=10):
#         print(pid)