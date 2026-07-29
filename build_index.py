"""
build_index.py

Run this script whenever you need to (re)build the semantic search index
from scratch, e.g.:
  - first-time setup
  - you've replaced my_data.csv with a fresh/larger product catalog
  - you want to fully rebuild after a lot of incremental addProducts() calls

This is NOT meant to run on every server start — it's an offline/batch step.
inference.py loads whatever this script produces.

Usage:
    python build_index.py --directory ./data
"""

import argparse
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


def ensure_nltk_resources():
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[nltk] warning: could not download '{resource}': {e}")


def combine_attributes(row) -> str:
    """
    Build one text blob per product out of its metadata fields.
    Adjust the field list here if your CSV schema changes.
    """
    attributes = [
        row.get("ProductID"),
        row.get("name"),
        row.get("age"),
        row.get("gender"),
        row.get("price"),
        row.get("usage"),
        row.get("description"),
        row.get("category"),
    ]

    combined_text = " ".join(
        str(attribute).strip()
        for attribute in attributes
        if attribute is not None and str(attribute).strip()
    )

    # de-dupe words while preserving order (matches original behavior)
    return " ".join(dict.fromkeys(combined_text.split()))


def preprocess(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def build(directory: str, model_name: str):
    data_path = os.path.join(directory, "my_data.csv")
    index_path = os.path.join(directory, "faiss_index.index")
    mapping_path = os.path.join(directory, "product_mapping.pkl")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Expected product CSV at {data_path}. "
            "Place your catalog there before running build_index.py."
        )

    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    print("[build] loading catalog...")
    df = pd.read_csv(data_path)

    # keep only active products in the trained index/dataframe
    if "Status" in df.columns:
        df = df[df["Status"] != "inactive"].reset_index(drop=True)
    else:
        df["Status"] = "active"

    if "ProductID" not in df.columns:
        raise ValueError("CSV must contain a 'ProductID' column (this is the external ID returned by search).")

    print(f"[build] {len(df)} active products found")

    print("[build] combining + cleaning text fields...")
    df["combined_text"] = df.apply(combine_attributes, axis=1)
    df["clean"] = df["combined_text"].apply(lambda t: preprocess(t, lemmatizer, stop_words))

    print("[build] loading embedding model...")
    model = load_embedding_model(directory, model_name)

    print("[build] encoding products (this is the slow step)...")
    embeddings = model.encode(
        df["clean"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")

    d = embeddings.shape[1]
    print(f"[build] embedding matrix shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # ProductID -> row position mapping. Row position in this dataframe
    # matches the vector's position in the FAISS index, since both are
    # built from the same ordered iteration.
    product_mapping = {pid: idx for idx, pid in enumerate(df["ProductID"])}

    print("[build] saving artifacts...")
    df.to_csv(data_path, index=False)
    faiss.write_index(index, index_path)
    joblib.dump(product_mapping, mapping_path)

    print(f"[build] done. index: {index_path} | mapping: {mapping_path} | data: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the semantic search FAISS index from my_data.csv")
    parser.add_argument("--directory", default="./data", help="Directory containing my_data.csv and where index artifacts will be saved")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face Hub model id to use for embeddings")
    args = parser.parse_args()

    build(args.directory, args.model_name)
