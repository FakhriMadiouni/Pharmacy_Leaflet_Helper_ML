"""Turns text into vectors (PROVIDED, nothing to write here).

Two encoders with the same interface, like the mock/real providers of Atelier 1:
  - "mock"      : counts words (hashed). Works offline, instantly. Good enough to test the code.
  - "fastembed" : the real models of Atelier 4 (MiniLM or mpnet), stored in ~/.cache/fastembed.
Every vector has length 1, so a dot product between two vectors is their cosine.
"""
import csv
import math
import os
import re
import zlib
from collections import Counter
from functools import lru_cache

import numpy as np

from app.config import DATA, EMBEDDING_MODE, EMBEDDING_MODEL

STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "is", "are", "it", "of", "to", "for", "and", "or", "in",
    "on", "with", "do", "does", "can", "what", "which", "how", "should", "you", "have", "has",
    "that", "this", "be", "at", "by", "from", "as", "so", "if", "we", "our", "your", "please",
    "there", "some", "any", "about",
}
DIM = 1024


def _stem(word: str) -> str:
    for ending in ("ing", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[: -len(ending)]
    return word


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


class MockEncoder:
    """Counts words; rare words (found in few passages) count more than common ones."""
    name = "mock"

    def __init__(self):
        with open(DATA / "passages.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        texts = [row["text"] for row in rows]
        self.n = len(texts)
        self.df = Counter(w for t in texts for w in set(content_words(t)))

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, count in Counter(content_words(text)).items():
                if word in self.df:
                    out[row, zlib.crc32(word.encode()) % DIM] += (
                        (1 + math.log(count)) * math.log(1 + self.n / self.df[word]))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedEncoder:
    MODELS = {
        "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    }

    def __init__(self, key: str):
        from fastembed import TextEmbedding  # imported here so mock mode needs no install

        self.name = key
        cache = os.path.expanduser("~/.cache/fastembed")  # keep it: no re-download
        self.model = TextEmbedding(model_name=self.MODELS[key], cache_dir=cache)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def get_encoder(mode: str | None = None, model: str | None = None):
    mode = mode or EMBEDDING_MODE
    if mode == "fastembed":
        return FastEmbedEncoder(model or EMBEDDING_MODEL)
    return MockEncoder()


if __name__ == "__main__":
    enc = get_encoder()
    a, b, c = enc.encode([
        "My tablet is white and round",
        "The pill I found is round and white",
        "Where is the nearest bus stop?",
    ])
    print("encoder:", enc.name, "| vector length:", a.shape[0])
    print("white-pill vs white-pill :", round(float(a @ b), 3))
    print("white-pill vs bus stop   :", round(float(a @ c), 3))
