"""Data utilities for RoBERTa-SC (Europarl preprocessing and loading)."""

from __future__ import annotations

import os
import pickle
import re
import unicodedata
from typing import List


def _to_ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize_text(s: str) -> str:
    from w3lib.html import remove_tags
    s = _to_ascii(s)
    s = remove_tags(s)
    return re.sub(r"\s+", " ", s).strip()


def extract_sentences(text_path: str, min_chars: int = 100, max_chars: int = 500) -> List[str]:
    """Extract and clean sentences from one Europarl ``.txt`` file."""
    with open(text_path, "r", encoding="utf8") as f:
        raw = f.read()
    sentences = re.findall(r">\s*(.*?)\n", raw)
    out = []
    for s in sentences:
        s = normalize_text(s)
        if min_chars <= len(s) <= max_chars:
            out.append(" ".join(s.split()).lower())
    return out


def build_sentence_corpus(data_dir: str, min_chars: int = 100, max_chars: int = 500) -> List[str]:
    """Build a de-duplicated sentence corpus from a directory of Europarl files."""
    sentences = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".txt"):
            sentences.extend(extract_sentences(os.path.join(data_dir, fn), min_chars, max_chars))
    return sorted(set(sentences))


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
