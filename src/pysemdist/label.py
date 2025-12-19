from __future__ import annotations
from typing import List, Tuple
from collections import Counter

import pandas as pd
from keybert import KeyBERT

_KW_MODEL: KeyBERT | None = None


def _get_kw_model() -> KeyBERT:
    global _KW_MODEL
    if _KW_MODEL is None:
        _KW_MODEL = KeyBERT()
    return _KW_MODEL


def extract_keywords(texts: List[str], top_k: int = 10) -> List[str]:
    """Extract top keywords from a list of texts using KeyBERT."""
    if not texts:
        return []

    kw_model = _get_kw_model()
    joined = "\n".join(texts)
    kws = kw_model.extract_keywords(
        joined,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_k,
    )
    return [kw for kw, _ in kws]


def representative_texts(df_cluster: pd.DataFrame, n: int = 10) -> List[str]:
    """Return up to n representative texts from a cluster."""
    if df_cluster.empty:
        return []

    if "cluster_score" in df_cluster.columns:
        df_cluster = df_cluster.sort_values("cluster_score", ascending=False)

    samp = df_cluster.head(n)
    if "text_norm" not in samp.columns:
        raise KeyError("Expected 'text_norm' column in cluster DataFrame")

    return samp["text_norm"].tolist()


def label_cluster(df_cluster: pd.DataFrame) -> Tuple[str, List[str]]:
    """Generate a simple human-readable label and keywords for a cluster."""
    texts = representative_texts(df_cluster, n=min(30, len(df_cluster)))
    kws = extract_keywords(texts, top_k=10)

    # Simple label from most common keywords
    words = [w for kw in kws for w in kw.split()]
    common = [w for w, _ in Counter(words).most_common(3)]
    label = " ".join(common).title() if common else "General Issue"

    return label, kws
