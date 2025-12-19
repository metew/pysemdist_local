#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pysemdist.config import Config
from pysemdist.io_utils import load_petitions, ensure_dirs
from pysemdist.preprocess import preprocess_df
from pysemdist.embed import embed_texts
from pysemdist.cluster import cluster_embeddings_hdbscan
from pysemdist.label import label_cluster
from pysemdist.meta import meta_cluster
from pysemdist.utils import compute_exemplars
from pysemdist.llm import OpenAICompatClient
from pysemdist.summarize import LLMSummarizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="End-to-end local clustering + LLM summarization (file input only, no DB)"
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input .json/.jsonl/.csv with fields id,text[,category][,locale]",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for parquet artifacts",
    )
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--target_cluster_size", type=int, default=200)
    ap.add_argument("--n_meta", type=int, default=40)

    # --- HDBSCAN / dimensionality params (your old knobs) ---
    ap.add_argument(
        "--hdb_metric",
        default="euclidean",
        help="Distance metric for HDBSCAN (default: euclidean)",
    )
    ap.add_argument(
        "--hdb_selection",
        default="eom",
        help="HDBSCAN cluster_selection_method (eom or leaf, default: eom)",
    )
    ap.add_argument(
        "--hdb_min_cluster_size",
        type=int,
        default=8,
        help="Minimum cluster size for HDBSCAN (default: 8)",
    )
    ap.add_argument(
        "--hdb_min_samples",
        type=int,
        default=3,
        help="Minimum samples for HDBSCAN density definition (default: 3)",
    )
    ap.add_argument(
        "--pca_n_components",
        type=int,
        default=128,
        help="Optional PCA dimensionality reduction before HDBSCAN (default: 128)",
    )
    ap.add_argument(
        "--subcluster_depth",
        type=int,
        default=0,
        help="Reserved for recursive subclustering depth (0 = no subclustering).",
    )

    # --- LLM (Ollama/OpenAI-compatible) options ---
    ap.add_argument(
        "--llm_base_url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL (Ollama, vLLM, etc.)",
    )
    ap.add_argument(
        "--llm_model",
        default="yi:9b-chat",
        help="Model name served by your endpoint (e.g., yi:9b-chat, llama3, etc.)",
    )
    ap.add_argument(
        "--llm_api_key",
        default="sk-local",
        help="API key if your server requires one (for Ollama, usually ignored)",
    )
    ap.add_argument(
        "--llm_options",
        type=str,
        default=None,
        help='JSON string of runtime options to include in the /chat/completions payload, e.g. \'{"num_thread":6,...}\'',
    )
    ap.add_argument(
        "--max_exemplars",
        type=int,
        default=20,
        help="Max exemplar texts per cluster for summarization.",
    )
    ap.add_argument(
        "--summarize_min_size",
        type=int,
        default=2,
        help="Only summarize clusters with at least this many items.",
    )
    ap.add_argument(
        "--summarize_skip_noise",
        action="store_true",
        default=True,
        help="Skip HDBSCAN noise cluster (-1) when summarizing.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        outdir=Path(args.outdir),
        model_name=args.model,
        batch_size=args.batch_size,
        target_cluster_size=args.target_cluster_size,
        n_meta=args.n_meta,
    )

    ensure_dirs(
        [
            cfg.silver_embeddings,
            cfg.gold_clusters,
            cfg.gold_cluster_summaries,
            cfg.gold_meta_clusters,
        ]
    )

    # 1) Load & preprocess
    df = load_petitions(args.input)
    df = preprocess_df(df)

    # 2) Embed
    print("Embedding texts...")
    X = embed_texts(df["text_norm"].tolist(), model_name=cfg.model_name, batch_size=cfg.batch_size)
    df["embedding"] = list(X)

    # 3) Cluster with HDBSCAN (using your CLI overrides)
    print("Clustering with HDBSCAN...")
    dfc = cluster_embeddings_hdbscan(
        df,
        embedding_col="embedding",
        min_cluster_size=args.hdb_min_cluster_size,
        min_samples=args.hdb_min_samples,
        metric=args.hdb_metric,
        cluster_selection_method=args.hdb_selection,
        pca_n_components=args.pca_n_components,
    )
    # NOTE: args.subcluster_depth is parsed but not used here.
    # When it's 0 (your default), that's equivalent to no recursive subclustering.

    # 4) Save per-row clusters (without embeddings) to gold
    dfc_out = dfc.drop(columns=["embedding"])
    clusters_path = cfg.gold_clusters
    dfc_out.to_parquet(clusters_path, index=False)
    print(f"Wrote cluster assignments → {clusters_path}")

    # 5) Build centroids and meta-clusters
    centroids_rows = []
    for cid, dfc_group in dfc.groupby("cluster_id"):
        emb = np.vstack(dfc_group["embedding"].to_list())
        centroid = emb.mean(axis=0)
        centroids_rows.append({"cluster_id": int(cid), "centroid": centroid})

    centroids_df = pd.DataFrame(centroids_rows)
    meta_df = meta_cluster(centroids_df, n_meta=cfg.n_meta)
    meta_path = cfg.gold_meta_clusters
    meta_df.to_parquet(meta_path, index=False)
    print(f"Wrote meta-clusters → {meta_path}")

    # 6) LLM-backed cluster summaries
    client = OpenAICompatClient(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
    )
    if args.llm_options:
        client.options = args.llm_options

    summarizer = LLMSummarizer(client=client)

    summary_rows = []
    for cid, dfc_group in dfc.groupby("cluster_id"):
        cid_int = int(cid)
        if args.summarize_skip_noise and cid_int == -1:
            continue
        if len(dfc_group) < args.summarize_min_size:
            continue

        # Human label + keywords from KeyBERT
        label, keywords = label_cluster(dfc_group)

        # Exemplars (by cluster_score)
        exemplar_ids = compute_exemplars(
            dfc_group, n=min(args.max_exemplars, len(dfc_group))
        )
        exemplar_texts = (
            dfc_group.sort_values("cluster_score", ascending=False)
            .head(len(exemplar_ids))["text_norm"]
            .tolist()
        )

        summary = summarizer.problem_and_tasks(exemplar_texts)

        summary_rows.append(
            {
                "cluster_id": cid_int,
                "size": int(len(dfc_group)),
                "label": label,
                "keywords": keywords,
                "exemplars": exemplar_ids,
                "problem_statement": summary.get("problem_statement", ""),
                "decision_tasks": summary.get("decision_tasks", []),
            }
        )

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
    else:
        df_summary = pd.DataFrame(
            columns=[
                "cluster_id",
                "size",
                "label",
                "keywords",
                "exemplars",
                "problem_statement",
                "decision_tasks",
            ]
        )

    # Normalize decision_tasks to always be a list
    def _normalize_tasks(val):
        if isinstance(val, list):
            return val
        if not val:
            return []
        if isinstance(val, str):
            return [t.strip() for t in val.split("\n") if t.strip()]
        return [str(val)]

    df_summary["decision_tasks"] = df_summary["decision_tasks"].apply(_normalize_tasks)

    df_summary_path = cfg.gold_cluster_summaries
    df_summary.to_parquet(df_summary_path, index=False)
    print(f"Wrote cluster summaries → {df_summary_path}")

    # Also emit a small CSV preview
    sample_csv = df_summary_path.parent / "cluster_summaries_sample.csv"
    df_summary.head(50).to_csv(sample_csv, index=False)
    print(f"Wrote cluster summaries sample CSV → {sample_csv}")

    # 7) Compact JSON preview to stdout
    preview = {
        "clusters": dfc_out[["id", "cluster_id", "cluster_score"]].head(3).to_dict(
            orient="records"
        ),
        "cluster_summaries": df_summary.head(3).to_dict(orient="records"),
        "meta_clusters": meta_df.head(5).to_dict(orient="records"),
    }
    print(json.dumps(preview, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
