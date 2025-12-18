============================================================
PETITION CLUSTERING & SUMMARIZATION PIPELINE
============================================================

OVERVIEW
--------
This project builds a full end-to-end pipeline that:
1. Loads petitions from a CSV or JSON/JSONL file.
2. Normalizes and embeds petition text using a local SentenceTransformer model.
3. Clusters similar petitions into semantic groups.
4. Summarizes each cluster into a generic problem statement and decision-maker task list
   using a local LLM (served through Ollama, vLLM, or any OpenAI-compatible API).
5. Optionally groups clusters into meta-themes.

OUTPUT STRUCTURE (MEDALLION ARCHITECTURE)
-----------------------------------------
data/
 ├── bronze/   - raw inputs (optional)
 ├── silver/   - cleaned text + embeddings
 └── gold/     - cluster, summary, and meta-cluster artifacts


REQUIREMENTS
------------
- Python 3.10+
- Virtual environment
- Packages listed in requirements.txt
- Ollama (or another OpenAI-compatible local endpoint)
- Models:
    * Embeddings: intfloat/e5-base-v2 or bge-large-en
    * LLM Summarizer: yi:9b-chat (served via Ollama at http://localhost:11434/v1)


INSTALLATION
------------
# clone repo
git clone <your_repo_url>
cd <repo_name>

# create virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# optional editable install
pip install -e .


OLLAMA SETUP
------------
Remember to Start your ollama server!

    ollama serve

Install Ollama (https://ollama.ai) and pull the summarization model:

    ollama pull yi:9b-chat

Ollama runs an OpenAI-compatible API by default at:
    http://localhost:11434/v1


INPUT FORMAT
------------
Your petitions CSV or JSON file must include:
    id,text

Optional columns:
    category,locale

Example data/zbronze/petitions.csv:
-------------------------------------------------
id,category,text
1,transport,The petition urges better public transport options in the suburban area. Buses are infrequent and always crowded.
2,water,This petition is about recurring water outages in the southern sector. Clean water access is a basic necessity.
3,hockey,This petition is about jason wants to own an AHL hockey team.
-------------------------------------------------


SAMPLE COMMANDS
---------------
1) Full pipeline for all petitions
-------------------------------------------------
# from repo root
python scripts/run_pipeline.py \
  --csv_path data/zbronze/petitions.csv \
  --outdir data \
  --model intfloat/e5-base-v2 \
  --llm_base_url http://localhost:11434/v1 \
  --llm_model yi:9b-chat \
  --llm_api_key sk-local \
  --target_cluster_size 20 \
  --subcluster_depth 2 \
  --sub_min_size 25 \
  --sub_min_improvement 0.03 \
  --max_exemplars 12
-------------------------------------------------

2) Process a single category (incremental batch)
-------------------------------------------------
python scripts/run_pipeline.py \
  --csv_path data/zbronze/petitions.csv \
  --outdir data \
  --model intfloat/e5-base-v2 \
  --llm_base_url http://localhost:11434/v1 \
  --llm_model yi:9b-chat \
  --llm_api_key sk-local \
  --max_exemplars 12 \
  --categories hockey
-------------------------------------------------

3) Create sample CSVs from Parquet outputs (head 50)
-------------------------------------------------
import pyarrow.dataset as ds
import pandas as pd, os

os.makedirs("data/samples", exist_ok=True)
for name in ["clusters", "cluster_summaries", "meta_clusters"]:
    df = ds.dataset(f"data/gold/{name}.parquet").head(50).to_pandas()
    df.drop(columns=["embedding"], errors="ignore").to_csv(
        f"data/samples/{name}_sample.csv", index=False)
-------------------------------------------------


OUTPUT FILES
------------
After a successful run, you’ll see:

data/gold/
 ├── clusters.parquet            - per-petition cluster assignments
 ├── cluster_summaries.parquet   - per-cluster problem statements & tasks
 └── meta_clusters.parquet       - higher-level themes (meta-clusters)

Example schema:
-------------------------------------------------
clusters.parquet:
  id, category, lang, text_norm, cluster_id, cluster_score, embedding

cluster_summaries.parquet:
  cluster_id, size, keywords, label, problem_statement, decision_tasks, exemplars

meta_clusters.parquet:
  meta_id, theme_label, meta_problem_statement, canonical_tasks, included_cluster_ids
-------------------------------------------------


INCREMENTAL PROCESSING
----------------------
You can run the pipeline repeatedly with new petition batches:
- Use --categories to process one domain at a time.
- Append new embeddings and update centroids incrementally.
- Merge clusters periodically to maintain global quality.


NOTES
-----
- The LLM summarizer assumes an OpenAI-compatible /chat/completions endpoint.
- When using Ollama locally, you can pass --llm_api_key sk-local as a placeholder.
- The --max_exemplars argument controls how many sample petitions are summarized per cluster.


TROUBLESHOOTING
---------------
- "ModuleNotFoundError: petitions" → run from repo root or install editable:
      pip install -e .
- PCA n_components error → occurs on tiny datasets; use the latest cluster.py fix.
- Tokenizers / OpenMP warnings → harmless, come from SentenceTransformers.


SUMMARY
-------
This repository provides a lightweight, end-to-end semantic clustering and summarization pipeline for petitions:

Input  : petition text (CSV/JSON)
Process: embedding → clustering → summarization
Output : structured insights for decision-makers

Powered by local inference:
- SentenceTransformer (e5-base-v2)
- Ollama (Yi-9B Chat) as LLM summarizer

Happy clustering! 🚀
============================================================
