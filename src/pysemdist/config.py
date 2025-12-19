from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    outdir: Path
    model_name: str = "intfloat/e5-base-v2"
    batch_size: int = 128
    target_cluster_size: int = 200  # used to compute k ≈ N/target
    n_meta: int = 40                # number of meta clusters/themes

    # file paths
    @property
    def silver_embeddings(self) -> Path:
        return self.outdir / "silver" / "embeddings.parquet"

    @property
    def gold_clusters(self) -> Path:
        return self.outdir / "gold" / "clusters.parquet"

    @property
    def gold_cluster_summaries(self) -> Path:
        return self.outdir / "gold" / "cluster_summaries.parquet"

    @property
    def gold_meta_clusters(self) -> Path:
        return self.outdir / "gold" / "meta_clusters.parquet"
    
    def __post_init__(self):
        (self.outdir / "silver").mkdir(parents=True, exist_ok=True)
        (self.outdir / "gold").mkdir(parents=True, exist_ok=True)