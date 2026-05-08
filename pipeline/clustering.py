from dataclasses import dataclass
import numpy as np
import pandas as pd
import hdbscan
import umap
from sentence_transformers import SentenceTransformer

from configs.base import DomainConfig


@dataclass
class ClusteringConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    umap_components: int = 15
    umap_random_state: int = 42
    hdbscan_min_cluster_size: int = 25


class ClusteringPipeline:
    def __init__(self, config: DomainConfig, clustering_config: ClusteringConfig = None):
        self.config = config
        self.clustering_config = clustering_config or ClusteringConfig()
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.clustering_config.embedding_model)
        return self._embedder

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        field = self.config.cluster_on
        mask = df[field].notna()
        texts = df.loc[mask, field].tolist()

        print(f"Embedding {len(texts)} '{field}' values...")
        embeddings = self.embedder.encode(texts)

        print(f"Reducing dimensions (UMAP: {len(embeddings[0])} → {self.clustering_config.umap_components})...")
        reducer = umap.UMAP(
            n_components=self.clustering_config.umap_components,
            random_state=self.clustering_config.umap_random_state,
        )
        reduced = reducer.fit_transform(embeddings)

        print("Clustering (HDBSCAN)...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.clustering_config.hdbscan_min_cluster_size
        )
        labels = clusterer.fit_predict(reduced)

        cluster_col = f"{field}_cluster"
        df = df.copy()
        df[cluster_col] = np.nan
        df.loc[mask, cluster_col] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"Found {n_clusters} clusters, {n_noise} noise points.")

        return df
