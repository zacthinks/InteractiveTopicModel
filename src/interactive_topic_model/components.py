"""Concrete implementations of ITM components."""

from typing import Sequence, Optional, Callable
import numpy as np


class SentenceTransformerEmbedder:
    """Wrapper for sentence-transformers models with optional preprocessing."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize sentence transformer embedder.
        
        Args:
            model_name: Name of the sentence-transformers model.
            preprocessor: Optional function to preprocess text before encoding.
        """
        from sentence_transformers import SentenceTransformer
        
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.preprocessor = preprocessor
    
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.preprocessor:
            texts = [self.preprocessor(t) for t in texts]
        return self.model.encode(list(texts), show_progress_bar=False)
    
    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model_name='{self.name}')"


def e5_base_embedder() -> SentenceTransformerEmbedder:
    """Convenience function for the e5-base-v2 embedding model."""
    return SentenceTransformerEmbedder(
        model_name="intfloat/e5-base-v2",
        preprocessor=lambda x: f"query: {x}",
    )


class UMAPReducer:
    """Wrapper for UMAP dimensionality reduction."""
    
    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: Optional[int] = 42,
    ):
        """
        Initialize UMAP reducer.
        
        Args:
            n_components: Number of dimensions to reduce to.
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance for UMAP.
            metric: Distance metric to use.
            random_state: Random seed for reproducibility.
        """
        import umap
        
        self.name = f"UMAP(n_components={n_components})"
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_jobs=1 if random_state else -1
        )
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings."""
        return self.reducer.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted reducer."""
        return self.reducer.transform(X)
    
    def __repr__(self) -> str:
        return self.name


class HDBSCANClusterer:
    """Wrapper for HDBSCAN clustering."""
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
    ):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum size of clusters.
            min_samples: Number of samples in neighborhood for core points.
            cluster_selection_epsilon: Distance threshold for cluster selection.
            metric: Distance metric to use.
        """
        import hdbscan
        
        self.name = f"HDBSCAN(min_cluster_size={min_cluster_size})"
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
        )
    
    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        """Fit clusterer on data."""
        self.clusterer.fit(X)
        return self
    
    @property
    def labels_(self) -> np.ndarray:
        """Cluster labels for each sample."""
        return self.clusterer.labels_
    
    @property
    def strengths_(self) -> np.ndarray:
        """Membership strength for each sample."""
        return self.clusterer.probabilities_
    
    def __repr__(self) -> str:
        return self.name


# =====================================================================
# Scoring functions
# =====================================================================


def embedding_scorer(
    doc_embedding: Optional[np.ndarray],
    doc_tfidf: Optional[np.ndarray],
    topic_embeddings: Optional[np.ndarray],
    topic_ctfidfs: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute cosine similarity between doc embedding and topic embeddings.
    
    Args:
        doc_embedding: Document embedding vector.
        doc_tfidf: Document TF-IDF vector (unused).
        topic_embeddings: Topic embedding matrix (n_topics, embedding_dim).
        topic_ctfidfs: Topic c-TF-IDF matrix (unused).
        
    Returns:
        Similarity scores array of shape (n_topics,).
    """
    if doc_embedding is None or topic_embeddings is None:
        raise ValueError("embedding_scorer requires doc_embedding and topic_embeddings")
    
    # Normalize document embedding
    d = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-12)
    
    # Normalize topic embeddings
    norms = np.linalg.norm(topic_embeddings, axis=1, keepdims=True) + 1e-12
    T = topic_embeddings / norms
    
    # Compute cosine similarities
    return T @ d


def tfidf_scorer(
    doc_embedding: Optional[np.ndarray],
    doc_tfidf: Optional[np.ndarray],
    topic_embeddings: Optional[np.ndarray],
    topic_ctfidfs: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute cosine similarity between doc TF-IDF and topic c-TF-IDF vectors.
    
    Args:
        doc_embedding: Document embedding vector (unused).
        doc_tfidf: Document TF-IDF vector.
        topic_embeddings: Topic embedding matrix (unused).
        topic_ctfidfs: Topic c-TF-IDF matrix (n_topics, vocab_size).
        
    Returns:
        Similarity scores array of shape (n_topics,).
    """
    if doc_tfidf is None or topic_ctfidfs is None:
        raise ValueError("tfidf_scorer requires doc_tfidf and topic_ctfidfs")
    
    # Normalize document TF-IDF
    d = doc_tfidf / (np.linalg.norm(doc_tfidf) + 1e-12)
    
    # Normalize topic c-TF-IDF vectors
    norms = np.linalg.norm(topic_ctfidfs, axis=1, keepdims=True) + 1e-12
    T = topic_ctfidfs / norms
    
    # Compute cosine similarities
    return T @ d


def harmonic_scorer(
    doc_embedding: Optional[np.ndarray],
    doc_tfidf: Optional[np.ndarray],
    topic_embeddings: Optional[np.ndarray],
    topic_ctfidfs: Optional[np.ndarray],
) -> np.ndarray:
    """
    Harmonic mean of embedding and TF-IDF similarity scores.
    
    Args:
        doc_embedding: Document embedding vector.
        doc_tfidf: Document TF-IDF vector.
        topic_embeddings: Topic embedding matrix (n_topics, embedding_dim).
        topic_ctfidfs: Topic c-TF-IDF matrix (n_topics, vocab_size).
        
    Returns:
        Harmonic mean scores array of shape (n_topics,).
    """
    emb_sims = embedding_scorer(doc_embedding, doc_tfidf, topic_embeddings, topic_ctfidfs)
    tfidf_sims = tfidf_scorer(doc_embedding, doc_tfidf, topic_embeddings, topic_ctfidfs)
    
    # Harmonic mean: 2 / (1/a + 1/b)
    return 2.0 / (1.0 / (emb_sims + 1e-8) + 1.0 / (tfidf_sims + 1e-8))
