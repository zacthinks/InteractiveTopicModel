"""Protocol definitions for ITM components."""

from typing import Protocol, Any, Sequence
import numpy as np


class Embedder(Protocol):
    """Maps documents to dense vectors."""
    
    def encode(self, docs: Sequence[str]) -> np.ndarray:
        """
        Encode documents to embeddings.
        
        Args:
            docs: Sequence of text documents.
            
        Returns:
            Array of shape (n_docs, embedding_dim).
        """
        ...


class Reducer(Protocol):
    """Optional dimensionality reduction over embeddings."""
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit reducer and transform embeddings.
        
        Args:
            X: Embeddings array of shape (n_samples, n_features).
            
        Returns:
            Reduced embeddings of shape (n_samples, n_components).
        """
        ...
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted reducer.
        
        Args:
            X: Embeddings array of shape (n_samples, n_features).
            
        Returns:
            Reduced embeddings of shape (n_samples, n_components).
        """
        ...


class Clusterer(Protocol):
    """Assigns cluster labels and membership strengths."""
    
    def fit(self, X: np.ndarray) -> Any:
        """
        Fit clusterer on data.
        
        Args:
            X: Data array of shape (n_samples, n_features).
            
        Returns:
            Self.
        """
        ...
    
    @property
    def labels_(self) -> np.ndarray:
        """Cluster labels for each sample. -1 indicates outlier."""
        ...
    
    @property
    def strengths_(self) -> np.ndarray:
        """Membership strength for each sample (higher = more confident)."""
        ...


class Scorer(Protocol):
    """Computes a document's score for set of topic representations."""
    
    def __call__(
        self,
        doc_embedding: np.ndarray | None,
        doc_tfidf: np.ndarray | None,
        topic_embeddings: np.ndarray | None,
        topic_ctfidfs: np.ndarray | None,
    ) -> np.ndarray:
        """
        Score document against multiple topics.
        
        Args:
            doc_embedding: Document embedding vector.
            doc_tfidf: Document TF-IDF vector.
            topic_embeddings: Topic embedding matrix (n_topics, embedding_dim).
            topic_ctfidfs: Topic c-TF-IDF matrix (n_topics, vocab_size).
            
        Returns:
            Scores array of shape (n_topics,).
        """
        ...
