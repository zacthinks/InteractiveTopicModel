"""
Interactive Topic Model (ITM) package.

A topic modeling framework for social science research with interactive refinement,
hierarchical structures, and undo/redo support.
"""

from .core import InteractiveTopicModel, InteractiveTopic, BasicTopicModel
from .components import (
    SentenceTransformerEmbedder,
    UMAPReducer,
    HDBSCANClusterer,
    e5_base_embedder,
    embedding_scorer,
    tfidf_scorer,
    harmonic_scorer,
)
from .utils import default_vectorizer, custom_vectorizer
from .exceptions import ITMError, IdentityError, NotFittedError
from .data_structures import Edit, SplitPreview
from .visualization import visualize_documents, visualize_topic_hierarchy

__version__ = "0.1.0"

__all__ = [
    "InteractiveTopicModel",
    "InteractiveTopic",
    "BasicTopicModel",
    "SentenceTransformerEmbedder",
    "UMAPReducer",
    "HDBSCANClusterer",
    "e5_base_embedder",
    "embedding_scorer",
    "tfidf_scorer",
    "harmonic_scorer",
    "default_vectorizer",
    "custom_vectorizer",
    "ITMError",
    "IdentityError",
    "NotFittedError",
    "Edit",
    "SplitPreview",
    "visualize_documents",
    "visualize_topic_hierarchy",
]
