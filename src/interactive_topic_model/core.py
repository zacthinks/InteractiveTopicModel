"""Core classes for Interactive Topic Model."""

from typing import Dict, List, Optional, Tuple, Callable, Any, Type, Union, Sequence, Iterable
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import copy
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from .protocols import Embedder, Reducer, Clusterer, Scorer
from .components import (
    SentenceTransformerEmbedder,
    UMAPReducer,
    HDBSCANClusterer,
    embedding_scorer,
    tfidf_scorer,
    harmonic_scorer,
)
from .utils import ClassTfidfTransformer, default_vectorizer
from .exceptions import ITMError, IdentityError, NotFittedError
from .data_structures import Edit, SplitPreview


# =====================================================================
# Undoable decorator
# =====================================================================


def undoable(func: Callable) -> Callable:
    """
    Decorator for operations that create undo/redo history.
    
    The decorated function should return (forward_state, backward_state, description).
    """
    
    @wraps(func)
    def wrapper(self, *args, _disable_tracking=False, **kwargs):
        # Get the ITM instance (handle both ITM and InteractiveTopic methods)
        if isinstance(self, InteractiveTopicModel):
            itm = self
        elif hasattr(self, "itm"):
            itm = self.itm
        else:
            # No ITM to track, just execute
            return func(self, *args, **kwargs)
        
        # Skip tracking if disabled
        if not itm._tracking_enabled or _disable_tracking:
            return func(self, *args, **kwargs)
        
        # Execute function to get state changes
        result = func(self, *args, **kwargs)
        
        # If function returns state changes, create edit
        if isinstance(result, tuple) and len(result) >= 3:
            r_len = len(result)
            forward_state, backward_state, description = result[:3]
            
            edit = Edit(
                operation=func.__name__,
                timestamp=datetime.now(),
                forward_state=forward_state,
                backward_state=backward_state,
                description=description,
            )
            
            # Add to undo stack and clear redo stack
            itm._undo_stack.append(edit)
            itm._redo_stack.clear()
            
            # inside undoable wrapper, replace this block:
            if len(result) > 3:
                extras = result[3:]
                return extras[0] if len(extras) == 1 else extras
            else:
                return None
        
        return result
    
    return wrapper


@contextmanager
def disable_tracking(itm: "InteractiveTopicModel"):
    """Context manager to temporarily disable undo/redo tracking."""
    old_value = itm._tracking_enabled
    itm._tracking_enabled = False
    try:
        yield
    finally:
        itm._tracking_enabled = old_value


# =====================================================================
# BasicTopicModel: Semantic space steward
# =====================================================================


class BasicTopicModel:
    """
    Minimal BERTopic-style topic modeler.
    
    Represents one semantic space with its own embedder, reducer, and clusterer.
    ITM instantiates one root BTM and may create additional BTMs for hierarchical splits.
    
    Key responsibilities:
    - Owns embedding, reduction, and clustering components
    - Caches embeddings, reduced embeddings, and lexical (c-TF-IDF) matrices
    - Does NOT own assignments (those belong to ITM)
    - Can recompute cached artifacts when document set changes
    """
    
    def __init__(
        self,
        itm: "InteractiveTopicModel",
        embedder: Optional[Union[str, Embedder]] = None,
        reducer: Optional[Reducer] = None,
        clusterer: Optional[Clusterer] = None,
        scorer: Optional[Scorer] = None,
        vectorizer: Optional[CountVectorizer] = None,
        n_representative_docs: int = 3,
        knn_cache_k: int = 150,
        knn_vote_k: int = 50,
        compute_knn_cache: bool = True,
        # Neighbor-vote + distance-penalty defaults
        neighbor_vote_k_min: int = 5,
        neighbor_close_m: int = 5,
        neighbor_penalty_lambda: float = 0.3,
    ):
        """
        Initialize BasicTopicModel.
        
        Args:
            itm: Reference to parent InteractiveTopicModel.
            embedder: Embedding model or model name.
            reducer: Dimensionality reduction model.
            clusterer: Clustering model.
            scorer: Scoring function for document-topic similarity.
            vectorizer: CountVectorizer for lexical features.
            n_representative_docs: Number of docs to use for topic embedding.
        """
        self.itm = itm
        
        # Set up embedder
        if embedder is None:
            self.embedder = SentenceTransformerEmbedder()
        elif isinstance(embedder, str):
            self.embedder = SentenceTransformerEmbedder(model_name=embedder)
        else:
            self.embedder = embedder
        
        # Set up reducer
        if reducer is None:
            self.reducer = UMAPReducer()
        else:
            self.reducer = reducer
        
        # Set up clusterer
        if clusterer is None:
            self.clusterer = HDBSCANClusterer()
        else:
            self.clusterer = clusterer
        
        # Set up scorer
        self.scorer = scorer if scorer is not None else embedding_scorer
        
        # Set up vectorizer
        self.vectorizer = vectorizer if vectorizer is not None else self.itm._default_vectorizer
        
        self.n_representative_docs = n_representative_docs
        
        # Document-level caches (immutable once computed)
        # Maps doc_id to position in embedding matrices
        self._doc_id_to_pos: Dict[int, int] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._reduced_embeddings: Optional[np.ndarray] = None
        
        # Lexical (DTM) caches for this semantic space (only used when this BTM has a custom vectorizer)
        self._dtm: Optional[sparse.csr_matrix] = None  # rows align to _lex_doc_id_to_pos
        self._vocabulary: Optional[List[str]] = None
        # Positional reverse mapping (kept in sync with _doc_id_to_pos)
        self._pos_to_doc_id: List[int] = []

        # Optional cached kNN graph in the ORIGINAL embedding space (cosine similarity).
        # Stores neighbor positions (NOT doc_ids) so it stays stable under label changes.
        self._knn_cache_k: int = int(knn_cache_k)
        self._knn_vote_k: int = int(knn_vote_k)
        self._neighbor_vote_k_min: int = int(neighbor_vote_k_min)
        self._neighbor_close_m: int = int(neighbor_close_m)
        self._neighbor_penalty_lambda: float = float(neighbor_penalty_lambda)
        self._compute_knn_cache: bool = bool(compute_knn_cache)
        self._knn_indices: Optional[np.ndarray] = None  # shape (N, k), int32
        self._knn_sims: Optional[np.ndarray] = None     # shape (N, k), float32

        
    
    def fit(self, doc_ids: List[int]) -> Dict[str, Any]:
        """
        Fit the topic model on specified documents.
        
        Args:
            doc_ids: List of document IDs to fit on.
            
        Returns:
            Dictionary with 'labels' and 'strengths' arrays (positional order).
        """
        texts = [self.itm._texts[self.itm._pos(doc_id)] for doc_id in doc_ids]
        
        # Build doc_id mapping
        self._doc_id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self._pos_to_doc_id = list(doc_ids)
        
        # Generate embeddings
        self._embeddings = self.embedder.encode(texts)

        # Optionally build cached kNN graph in embedding space (cosine similarity)
        if self._compute_knn_cache and self._knn_cache_k > 0:
            self._build_knn_cache()
        
        # Reduce dimensionality
        self._reduced_embeddings = self.reducer.fit_transform(self._embeddings)
        
        # Cluster
        self.clusterer.fit(self._reduced_embeddings)
        labels = self.clusterer.labels_
        strengths = self.clusterer.strengths_
        
        # Store vocabulary if using custom vectorizer
        if self.vectorizer is not self.itm._default_vectorizer:
            # fit_transform builds vectorizer vocabulary scoped to these docs
            self._dtm = self.vectorizer.fit_transform(texts)
            self._vocabulary = self.vectorizer.get_feature_names_out().tolist()
        else:
            # bubble up to ITM lexical space
            self._dtm = None
            self._vocabulary = None
            
        return {"labels": labels, "strengths": strengths}
    
    def add_docs_to_space(self, doc_ids: Iterable[int]) -> None:
        """
        Ensure the given doc_ids have rows in this BTM's local caches (embeddings, reduced, dtm),
        appending missing rows in a single batch so that indices align.

        - If this BTM bubbles up its lexical space (vectorizer is global default), only embeddings
        are appended here (DTM is left to ITM).
        - If this BTM owns a lexical space, new DTM rows are appended with vectorizer.transform(...).
        """
        doc_ids = list(doc_ids)
        if not doc_ids:
            return

        # Reject duplicates (keep behavior consistent with other methods)
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("add_docs_to_space() does not allow duplicate doc_ids")

        # find which docs are missing locally
        missing = [d for d in doc_ids if d not in self._doc_id_to_pos]
        if not missing:
            return

        # compute embeddings for missing docs (texts come from ITM)
        texts = [self.itm._texts[self.itm._pos(doc_id)] for doc_id in missing]
        new_embeddings = self.embedder.encode(texts)  # shape (n_new, D)
        
        # reduce new embeddings
        if not hasattr(self.reducer, "transform"):
            raise RuntimeError(
                "Reducer must implement transform() to support incremental additions. "
                "Either use a reducer that supports transform (e.g., fitted PCA/UMAP with transform), "
                "or avoid incremental add_docs_to_space() calls."
        )
        new_reduced = self.reducer.transform(new_embeddings)

        # append to embeddings caches (initialize if needed)
        if self._embeddings is None:
            self._embeddings = new_embeddings.copy()
            self._reduced_embeddings = new_reduced.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
            self._reduced_embeddings = np.vstack([self._reduced_embeddings, new_reduced])

        # append to lexical DTM if this BTM owns a vectorizer
        if self.vectorizer is not self.itm._default_vectorizer:
            # vectorizer must already be fit (normally done in fit()); use transform for new docs
            new_rows = self.vectorizer.transform(texts)
            if self._dtm is None:
                self._dtm = new_rows
                # if vocabulary wasn't set earlier, set it now
                if self._vocabulary is None:
                    self._vocabulary = self.vectorizer.get_feature_names_out().tolist()
            else:
                self._dtm = sparse.vstack([self._dtm, new_rows], format="csr")
        # If this BTM bubbles up lexical space, do NOT touch ITM._dtm here.

        # update mapping: new rows are appended at end
        base = len(self._doc_id_to_pos)
        for i, doc_id in enumerate(missing):
            self._doc_id_to_pos[doc_id] = base + i
            self._pos_to_doc_id.append(doc_id)

        # Optionally update kNN cache for the newly appended rows (does not retroactively update older rows)
        if self._compute_knn_cache and self._knn_cache_k > 0:
            self._update_knn_for_new_rows(start_pos=base, n_new=len(missing))

    # ----------------------------
    # Cached kNN graph utilities
    # ----------------------------

    def _build_knn_cache(self) -> None:
        """Build a full kNN cache for all docs currently in this semantic space.

        Uses cosine distance in the ORIGINAL embedding space (not reduced space).
        Stores neighbor *positions* (row indices) and cosine similarities.
        """
        if self._embeddings is None or self._embeddings.size == 0:
            self._knn_indices = None
            self._knn_sims = None
            return

        k = int(self._knn_cache_k)
        n = int(self._embeddings.shape[0])
        if k <= 0 or n <= 1:
            self._knn_indices = np.zeros((n, 0), dtype=np.int32)
            self._knn_sims = np.zeros((n, 0), dtype=np.float32)
            return

        # sklearn returns cosine *distances* in [0, 2] (for non-normalized); we store cosine similarities.
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="brute")
        nn.fit(self._embeddings)
        distances, indices = nn.kneighbors(self._embeddings, return_distance=True)

        # drop self-neighbor (distance 0) when present
        if indices.shape[1] > k:
            indices = indices[:, 1:]
            distances = distances[:, 1:]

        sims = (1.0 - distances).astype(np.float32)
        self._knn_indices = indices.astype(np.int32, copy=False)
        self._knn_sims = sims

    def _update_knn_for_new_rows(self, *, start_pos: int, n_new: int) -> None:
        """Append kNN rows for newly added docs.

        This updates ONLY the rows for new docs. Older rows remain unchanged (asymmetric kNN graph),
        which is usually fine when additions are rare.
        """
        if n_new <= 0:
            return
        if self._embeddings is None or self._embeddings.size == 0:
            return

        # If cache doesn't exist yet, just build it from scratch.
        if self._knn_indices is None or self._knn_sims is None:
            self._build_knn_cache()
            return

        k = int(self._knn_cache_k)
        n = int(self._embeddings.shape[0])
        if k <= 0 or n <= 1:
            return

        new_emb = self._embeddings[start_pos:start_pos + n_new]
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="brute")
        nn.fit(self._embeddings)
        distances, indices = nn.kneighbors(new_emb, return_distance=True)

        # remove self neighbor if present (only guaranteed when querying the same rows)
        # We'll drop any neighbor equal to its own position, then take top-k remaining.
        cleaned_idx = []
        cleaned_sim = []
        for row_i in range(indices.shape[0]):
            pos = start_pos + row_i
            idx_row = indices[row_i].tolist()
            dist_row = distances[row_i].tolist()
            pairs = [(j, d) for j, d in zip(idx_row, dist_row) if j != pos]
            pairs = pairs[:k]
            cleaned_idx.append([p[0] for p in pairs] + [0] * max(0, k - len(pairs)))
            cleaned_sim.append([(1.0 - p[1]) for p in pairs] + [0.0] * max(0, k - len(pairs)))

        new_idx = np.asarray(cleaned_idx, dtype=np.int32)
        new_sim = np.asarray(cleaned_sim, dtype=np.float32)

        # Append
        self._knn_indices = np.vstack([self._knn_indices, new_idx])
        self._knn_sims = np.vstack([self._knn_sims, new_sim])

    def get_knn(
        self,
        doc_id: int,
        *,
        k: Optional[int] = None,
        include_self: bool = False,
    ) -> Tuple[List[int], np.ndarray]:
        """Return cached kNN for a doc_id as (neighbor_doc_ids, neighbor_sims)."""
        if doc_id not in self._doc_id_to_pos:
            self.add_docs_to_space([doc_id])

        pos = self._doc_id_to_pos[doc_id]
        if self._knn_indices is None or self._knn_sims is None:
            if self._compute_knn_cache and self._knn_cache_k > 0:
                self._build_knn_cache()
            else:
                return [], np.zeros((0,), dtype=np.float32)

        if self._knn_indices is None or self._knn_sims is None:
            return [], np.zeros((0,), dtype=np.float32)

        row_idx = self._knn_indices[pos]
        row_sim = self._knn_sims[pos]
        kk = int(k) if k is not None else row_idx.shape[0]
        kk = max(0, min(kk, row_idx.shape[0]))

        neigh_pos = row_idx[:kk].tolist()
        neigh_sim = row_sim[:kk].copy()

        neigh_doc_ids = []
        for p in neigh_pos:
            if p < 0 or p >= len(self._pos_to_doc_id):
                continue
            d = self._pos_to_doc_id[p]
            if not include_self and d == doc_id:
                continue
            neigh_doc_ids.append(d)

        # Trim sims to match doc_ids length (in case we dropped invalids)
        neigh_sim = neigh_sim[:len(neigh_doc_ids)]
        return neigh_doc_ids, neigh_sim

    def get_embeddings(self, doc_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        doc_ids = list(doc_ids)
        if not doc_ids:
            return np.zeros((0, 0)), np.zeros((0, 0))

        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("get_embeddings() does not allow duplicate doc_ids")

        # ensure missing docs are added (this will append embeddings and dtm rows in lockstep)
        missing = [d for d in doc_ids if d not in self._doc_id_to_pos]
        if missing:
            self.add_docs_to_space(missing)

        # now build index list and return rows (preserve order)
        positions = [self._doc_id_to_pos[d] for d in doc_ids]
        embeddings = self._embeddings[positions]
        reduced = self._reduced_embeddings[positions]
        return embeddings, reduced
    
    def get_lexical_dtm(self, doc_ids: Iterable[int]) -> sparse.csr_matrix:
        doc_ids = list(doc_ids)
        if not doc_ids:
            width = (self._dtm.shape[1] if self._dtm is not None else (self.itm._dtm.shape[1] if self.itm._dtm is not None else 0))
            return sparse.csr_matrix((0, width))

        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("get_lexical_dtm() does not allow duplicate doc_ids")

        # If lexical space is bubbled to ITM, just return rows from ITM._dtm
        if self.vectorizer is self.itm._default_vectorizer:
            if self.itm._dtm is None:
                raise RuntimeError("ITM master DTM not available; did you call fit()?")
            positions = [self.itm._pos(doc_id) for doc_id in doc_ids]
            return self.itm._dtm[positions]

        # Owned lexical space: ensure doc rows exist (this will call vectorizer.transform for new docs)
        missing = [d for d in doc_ids if d not in self._doc_id_to_pos]
        if missing:
            # ensure embeddings and dtm rows are appended in same order
            self.add_docs_to_space(missing)

        positions = [self._doc_id_to_pos[d] for d in doc_ids]
        return self._dtm[positions]

    
    def compute_topic_embedding(self, doc_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute topic embedding from document IDs.
        
        Args:
            doc_ids: Document IDs belonging to this topic.
            
        Returns:
            Topic embedding vector (mean of k most central docs) and indices of those docs.
        """
        if not doc_ids:
            return (np.zeros(self._embeddings.shape[1]) if self._embeddings is not None else np.zeros(0)), np.array([])
        
        embeddings, _ = self.get_embeddings(doc_ids)
        centroid_embedding, top_k = self._compute_centroid_embedding(embeddings)
        return centroid_embedding, [doc_ids[i] for i in top_k]
    
    def _compute_centroid_embedding(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute topic embedding as mean of k most central document embeddings.
        
        Args:
            embeddings: Array of document embeddings (n_docs, embedding_dim).
            
        Returns:
            Topic embedding vector and indices of those documents.
        """
        if len(embeddings) == 0:
            return np.zeros(embeddings.shape[1]), np.array([])

        if len(embeddings) <= self.n_representative_docs:
            return embeddings.mean(axis=0), np.arange(len(embeddings))
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find k most central documents (highest average similarity)
        centrality = similarities.mean(axis=1)
        top_k_indices = np.argsort(centrality)[-self.n_representative_docs :]
        
        return embeddings[top_k_indices].mean(axis=0), top_k_indices
    
    def get_vocabulary(self) -> List[str]:
        """Get vocabulary (from custom vectorizer or ITM's default)."""
        if self._vocabulary is not None:
            return self._vocabulary
        return self.itm._vocabulary


# =====================================================================
# InteractiveTopic: Stable topic node
# =====================================================================


class InteractiveTopic:
    """
    Represents a stable topic node with property-based access to ITM data.
    
    Topics are stored in a flat dictionary with hierarchy via parent pointers.
    All document membership is tracked in ITM.assignments, not here.
    """
    
    def __init__(
        self,
        itm: "InteractiveTopicModel",
        topic_id: int,
        label: Optional[str] = None,
        parent: Optional[Union["InteractiveTopic", "InteractiveTopicModel"]] = None,
    ):
        """
        Initialize topic node.
        
        Args:
            itm: Reference to parent InteractiveTopicModel.
            topic_id: Unique stable topic ID.
            label: Optional human-readable label.
            parent: Parent topic or ITM (for hierarchy).
        """
        self.itm = itm
        self.topic_id = topic_id
        self._label = label
        self.parent = parent if parent is not None else itm
        self.active = True

        # Cached within-topic closeness stats for neighbor-distance penalty.
        # Tuple: (m_close, mu, sigma) where mu is median of per-doc local mean similarities
        # and sigma is a robust spread estimate (scaled MAD). Invalidated when memberships change.
        self._closeness_stats: Optional[Tuple[int, float, float]] = None

        # Hierarchical split support
        self.btm: Optional[BasicTopicModel] = None  # Set if this topic has its own semantic space
        self._split_preview: Optional[SplitPreview] = None
        
        # Cached representations (invalidated when assignments change)
        self._cached_embedding: Optional[np.ndarray] = None
        self._cached_ctfidf: Optional[np.ndarray] = None
        self._cached_top_terms: Optional[List[Tuple[str, float]]] = None
        self._cached_representative_doc_ids: Optional[List[int]] = None
    
    def __repr__(self) -> str:
        status = "active" if self.active else "inactive"
        label_str = f"'{self.label}'" if self._label else "unlabeled"
        count = self.get_count()
        return f"InteractiveTopic(id={self.topic_id}, {label_str}, {status}, n={count})"
    
    # ----------------------------
    # Properties
    # ----------------------------
    
    @property
    def label(self) -> str:
        """Get topic label (auto-generated if not set)."""
        if self._label:
            return self._label
        return f"Topic {self.topic_id}"
    
    @label.setter
    def label(self, value: str) -> None:
        """Set topic label (undoable)."""
        self._set_label(value)
    
    @undoable
    def _set_label(self, value: str) -> Tuple[Dict, Dict, str]:
        """Internal label setter with undo support."""
        if value == self._label:
            return

        old_label = self._label
        self._label = value
        
        forward = {"topic_labels": {self.topic_id: value}}
        backward = {"topic_labels": {self.topic_id: old_label}}
        description = f"Set label for topic {self.topic_id} to '{value}'"
        
        return forward, backward, description
    
    @property
    def semantic_space(self) -> BasicTopicModel:
        """
        Semantic space is always defined by the parent. If parent lacks BTM,
        traverse up until finding a BTM.
        """
        target = self.parent
        while target.btm is None:
            target = target.parent

        return target.btm
    
    # ----------------------------
    # Document access
    # ----------------------------
    
    def _get_doc_mask(self, include_descendants: bool = True) -> np.ndarray:
        """
        Get boolean mask for documents belonging to this topic.
        
        Args:
            include_descendants: Whether to include documents from child topics.
            
        Returns:
            Boolean array of shape (n_docs,).
        """
        mask = self.itm._assignments == self.topic_id
        
        if include_descendants:
            # Find all descendant topics
            descendants = self._get_descendants()
            for desc_id in descendants:
                mask |= self.itm._assignments == desc_id
        
        return mask
    
    def _get_descendants(self) -> List[int]:
        """Get all descendant topic IDs."""
        descendants = []
        for tid, topic in self.itm.topics.items():
            if self._is_ancestor_of(topic):
                descendants.append(tid)
        return descendants
    
    def _is_ancestor_of(self, topic: "InteractiveTopic") -> bool:
        """Check if this topic is an ancestor of another topic."""
        current = topic.parent
        while current is not None and isinstance(current, InteractiveTopic):
            if current.topic_id == self.topic_id:
                return True
            current = current.parent
        return False
    
    def get_count(self, include_descendants: bool = True) -> int:
        """Get number of documents in this topic."""
        return int(self._get_doc_mask(include_descendants).sum())
    
    def get_doc_ids(self, include_descendants: bool = True) -> List[int]:
        """Get document IDs belonging to this topic."""
        mask = self._get_doc_mask(include_descendants)
        return self.itm._doc_ids[mask].tolist()
    
    def get_texts(self, include_descendants: bool = True) -> List[str]:
        """Get document texts belonging to this topic."""
        mask = self._get_doc_mask(include_descendants)
        return [self.itm._texts[i] for i in np.where(mask)[0]]

    def get_examples(
        self,
        n: int = 10,
        include_descendants: bool = True,
        remove_duplicates: bool = True,
        random_state: Optional[int] = None,
    ) -> List[str]:
        """
        Get example documents from this topic.
        
        Args:
            n: Number of examples to return.
            include_descendants: Whether to include descendant topics.
            remove_duplicates: Whether to remove duplicate texts.
            random_state: Random seed for sampling.
            
        Returns:
            List of example document texts.
        """
        texts = self.get_texts(include_descendants=include_descendants)
        
        if remove_duplicates:
            texts = list(set(texts))
        
        if len(texts) <= n:
            return texts
        
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(texts), size=n, replace=False)
        return [texts[i] for i in indices]
    
    def get_representative_doc_ids(self) -> List[int]:
        """
        Get representative document IDs for this topic.
        
        Representative docs are selected based on their strength/centrality
        and cached along with other topic representations.
        
        Returns:
            List of representative document IDs.
        """
        if self._cached_representative_doc_ids is None:
            self._compute_representations()
        return self._cached_representative_doc_ids.copy()
    
    def get_representative_docs(self) -> List[str]:
        """
        Get representative documents for this topic.
        
        Returns:
            List of representative document texts.
        """
        doc_ids = self.get_representative_doc_ids()
        return [self.itm._texts[self.itm._pos(doc_id)] for doc_id in doc_ids]
    
    # ----------------------------
    # Top terms
    # ----------------------------
    
    def get_embedding(self) -> np.ndarray:
        """
        Get topic embedding, computing if needed.
        
        Returns:
            Topic embedding vector.
        """
        if self._cached_embedding is None:
            self._compute_representations()
        return self._cached_embedding
    
    def get_ctfidf(self) -> np.ndarray:
        """
        Get topic c-TF-IDF vector, computing if needed.
        
        Returns:
            Topic c-TF-IDF vector.
        """
        if self._cached_ctfidf is None:
            self._compute_representations()
        return self._cached_ctfidf
    
    def get_top_terms(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top terms for this topic.
        
        Args:
            n: Number of terms to return.
            
        Returns:
            List of (term, score) tuples.
        """
        if self._cached_top_terms is None:
            self._compute_representations()
        return self._cached_top_terms[:n]
    
    def get_auto_label(self, n: int = 4) -> str:
        """
        Generate automatic label from top terms.
        
        Args:
            n: Number of terms to use.
            
        Returns:
            Label string.
        """
        if self.topic_id == InteractiveTopicModel.OUTLIER_ID:
            return "OUTLIERS"
        
        top_terms = self.get_top_terms(n=n)
        if not top_terms:
            return f"Topic {self.topic_id}"
        terms = [term for term, _ in top_terms[:n]]
        return f"{self.topic_id}_{'_'.join(terms)}"
    
    def get_doc_diagnostics(
        self,
        *,
        include_descendants: bool = True,
        max_pairwise_docs: int = 3000,
        sample_k: int = 400,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Return per-document diagnostics for this topic.

        Metrics:
        - sim_to_center: cosine similarity between each doc embedding and the topic embedding
        - mean_sim_to_others: mean cosine similarity to other docs in the topic
        - median_sim_to_others: median cosine similarity to other docs in the topic

        Notes:
        - If n_docs <= max_pairwise_docs: mean/median are computed exactly via full pairwise matrix.
        - If n_docs > max_pairwise_docs: mean/median are approximated by sampling up to sample_k
            other docs per doc (to avoid O(n^2) memory/time).

        Args:
            include_descendants: include docs from descendant topics.
            max_pairwise_docs: threshold for exact O(n^2) computation.
            sample_k: sample size per doc for approximate mean/median when topic is large.
            random_state: RNG seed for sampling stability.

        Returns:
            pd.DataFrame with one row per doc.
        """
        self.itm._require_fitted()

        doc_ids = [int(d) for d in self.get_doc_ids(include_descendants=include_descendants)]
        if not doc_ids:
            return pd.DataFrame(
                columns=[
                    "doc_id",
                    "topic_id",
                    "label",
                    "sim_to_center",
                    "mean_sim_to_others",
                    "median_sim_to_others",
                    "text",
                ]
            )

        # Embeddings for docs in this topic
        btm = self.semantic_space
        doc_embeddings, _ = btm.get_embeddings(doc_ids)  # (n, dim)

        # Topic embedding (cached)
        topic_emb = self.get_embedding().reshape(1, -1)

        # Similarity to center
        sim_to_center = cosine_similarity(doc_embeddings, topic_emb).ravel()

        n = len(doc_ids)
        mean_sim = np.full(n, np.nan, dtype=float)
        median_sim = np.full(n, np.nan, dtype=float)

        if n == 1:
            # no "others" to compare to
            pass
        elif n <= max_pairwise_docs:
            S = cosine_similarity(doc_embeddings)  # (n, n)
            np.fill_diagonal(S, np.nan)
            mean_sim = np.nanmean(S, axis=1)
            median_sim = np.nanmedian(S, axis=1)
        else:
            # Approximate by sampling to avoid O(n^2)
            rng = np.random.default_rng(random_state)

            # Normalize embeddings once so cosine(u,v) == dot(u,v)
            norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            E = doc_embeddings / safe_norms

            k = min(sample_k, n - 1)
            for i in range(n):
                if k <= 0:
                    continue

                # sample other indices excluding i
                idx = rng.integers(0, n, size=k + 8)
                idx = idx[idx != i]
                if len(idx) < k:
                    candidates = np.array([j for j in range(n) if j != i], dtype=int)
                    if len(candidates) > k:
                        idx = rng.choice(candidates, size=k, replace=False)
                    else:
                        idx = candidates
                else:
                    idx = idx[:k]

                sims = E[i].dot(E[idx].T)
                mean_sim[i] = float(np.mean(sims))
                median_sim[i] = float(np.median(sims))

        texts = [self.itm._texts[self.itm._pos(d)] for d in doc_ids]

        df = pd.DataFrame(
            {
                "doc_id": doc_ids,
                "topic_id": int(self.topic_id),
                "label": self.label,
                "sim_to_center": sim_to_center,
                "mean_sim_to_others": mean_sim,
                "median_sim_to_others": median_sim,
                "text": texts,
            }
        )

        # Put the most suspicious docs at the top
        df = df.sort_values(["sim_to_center", "mean_sim_to_others"], ascending=[True, True]).reset_index(drop=True)
        return df

    def _compute_representations(self) -> None:
        """
        Compute and cache all topic representations.
        Uses ClassTfidfTransformer at parent level for proper c-TF-IDF calculation.
        """
        doc_ids = self.get_doc_ids(include_descendants=True)
        btm = self.semantic_space
        
        if not doc_ids:
            # Empty topic
            embedding_dim = btm._embeddings.shape[1] if btm._embeddings is not None else 0
            vocab_size = len(btm.get_vocabulary())
            self._cached_embedding = np.zeros(embedding_dim)
            self._cached_ctfidf = np.zeros(vocab_size)
            self._cached_top_terms = []
            self._cached_representative_doc_ids = []
            return
        
        # Compute embedding
        self._cached_embedding, self._cached_representative_doc_ids = btm.compute_topic_embedding(doc_ids)
        
        # Compute c-TF-IDF at parent level using ClassTfidfTransformer
        # This allows c-TF-IDF to be computed relative to siblings, not just this topic
        parent = self.parent
        btm = self.semantic_space
        
        # Determine which topics to include in c-TF-IDF calculation
        sibling_topics = [
            t for t in self.itm.topics.values()
            if (t.parent == parent and t.topic_id >= 0 and t.semantic_space is btm)]
        
        # Collect all doc positions and labels for siblings
        all_doc_ids: List[int] = []
        all_labels: List[int] = []

        for topic in sibling_topics:
            sib_doc_ids = topic.get_doc_ids(include_descendants=True)
            all_doc_ids.extend(int(d) for d in sib_doc_ids)
            all_labels.extend([int(topic.topic_id)] * len(sib_doc_ids))

        if all_doc_ids:
            # Use the BTM lexical space to get the DTM (bubbles up to ITM._dtm if needed)
            parent_dtm = btm.get_lexical_dtm(all_doc_ids)
            labels_array = np.array(all_labels, dtype=np.int32)

            ctfidf_transformer = ClassTfidfTransformer(reduce_frequent_words=False)
            ctfidf_matrix, label_to_index = ctfidf_transformer.fit_transform(parent_dtm, labels_array)

            if self.topic_id in label_to_index:
                topic_idx = label_to_index[self.topic_id]
                self._cached_ctfidf = ctfidf_matrix[topic_idx].toarray().ravel()
            else:
                # Topic absent (should be rare) -> zero vector with parent's lexical width
                self._cached_ctfidf = np.zeros(parent_dtm.shape[1], dtype=float)
        else:
            # No sibling docs: produce zero-length vector sized to this BTM's vocabulary
            vocab_len = len(btm.get_vocabulary())
            self._cached_ctfidf = np.zeros(vocab_len, dtype=float)
        
        # Get top terms
        vocabulary = btm.get_vocabulary()
        top_indices = np.argsort(self._cached_ctfidf)[::-1][:20]  # Cache top 20
        top_terms = [
            (vocabulary[idx], self._cached_ctfidf[idx])
            for idx in top_indices
            if self._cached_ctfidf[idx] > 0
        ]
        self._cached_top_terms = top_terms
    
    def invalidate_representations(self) -> None:
        """Clear cached representations (called when assignments change)."""
        self._cached_embedding = None
        self._cached_ctfidf = None
        self._cached_top_terms = None
        self._cached_representative_doc_ids = None
        self._closeness_stats = None

    def _get_closeness_stats(self, *, m_close: int) -> Tuple[float, float]:
        """
        Return (mu, sigma) for within-topic local closeness, used for the distance penalty.

        - For each doc in this topic, compute the mean cosine similarity to its top `m_close`
          nearest neighbors *that are also in this topic* (using the semantic space's cached kNN).
        - mu is the median of those per-doc means.
        - sigma is a robust spread estimate using scaled MAD: 1.4826 * median(|x - mu|).

        Cached as a single triple (m_close, mu, sigma) and invalidated when assignments change.
        """
        m_close = int(m_close)
        if self._closeness_stats is not None:
            cached_m, mu, sigma = self._closeness_stats
            if int(cached_m) == m_close:
                return float(mu), float(sigma)

        btm = self.semantic_space
        doc_ids = [int(d) for d in self.get_doc_ids(include_descendants=True)]
        if len(doc_ids) <= 1:
            mu, sigma = (0.0, 1e-6)
            self._closeness_stats = (m_close, float(mu), float(sigma))
            return float(mu), float(sigma)

        doc_set = set(doc_ids)
        # Scan enough neighbors to find `m_close` in-topic neighbors even when a topic is sparse.
        cache_k = int(getattr(btm, "_knn_cache_k", max(50, m_close * 10)))
        scan_k = min(cache_k, max(50, m_close * 10))

        vals: List[float] = []
        for d in doc_ids:
            neigh_ids, neigh_sims = btm.get_knn(d, k=scan_k, include_self=False)
            sims: List[float] = []
            for nid, sim in zip(neigh_ids, neigh_sims):
                if nid is None:
                    continue
                nid = int(nid)
                if nid not in doc_set:
                    continue
                w = float(sim)
                if w <= 0.0:
                    continue
                sims.append(w)
                if len(sims) >= m_close:
                    break
            if sims:
                vals.append(float(np.mean(sims)))

        if not vals:
            mu, sigma = (0.0, 1e-6)
        else:
            arr = np.asarray(vals, dtype=float)
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(arr - mu)))
            sigma = float(1.4826 * mad)
            if sigma <= 1e-12:
                sigma = 1e-6

        self._closeness_stats = (m_close, float(mu), float(sigma))
        return float(mu), float(sigma)

    
    @undoable
    def auto_label(self, n: int = 4) -> Tuple[Dict, Dict, str]:
        """
        Set automatic label from top terms.
        
        Args:
            n: Number of terms to use in label.
        """
        old_label = self._label
        new_label = self.get_auto_label(n=n)
        self._label = new_label
        
        forward = {"topic_labels": {self.topic_id: new_label}}
        backward = {"topic_labels": {self.topic_id: old_label}}
        description = f"Auto-labeled topic {self.topic_id}"
        
        return forward, backward, description
    
    # ----------------------------
    # Split operations
    # ----------------------------
    
    def preview_split(
        self,
        *,
        min_topic_size: int = 2,
        embedder: Optional[Union[str, Embedder]] = None,
        reducer: Optional[Reducer] = None,
        clusterer: Optional[Clusterer] = None,
        vectorizer: Optional[CountVectorizer] = None,
        knn_cache_k: Optional[int] = None,
        knn_vote_k: Optional[int] = None,
        compute_knn_cache: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Preview a split of this topic.
        
        Args:
            embedder: If provided, build a new BTM (new semantic space) for this split preview.
            reducer: If embedder is None (flat split) and reducer is provided, re-reduce full embeddings.
                     If embedder is provided, reducer is passed into the new BTM (default if None).
            clusterer: Clustering model to use (default created if None).
            min_topic_size: Minimum size for new topics.
            keep_outliers: Whether to keep outliers in parent topic.
            
        Returns:
            DataFrame preview of proposed split, or None if no preview exists.
        """
        if len(self._get_descendants()) > 0:
            print("Cannot split a topic that has already been split.")
            return None

        if clusterer is None:
            clusterer = HDBSCANClusterer(min_cluster_size=min_topic_size)
        
        doc_ids = self.get_doc_ids()
        
        if len(doc_ids) < min_topic_size * 2:
            print(f"Topic too small to split (need at least {min_topic_size * 2} docs)")
            return None
        
        split_btm: Optional[BasicTopicModel] = None

        if embedder is not None:
            # New semantic space for this (would-be) split parent topic.
            # Inherit kNN defaults from the ITM unless explicitly overridden.
            _knn_cache_k = self.itm.knn_cache_k if knn_cache_k is None else int(knn_cache_k)
            _knn_vote_k = self.itm.knn_vote_k if knn_vote_k is None else int(knn_vote_k)
            _compute_knn = self.itm.compute_knn_cache if compute_knn_cache is None else bool(compute_knn_cache)

            split_btm = BasicTopicModel(
                itm=self.itm,
                embedder=embedder,
                reducer=reducer,          # default reducer inside BTM if None
                clusterer=clusterer,      # use caller/default clusterer
                scorer=self.itm.btm.scorer,
                vectorizer=vectorizer,
                n_representative_docs=self.itm.btm.n_representative_docs,
                knn_cache_k=_knn_cache_k,
                knn_vote_k=_knn_vote_k,
                compute_knn_cache=_compute_knn,
            )
            fit_out = split_btm.fit(doc_ids)
            labels = fit_out["labels"]
            strengths = fit_out["strengths"]
        else:
            # Flat split: use existing semantic space (parent-owned) and reduced embeddings
            btm = self.semantic_space
            embeddings, reduced = btm.get_embeddings(doc_ids)

            if reducer is None:
                X = reduced  # use cached reduced embeddings
            else:
                X = reducer.fit_transform(embeddings)  # re-reduce full embeddings with provided reducer

            clusterer.fit(X)
            labels = clusterer.labels_
            strengths = clusterer.strengths_
        
        # Generate new topic IDs
        unique_labels = sorted(set(labels))
        label_to_new_id = {}
        new_topic_ids = []
        placeholder_prefix = f"preview-{int(self.topic_id)}"
        
        for label in unique_labels:
            placeholder = f"{placeholder_prefix}-{label}"
            label_to_new_id[int(label)] = placeholder
            new_topic_ids.append(placeholder)
        
        if len(new_topic_ids) == 0:
            print("Clustering produced no valid clusters. No split proposed.")
            self._split_preview = None
            return None

        # Build proposed assignments
        proposed_assignments = {}
        proposed_strengths = {}
        
        for doc_id, label, strength in zip(doc_ids, labels, strengths):
            new_topic_id = label_to_new_id[label]
            proposed_assignments[doc_id] = new_topic_id
            proposed_strengths[doc_id] = float(strength)
        
        # Create and store preview
        self._split_preview = SplitPreview(
            parent_topic_id=self.topic_id,
            proposed_assignments=proposed_assignments,
            proposed_strengths=proposed_strengths,
            new_topic_ids=new_topic_ids,
            btm=split_btm,
        )
        
        # Convert to DataFrame for user inspection
        rows = []
        for doc_id, new_topic_id in proposed_assignments.items():
            text = self.itm._texts[self.itm._pos(doc_id)]
            strength = proposed_strengths[doc_id]
            rows.append(
                {
                    "doc_id": doc_id,
                    "new_topic_id": new_topic_id,
                    "strength": strength,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                }
            )
        
        df = pd.DataFrame(rows)
        return df.sort_values(["new_topic_id", "strength"], ascending=[True, False])
    
    def clear_split_preview(self) -> None:
        """Clear the split preview."""
        self._split_preview = None
    
    @undoable
    def commit_split(
        self, 
        new_topic_labels: Dict[str, str]=None,
        delete_parent: bool = False
    ) -> Tuple[Dict, Dict, str]:
        """
        Commit the previewed split.

        Args:
            new_topic_labels: Optional mapping from preview topic IDs to labels.    
            delete_parent: If True, *remove* the original parent topic from itm.topics and
                        reparent the newly-created topics to this topic's parent.
                        Not allowed if the preview created a new semantic space (preview.btm).
        
        Returns:
            Tuple of (forward_state, backward_state, description) for undo/redo.
        """
        if self._split_preview is None:
            raise ITMError("No split preview to commit. Call preview_split() first.")
        
        preview = self._split_preview
        new_topic_labels = new_topic_labels or {}

        old_btm = self.btm
        new_btm = getattr(preview, "btm", None)
    
        # If the preview created a new semantic space, we cannot delete the parent.
        if delete_parent and new_btm is not None:
            raise ITMError("Cannot delete parent when the preview created a new semantic space.")
        
        # Verify no documents have changed assignment since preview
        for doc_id in preview.proposed_assignments.keys():
            pos = self.itm._pos(doc_id)
            current_assignment = self.itm._assignments[pos]
            if current_assignment != self.topic_id:
                raise ITMError(
                    f"Document {doc_id} has changed assignment since preview "
                    f"(was {self.topic_id}, now {current_assignment}). "
                    "Please regenerate the preview."
                )
       
        # Create new topics
        placeholder_to_real: Dict[object, int] = {}
        real_new_topic_ids: list[int] = []

        parent_for_new_topics = self.parent if (delete_parent or (self.topic_id == self.itm.OUTLIER_ID)) else self

        for pid in preview.new_topic_ids:
            # If it's already an int (e.g., parent topic id or OUTLIER_ID), skip
            if isinstance(pid, int):
                continue
            # Detect our placeholder pattern (string starting with "preview-")
            if isinstance(pid, str) and pid.startswith("preview-"):
                new_real = int(self.itm._new_topic_id())   # allocate real ID now
                placeholder_to_real[pid] = new_real
                real_new_topic_ids.append(new_real)
                # ensure topic record exists (parent=self)
                self.itm._ensure_topic(
                    new_real, 
                    parent=parent_for_new_topics, 
                    label=new_topic_labels.get(str(pid), None))
            else:
                # unknown type (conservative approach: treat as error)
                raise ITMError(f"Unknown preview topic id format: {pid!r}")
        
        # Build final assignments mapping (doc_id -> real_topic_id)
        final_assignments = {}
        for doc_id, assigned in preview.proposed_assignments.items():
            if isinstance(assigned, int):
                # parent/outlier mapping preserved as-is
                final_assignments[int(doc_id)] = int(assigned)
            else:
                # placeholder -> real mapping
                if assigned not in placeholder_to_real:
                    # It could be that the assigned value was parent's id stringified, but we handled ints above
                    raise ITMError(f"No mapping for preview topic id {assigned!r}")
                final_assignments[int(doc_id)] = int(placeholder_to_real[assigned])

        # Save old state for undo
        old_assignments = {}
        old_strengths = {}
        for doc_id in preview.proposed_assignments.keys():
            pos = self.itm._pos(doc_id)
            old_assignments[doc_id] = int(self.itm._assignments[pos])
            old_strengths[doc_id] = float(self.itm._strengths[pos])
        
        # Apply new assignments
        for doc_id, new_topic_id in final_assignments.items():
            pos = self.itm._pos(doc_id)
            self.itm._assignments[pos] = new_topic_id
            self.itm._strengths[pos] = preview.proposed_strengths[doc_id]
        
        # Attach new semantic space (BTM) to the topic being split (parent for new children)
        if new_btm is not None:
            self.btm = new_btm

        # Invalidate representations for affected topics
        affected_topics = [self.topic_id] + real_new_topic_ids
        for topic_id in affected_topics:
            if topic_id in self.itm.topics:
                self.itm.topics[topic_id].invalidate_representations()
        
        # Clear preview
        self._split_preview = None

        # Prepare undo/redo state
        forward = {
            "assignments": final_assignments.copy(),
            "strengths": preview.proposed_strengths.copy(),
            "new_topics": real_new_topic_ids.copy(),
        }

        backward = {
            "assignments": old_assignments,
            "strengths": old_strengths,
            "remove_topics": real_new_topic_ids.copy(),
        }
        if delete_parent:
            # Gather children *before* we remove the topic so we can restore links later.
            children_ids = [t.topic_id for t in self.itm.topics.values() if getattr(t, "parent", None) is self]

            # Make a deep copy of the topic object so future mutations won't affect the undo snapshot
            topic_snapshot = copy.deepcopy(self)

            # Remove the topic object from itm.topics (this is the deletion)
            removed_obj = self.itm.topics.pop(self.topic_id, None)

            # Save full snapshot + relational metadata for undo
            backward.setdefault("restore_topics", {})[self.topic_id] = {
                "topic_snapshot": topic_snapshot,
                "parent_id": getattr(self.parent, "topic_id", None),
                "child_ids": children_ids,
            }
            forward.setdefault("deleted_topics", []).append(self.topic_id)
        else:
            # Deactivate old topic if all docs moved out (preserve previous logic)
            if self.get_count(include_descendants=False) == 0:
                self.active = False
                forward.setdefault("deactivate_topics", []).append(self.topic_id)
                backward.setdefault("reactivate_topics", []).append(self.topic_id)

                if self.topic_id != self.itm.OUTLIER_ID:
                    old_label = self._label
                    self._label = "[SPLIT] " + old_label
                    forward.setdefault("topic_labels", {})[self.topic_id] = self._label
                    backward.setdefault("topic_labels", {})[self.topic_id] = old_label

        if new_btm is not None or old_btm is not None:
            forward["topic_btms"] = {self.topic_id: new_btm}
            backward["topic_btms"] = {self.topic_id: old_btm}
        
        description = f"Split topic {self.topic_id} into {len(real_new_topic_ids)} new topics"
        
        return forward, backward, description

    @undoable
    def junk(self) -> Tuple[Dict, Dict, str]:
        """
        Mark this topic as junk by moving all documents to outliers,
        relabeling it as [JUNK], and deactivating it.
        """
        # Get all document IDs from this topic (not including descendants)
        doc_ids = self.get_doc_ids(include_descendants=False)
        
        if not doc_ids:
            raise ITMError(f"Topic {self.topic_id} has no documents to junk")
        
        # Get mask and positions for vectorized operations
        mask = self._get_doc_mask(include_descendants=False)
        positions = np.where(mask)[0]
        
        # Store old values for undo/redo tracking
        old_assignments_array = self.itm._assignments[positions].copy()
        old_strengths_array = self.itm._strengths[positions].copy()
        
        # Vectorized assignment to outliers
        self.itm._assignments[positions] = self.itm.OUTLIER_ID
        self.itm._strengths[positions] = 0.0
        
        # Build dictionaries for state tracking
        old_assignments = dict(zip(doc_ids, old_assignments_array))
        old_strengths = dict(zip(doc_ids, old_strengths_array))
        new_assignments = {doc_id: self.itm.OUTLIER_ID for doc_id in doc_ids}
        new_strengths = {doc_id: 0.0 for doc_id in doc_ids}
        
        # Update label
        old_label = self._label
        new_label = "[JUNK] " + (old_label if old_label else f"Topic {self.topic_id}")
        self._label = new_label
        
        # Deactivate topic
        old_active = self.active
        self.active = False
        
        # Invalidate representations for this topic and outliers
        self.invalidate_representations()
        if self.itm.OUTLIER_ID in self.itm.topics:
            self.itm.topics[self.itm.OUTLIER_ID].invalidate_representations()
        
        # Build forward and backward states
        forward = {
            "assignments": new_assignments,
            "strengths": new_strengths,
            "topic_labels": {self.topic_id: new_label},
            "deactivate_topics": [self.topic_id],
        }
        
        backward = {
            "assignments": old_assignments,
            "strengths": old_strengths,
            "topic_labels": {self.topic_id: old_label},
        }
        
        if old_active:
            backward["reactivate_topics"] = [self.topic_id]
        
        description = f"Junked topic {self.topic_id}, moved {len(doc_ids)} documents to outliers"
        
        return forward, backward, description

    def visualize_documents(
        self,
        *,
        use_reduced: bool = True,
        title: Optional[str] = None,
        reducer: Optional[Reducer] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize documents in 2D space.
        
        Args:
            title: Optional title for the plot.
            reducer: Optional 2D reducer (if None, uses UMAP with n_components=2).
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        from .visualization import visualize_documents as _visualize_documents
        return _visualize_documents(
            self.itm, 
            plot_btm=self.semantic_space, 
            doc_ids=self.get_doc_ids(), 
            use_reduced=use_reduced,
            title=title or f"Document Scatter Plot for Topic {self.topic_id} ({self.label})", 
            reducer=reducer, 
            save_path=save_path
        )
    
    def visualize_documents_with_search(
        self,
        search_string: str,
        *,
        use_reduced: bool = True,
        title: Optional[str] = None,
        reducer: Optional[Reducer] = None,
        search_embedder: Optional[Union[str, Embedder]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize documents with search highlighting.
        
        Args:
            search_string: Text to search for similarity.
            title: Optional title for the plot.
            reducer: Optional 2D reducer (if None, uses UMAP with n_components=2).
            search_embedder: Embedder instance or model name for search.
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        from .visualization import visualize_documents_with_search as _visualize_documents_with_search
        return _visualize_documents_with_search(
            self.itm,
            plot_btm=self.semantic_space,
            doc_ids=self.get_doc_ids(),
            search_string=search_string,
            use_reduced=use_reduced,
            title=title,
            reducer=reducer,
            search_embedder=search_embedder,
            save_path=save_path,
        )
    
    def visualize_split_preview(
        self,
        *,
        use_reduced: bool = True,
        title: Optional[str] = None,
        reducer: Optional[Reducer] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the split preview in 2D space.
        
        Args:
            title: Optional title for the plot.
            reducer: Optional 2D reducer (if None, uses UMAP with n_components=2).
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        if self._split_preview is None:
            raise ITMError("No split preview to commit. Call preview_split() first.")
        
        preview = self._split_preview
        
        from .visualization import visualize_documents as _visualize_documents
        return _visualize_documents(
            self.itm, 
            plot_btm=getattr(preview, "btm", None) or self.semantic_space, 
            doc_ids=list(preview.proposed_assignments.keys()), 
            custom_topic_labels=[str(label) for label in preview.proposed_assignments.values()],
            use_reduced=use_reduced,
            title=title or f"Document Scatter Plot for Topic {self.topic_id} ({self.label}) Proposed Split", 
            reducer=reducer, 
            save_path=save_path
        )

# =====================================================================
# InteractiveTopicModel: Main engine
# =====================================================================


class InteractiveTopicModel:
    """
    Interactive topic model with undo/redo, hierarchical topics, and preview-commit workflow.
    
    Core principles:
    - doc_id is the stable identity (never changes)
    - self.assignments is the single source of truth for membership
    - Topics stored in flat dictionary with parent pointers for hierarchy
    - BasicTopicModel represents semantic spaces
    - Hierarchy exists only when embedding spaces differ
    """
    
    topic_cls: Type[InteractiveTopic] = InteractiveTopic
    OUTLIER_ID = -1
    
    def __init__(
        self,
        texts: Sequence[str],
        *,
        vectorizer: Optional[CountVectorizer] = None,
        embedder: Optional[Union[str, Embedder]] = None,
        reducer: Optional[Reducer] = None,
        clusterer: Optional[Clusterer] = None,
        scorer: Optional[Scorer] = None,
        n_representative_docs: int = 3,
        # kNN cache / voting defaults (used by BTMs created by this ITM)
        knn_cache_k: int = 150,
        knn_vote_k: int = 50,
        compute_knn_cache: bool = True,
    ):
        """
        Initialize Interactive Topic Model.
        
        Args:
            texts: Sequence of document texts.
            vectorizer: Default vectorizer for c-TF-IDF.
            embedder: Embedding model.
            reducer: Dimensionality reduction model.
            clusterer: Clustering model.
            scorer: Scoring function.
            n_representative_docs: Number of docs for topic embedding.
        """
        
        self._texts = texts
        n_docs = len(self._texts)
        self._doc_ids = np.arange(n_docs)
        
        # Validate doc_id uniqueness
        if len(np.unique(self._doc_ids)) != n_docs:
            raise IdentityError("doc_ids must be unique")
        
        # Create doc_id to position mapping
        self._doc_id_to_pos = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
        
        # Single source of truth for assignments
        self._assignments = np.full(n_docs, -1, dtype=np.int32)  # -1 = unassigned
        self._strengths = np.zeros(n_docs, dtype=np.float32)
        self._validated = np.zeros(n_docs, dtype=bool)
        
        # Topic storage (flat dictionary with parent pointers)
        self.topics: Dict[int, InteractiveTopic] = {}
        self._next_topic_id = 0
        
        self._default_vectorizer = vectorizer or default_vectorizer()

        # Store kNN defaults so child semantic spaces can inherit them
        self.knn_cache_k = int(knn_cache_k)
        self.knn_vote_k = int(knn_vote_k)
        self.compute_knn_cache = bool(compute_knn_cache)

        # Create root BTM (semantic space)
        self.btm = BasicTopicModel(
            itm=self,
            embedder=embedder,
            reducer=reducer,
            clusterer=clusterer,
            scorer=scorer,
            vectorizer=self._default_vectorizer,
            n_representative_docs=n_representative_docs,
            knn_cache_k=self.knn_cache_k,
            knn_vote_k=self.knn_vote_k,
            compute_knn_cache=self.compute_knn_cache,
        )
        
        self._fitted = False
        
        # Master DTM and vocabulary (computed once during fit)
        self._dtm: Optional[sparse.csr_matrix] = None
        self._vocabulary: Optional[List[str]] = None
        self._tfidf_transformer: Optional[TfidfTransformer] = None
        
        # Undo/redo support
        self._undo_stack: List[Edit] = []
        self._redo_stack: List[Edit] = []
        self._tracking_enabled = True
        
        # Create outlier topic
        self._ensure_topic(self.OUTLIER_ID, label="Outliers", parent=self)
        self.topics[self.OUTLIER_ID].active = False
    
    def __repr__(self) -> str:
        n_docs = len(self._doc_ids)
        n_topics = len([t for t in self.topics.values() if t.active and t.topic_id >= 0])
        n_assigned = (self._assignments >= 0).sum()
        status = "fitted" if self._fitted else "not fitted"
        
        return (
            f"InteractiveTopicModel({status}, "
            f"docs={n_docs}, topics={n_topics}, assigned={n_assigned})"
        )
    
    # ----------------------------
    # Helper methods
    # ----------------------------
    
    def _pos(self, doc_id: int) -> int:
        """Get positional index for doc_id."""
        if doc_id not in self._doc_id_to_pos:
            raise IdentityError(f"Unknown doc_id: {doc_id}")
        return self._doc_id_to_pos[doc_id]
    
    def _require_fitted(self) -> None:
        """Raise error if model not fitted."""
        if not self._fitted:
            raise NotFittedError("Model must be fitted before this operation")
    
    def _new_topic_id(self) -> int:
        """Generate new unique topic ID."""
        new_id = self._next_topic_id
        self._next_topic_id += 1
        return new_id
    
    def _ensure_topic(
        self,
        topic_id: int,
        label: Optional[str] = None,
        parent: Optional[Union[InteractiveTopic, "InteractiveTopicModel"]] = None,
    ) -> InteractiveTopic:
        """
        Ensure topic exists, create if needed.
        
        Args:
            topic_id: Topic ID.
            label: Optional label.
            parent: Parent topic or ITM.
            
        Returns:
            InteractiveTopic instance.
        """
        if topic_id not in self.topics:
            self.topics[topic_id] = self.topic_cls(
                itm=self, topic_id=topic_id, label=label, parent=parent
            )
            if topic_id >= self._next_topic_id:
                self._next_topic_id = topic_id + 1
        return self.topics[topic_id]
    
    # ----------------------------
    # Semantic space property (for compatibility with InteractiveTopic)
    # ----------------------------
    
    @property
    def semantic_space(self) -> BasicTopicModel:
        """Return root BTM."""
        return self.btm

    @property
    def outlier_topic(self) -> InteractiveTopic:
        """Return outlier topic."""
        return self.topics[self.OUTLIER_ID]
    
    def topics_in_semantic_space(self, space_btm) -> List["InteractiveTopic"]:
        """All active (non-outlier) topics whose semantic_space is `space_btm`."""
        return [
            t for t in self.topics.values()
            if t.topic_id >= 0 and t.semantic_space is space_btm
        ]

    def children_in_space(self, parent: "InteractiveTopic", space_btm) -> List["InteractiveTopic"]:
        """Children of `parent` that share the given semantic space."""
        return [
            c for c in self.topics.values()
            if (
                c.topic_id >= 0
                and c.parent is parent
                and c.semantic_space is space_btm
            )
        ]

    def frontier_topics_in_space(self, space_btm) -> List["InteractiveTopic"]:
        """
        Topics in this semantic space that have NO children in the same space.
        (Avoids assigning to empty/group container topics.)
        """
        topics = self.topics_in_semantic_space(space_btm)
        frontier = []
        for t in topics:
            if not self.children_in_space(t, space_btm):
                frontier.append(t)
        return frontier
    
    def assignment_candidates(self, space_btm, *, prefer_frontier: bool = True, exclude_empty: bool = True) -> List["InteractiveTopic"]:
        if prefer_frontier:
            topics = self.frontier_topics_in_space(space_btm)
        else:
            topics = self.topics_in_semantic_space(space_btm)
        if exclude_empty:
            topics = [t for t in topics if t.get_count() > 0]
        
        return topics

    # ----------------------------
    # Fit operation
    # ----------------------------
    
    def fit(self) -> None:
        """
        Fit initial topic model on all documents.
        
        Creates initial topics and assigns documents.
        """
        doc_ids = self._doc_ids.tolist()
        
        # Build master DTM (once, never changes)
        if self._default_vectorizer is None:
            self._default_vectorizer = default_vectorizer()
        texts = [self._texts[i] for i in range(len(self._texts))]
        self._dtm = self._default_vectorizer.fit_transform(texts)
        self._vocabulary = self._default_vectorizer.get_feature_names_out().tolist()
        
        # Fit TF-IDF transformer for document scoring
        self._tfidf_transformer = TfidfTransformer()
        self._tfidf_transformer.fit(self._dtm)
        
        # Fit BTM
        result = self.btm.fit(doc_ids)
        self._assignments = result["labels"]
        self._strengths = result["strengths"]
        
        # Create topics and assign documents
        unique_labels = sorted(set(self._assignments) | {self.OUTLIER_ID})
        
        for label in unique_labels:
            if label == -1:
                # Outliers
                topic_id = self.OUTLIER_ID
            else:
                # Regular topic
                topic_id = self._new_topic_id()
            self._ensure_topic(topic_id, parent=self)
            self.topics[topic_id].auto_label(_disable_tracking=True)
        
        self._fitted = True
    
    # ----------------------------
    # Assignment operations
    # ----------------------------
    
    @undoable
    def assign_doc(
        self, doc_id: int, topic_id: int, strength: float = 1.0, validated: bool = True
    ) -> Tuple[Dict, Dict, str]:
        """
        Assign a document to a topic.
        
        Args:
            doc_id: Document ID.
            topic_id: Topic ID.
            strength: Assignment strength.
            validated: Whether to mark as validated.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        self._require_fitted()
        pos = self._pos(doc_id)
        
        # Save old state
        old_assignment = int(self._assignments[pos])
        old_strength = float(self._strengths[pos])
        old_validated = bool(self._validated[pos])
        
        # Ensure topic exists
        self._ensure_topic(topic_id)
        
        # Apply new assignment
        self._assignments[pos] = topic_id
        self._strengths[pos] = strength
        self._validated[pos] = validated
        
        # Invalidate representations for affected topics
        affected_topics = {old_assignment, topic_id}
        for tid in affected_topics:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()
        
        # Prepare undo/redo state
        forward = {
            "assignments": {doc_id: topic_id},
            "strengths": {doc_id: strength},
            "validated": {doc_id: validated},
        }
        
        backward = {
            "assignments": {doc_id: old_assignment},
            "strengths": {doc_id: old_strength},
            "validated": {doc_id: old_validated},
        }
        
        description = f"Assigned doc {doc_id} to topic {topic_id}"
        
        return forward, backward, description
    
    @undoable
    def manual_reassign(
        self,
        reassignments: Dict[Any, int],
        *,
        strength: float = 1.0, 
        validated: bool = True
    ) -> Tuple[Dict, Dict, str, Dict[str, int]]:
        self._require_fitted()

        old_assignments: Dict[int, int] = {}
        old_strengths: Dict[int, float] = {}
        old_validated: Dict[int, bool] = {}

        new_assignments: Dict[int, int] = {}
        new_strengths: Dict[int, float] = {}
        new_validated: Dict[int, bool] = {}

        for doc_id, topic_id in reassignments.items():
            pos = self._pos(doc_id)

            old_assignments[doc_id] = int(self._assignments[pos])
            old_strengths[doc_id] = float(self._strengths[pos])
            old_validated[doc_id] = bool(self._validated[pos])

            self._assignments[pos] = int(topic_id)
            new_assignments[doc_id] = int(topic_id)

            self._strengths[pos] = float(strength)
            new_strengths[doc_id] = float(strength)

            if validated:
                self._validated[pos] = True
                new_validated[doc_id] = True

        affected = set(new_assignments.values()) | set(old_assignments.values())
        for tid in affected:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()

        forward = {"assignments": new_assignments}
        backward = {"assignments": old_assignments}

        forward["strengths"] = new_strengths
        backward["strengths"] = old_strengths

        if validated:
            forward["validated"] = new_validated
            backward["validated"] = old_validated

        description = f"Reassigned {len(new_assignments)} documents"
        summary = {"reassigned": len(new_assignments)}
        return forward, backward, description, summary

    @undoable
    def assign_docs_to_topic(
        self,
        doc_ids: List[int],
        topic_id: int,
        *,
        set_validation: Optional[bool] = None,
    ) -> Tuple[Dict, Dict, str]:
        """
        Assign the given documents to a topic.

        Args:
            doc_ids: iterable of document ids.
            topic_id: target topic id (must exist, or be OUTLIER_ID).
            set_validation:
                - None: do not change validation status
                - True / False: set validation status for these docs

        Returns:
            (forward_state, backward_state, description)
        """
        self._require_fitted()

        topic_id = int(topic_id)
        doc_ids = [int(d) for d in doc_ids]

        if topic_id != self.OUTLIER_ID and topic_id not in self.topics:
            raise KeyError(f"Unknown topic_id: {topic_id}")

        forward: Dict[str, Any] = {}
        backward: Dict[str, Any] = {}

        forward_assignments: Dict[int, int] = {}
        backward_assignments: Dict[int, int] = {}

        forward_validated: Dict[int, bool] = {}
        backward_validated: Dict[int, bool] = {}

        affected_topics = set()

        for doc_id in doc_ids:
            pos = self._pos(doc_id)
            old_tid = int(self._assignments[pos])

            # --- assignment ---
            if old_tid != topic_id:
                backward_assignments[doc_id] = old_tid
                forward_assignments[doc_id] = topic_id
                self._assignments[pos] = topic_id
                affected_topics.add(old_tid)
                affected_topics.add(topic_id)

            # --- validation ---
            if set_validation is not None:
                old_val = bool(self._validated[pos])
                if old_val != bool(set_validation):
                    backward_validated[doc_id] = old_val
                    forward_validated[doc_id] = bool(set_validation)
                    self._validated[pos] = bool(set_validation)

        # Invalidate representations for affected topics
        for tid in affected_topics:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()

        if forward_assignments:
            forward["assignments"] = forward_assignments
            backward["assignments"] = backward_assignments

        if set_validation is not None and forward_validated:
            forward["validated"] = forward_validated
            backward["validated"] = backward_validated

        desc = f"Assigned {len(forward_assignments)} docs to topic {topic_id}"
        if set_validation is not None:
            desc += f" (validation set to {set_validation})"

        return forward, backward, desc

    def assign_docs_to_outlier(
        self,
        doc_ids: List[int],
        *,
        set_validation: Optional[bool] = None,
    ):
        """
        Assign the given documents to the outlier topic.
        """
        return self.assign_docs_to_topic(
            doc_ids,
            self.OUTLIER_ID,
            set_validation=set_validation,
        )

    @undoable
    def validate_doc(self, doc_id: int) -> Tuple[Dict, Dict, str]:
        """
        Mark a document as validated.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        pos = self._pos(doc_id)
        old_validated = bool(self._validated[pos])
        
        self._validated[pos] = True
        
        forward = {"validated": {doc_id: True}}
        backward = {"validated": {doc_id: old_validated}}
        description = f"Validated doc {doc_id}"
        
        return forward, backward, description
    
    # ----------------------------
    # Suggest assignment
    # ----------------------------
    def suggest_assignment(
        self,
        doc_id: int,
        *,
        mode: Optional[Union[str, Callable]] = "neighbors",
        neighbor_k: Optional[int] = None,
    ) -> Tuple[Optional[int], Tuple[float, float]]:
        """
        Suggest best LEAF topic for a document, respecting semantic spaces.
        """
        self._require_fitted()

        # doc representations that are global
        pos = self._pos(doc_id)

        # TF-IDF stays global (master DTM)
        if self._tfidf_transformer is not None and self._dtm is not None:
            doc_tfidf = self._tfidf_transformer.transform(self._dtm[pos]).toarray().ravel()
        else:
            doc_tfidf = None

        # Determine scorer
        # Backwards compatibility: treat mode=None as the default.
        if mode is None:
            mode = "neighbors"

        use_neighbors = False
        if mode == "embedding":
            scorer = embedding_scorer
        elif mode == "tfidf":
            scorer = tfidf_scorer
        elif mode == "harmonic":
            scorer = harmonic_scorer
        elif mode == "neighbors":
            scorer = None
            use_neighbors = True
        elif callable(mode):
            scorer = mode
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")

        # descend through semantic spaces until a leaf
        space_btm = self.btm  # start at root semantic space
        best_leaf_id: Optional[int] = None
        best_leaf_score: float = 0.0
        best_leaf_support: float = 1.0

        while True:
            candidates = self.assignment_candidates(space_btm, prefer_frontier=True)
            if not candidates:
                break
            support: float = 1.0
            topic_ids: List[int] = [t.topic_id for t in candidates]            # Neighbor-vote parameters
            if use_neighbors:
                # kVote is chosen to avoid overweighting large topics:
                #   kVote = clamp(min_topic_size_among_candidates, kVoteMin, kVoteMax)
                k_vote_max = int(neighbor_k) if neighbor_k is not None else int(getattr(space_btm, "_knn_vote_k", 50))
                k_vote_min = int(getattr(self, "_neighbor_vote_k_min", 5))
                min_size = min(int(t.get_count()) for t in candidates) if candidates else 0
                vote_k = max(k_vote_min, min(min_size, k_vote_max))

                cache_k = int(getattr(space_btm, "_knn_cache_k", vote_k))
                # Scan more than we vote with so we can skip inactive neighbors.
                neighbor_scan_k = min(cache_k, max(50, vote_k * 3))
            else:
                vote_k = 0
                neighbor_scan_k = 0
            if use_neighbors:
                # Neighbor-vote scoring: weighted vote among nearest-neighbor *documents*.
                # We condition on the current semantic space by ignoring neighbors whose current
                # assignment cannot be lifted into this space.
                neigh_doc_ids, neigh_sims = space_btm.get_knn(doc_id, k=neighbor_scan_k)
                candidate_set = set(int(tid) for tid in topic_ids)

                scores_map: Dict[int, float] = {int(tid): 0.0 for tid in topic_ids}
                counts_map: Dict[int, int] = {int(tid): 0 for tid in topic_ids}

                # Track per-topic nearest similarities for the distance penalty
                m_close = int(getattr(self, "_neighbor_close_m", 5))
                topic_sims: Dict[int, List[float]] = {int(tid): [] for tid in topic_ids}

                total_w_in = 0.0     # weight that contributes to candidates in this semantic space
                total_w_all = 0.0    # weight examined (positive similarities), regardless of space

                for nd, sim in zip(neigh_doc_ids, neigh_sims):
                    if nd is None:
                        continue
                    w = float(sim)
                    if w <= 0.0:
                        continue

                    total_w_all += w

                    assigned = int(self._assignments[self._pos(int(nd))])
                    if assigned < 0 or assigned not in self.topics:
                        continue

                    # Lift neighbor assignment up until we're in this semantic space.
                    tcur = self.topics[assigned]
                    while tcur.semantic_space is not space_btm:
                        if not isinstance(tcur.parent, InteractiveTopic):
                            tcur = None
                            break
                        tcur = tcur.parent
                    if tcur is None:
                        continue

                    # If assigned topic isn't a candidate (e.g., container), climb within this space.
                    while tcur.topic_id not in candidate_set:
                        if not isinstance(tcur.parent, InteractiveTopic):
                            break
                        if tcur.parent.semantic_space is not space_btm:
                            break
                        tcur = tcur.parent
                    if tcur.topic_id not in candidate_set:
                        continue

                    tid = int(tcur.topic_id)

                    # Per-topic cap: each candidate can contribute at most vote_k neighbors.
                    if counts_map[tid] >= vote_k:
                        continue

                    scores_map[tid] += w
                    total_w_in += w
                    counts_map[tid] += 1

                    if len(topic_sims[tid]) < m_close:
                        topic_sims[tid].append(w)

                    # Early stop: once all topics hit the per-topic cap.
                    if all(v >= vote_k for v in counts_map.values()):
                        break

                if total_w_in > 0:
                    raw_scores = np.asarray([scores_map[int(tid)] / total_w_in for tid in topic_ids], dtype=float)
                else:
                    raw_scores = np.zeros((len(topic_ids),), dtype=float)

                # Distance penalty: down-weight topics where the doc is atypically far from members.
                penalized_scores = raw_scores.copy()
                lambda_ = float(getattr(self, "_neighbor_penalty_lambda", 0.3))

                for i, tid in enumerate(topic_ids):
                    sims = topic_sims[int(tid)]
                    s_topic = float(np.mean(sims)) if sims else 0.0

                    topic_obj = self.topics[int(tid)]
                    mu, sigma = topic_obj._get_closeness_stats(m_close=m_close)

                    # z = how many robust "sigmas" below typical closeness we are
                    z = (mu - s_topic) / sigma if sigma > 0 else 0.0
                    if z > 0:
                        penalized_scores[i] = raw_scores[i] * float(np.exp(-lambda_ * z))

                scores = penalized_scores
                support = float(total_w_in / total_w_all) if total_w_all > 0 else 0.0

            else:
                # Embedding / lexical scoring against topic representations
                # IMPORTANT: embed the doc in the CURRENT semantic space, not root
                doc_emb, _ = space_btm.get_embeddings([doc_id])
                doc_embedding = doc_emb[0] if doc_emb.size else None

                topic_embs = [t.get_embedding() for t in candidates]
                topic_cts = [t.get_ctfidf() for t in candidates]

                topic_embeddings = np.asarray(topic_embs)
                topic_ctfidfs = np.asarray(topic_cts) if topic_cts else None

                scores = scorer(doc_embedding, doc_tfidf, topic_embeddings, topic_ctfidfs)

            best_idx = int(np.argmax(scores))
            best_id = int(topic_ids[best_idx])
            best_score = float(scores[best_idx])

            best_topic = self.topics[best_id]

            # If we can descend, do so
            if best_topic.btm is not None:
                space_btm = best_topic.btm
                continue

            # Otherwise, we've reached a leaf in this descent path
            best_leaf_score = best_score
            if best_leaf_score <= 1e-5:
                best_leaf_id = None
            else:
                best_leaf_id = best_id
            best_leaf_support = float(support)
            break

        return best_leaf_id, (best_leaf_score, best_leaf_support)

    @undoable
    def reassign_docs(
        self,
        doc_ids: Optional[Sequence[int]]= None,
        *,
        mode: Optional[Union[str, Any]] = "neighbors",
        neighbor_k: Optional[int] = None,
        threshold: float = float('+inf'),
        validate_reassignments: bool = True,
        ignore_consensus: bool = True,
        export_path: Optional[str] = None,
    ):
        """
        Suggest and optionally reassign documents to best-fit topics.

        Args:
            doc_ids: Iterable of document ids to consider (default: all docs).
            mode: Scoring mode for suggestions.
            neighbor_k: k parameter for neighbor-vote scoring.
            threshold: Minimum score to accept reassignment (default: infinity, i.e., no reassignment).
            validate_reassignments: Whether to mark reassigned docs as validated.
            ignore_consensus: Whether to skip docs where suggested topic matches current.
            export_path: Optional path to export suggestions CSV.

        Returns:
            A DataFrame with suggested assignments.
        """
        self._require_fitted()

        if doc_ids is None:
            doc_ids = self._doc_ids

        results = []
        to_reassign: Dict[int, Tuple[int, float]] = {}

        for doc_id in doc_ids:
            suggested_topic_id, (score, support) = self.suggest_assignment(
                doc_id, mode=mode, neighbor_k=neighbor_k
            )
            suggested_topic = suggested_topic_id if suggested_topic_id is not None else self.OUTLIER_ID
            current_topic = int(self._assignments[self._pos(doc_id)])

            if ignore_consensus and suggested_topic == current_topic:
                continue

            suggested_label = (
                self.topics[suggested_topic].label if suggested_topic in self.topics else None
            )
            current_label = (
                self.topics[current_topic].label if current_topic in self.topics else None
            )
            row = {
                "doc_id": doc_id,
                "current_topic": current_topic,
                "current_label": current_label,
                "suggested_topic": suggested_topic,
                "suggested_label": suggested_label,
                "score": float(score),
                "support": float(support),
                "text": self._texts[self._pos(doc_id)],
                "reassigned": False,
            }
            results.append(row)

            if score >= threshold:
                to_reassign[doc_id] = (int(suggested_topic), float(score))

        df = pd.DataFrame(results).sort_values(['score', 'support'], ascending=[False, False])

        if export_path is not None:
            df['reassigned_topic_id'] = ''
            df.to_csv(export_path, index=False)

        if not to_reassign:
            return df

        old_assignments: Dict[int, int] = {}
        old_strengths: Dict[int, float] = {}
        old_validated: Dict[int, bool] = {}

        new_assignments: Dict[int, int] = {}
        new_strengths: Dict[int, float] = {}
        new_validated: Dict[int, bool] = {}

        for doc_id, (new_tid, new_score) in to_reassign.items():
            pos = self._pos(doc_id)
            old_assignments[doc_id] = int(self._assignments[pos])
            old_strengths[doc_id] = float(self._strengths[pos])
            old_validated[doc_id] = bool(self._validated[pos])

            self._assignments[pos] = int(new_tid)
            new_assignments[doc_id] = int(new_tid)
            self._strengths[pos] = float(new_score)
            new_strengths[doc_id] = float(new_score)

            if validate_reassignments:
                self._validated[pos] = True
                new_validated[doc_id] = True

        affected = set(new_assignments.values()) | set(old_assignments.values())
        for tid in affected:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()

        df.loc[df["doc_id"].isin(to_reassign.keys()), "reassigned"] = True

        forward = {"assignments": new_assignments, "strengths": new_strengths}
        backward = {"assignments": old_assignments, "strengths": old_strengths}

        if validate_reassignments:
            forward["validated"] = new_validated
            backward["validated"] = old_validated

        description = f"Refit docs: reassigned {len(new_assignments)}"
        return forward, backward, description, df

    def reassign_outliers(
        self,
        *,
        mode: Optional[Union[str, Any]] = "neighbors",
        neighbor_k: Optional[int] = None,
        threshold: float = float('+inf'),
        validate_reassignments: bool = True,
        ignore_consensus: bool = True,
        export_path: Optional[str] = None,
    ):
        self._require_fitted()

        outlier_mask = self._assignments == self.OUTLIER_ID
        outlier_doc_ids = [int(doc_id) for doc_id in self._doc_ids[outlier_mask]]

        return self.reassign_docs(
            doc_ids=outlier_doc_ids,
            mode=mode,
            neighbor_k=neighbor_k,
            threshold=threshold,
            validate_reassignments=validate_reassignments,
            ignore_consensus=ignore_consensus,
            export_path=export_path,
        )
    
    # ----------------------------
    # Topic operations
    # ----------------------------
    
    @undoable
    def merge_topics(
        self,
        topic_ids: List[int],
        new_label: Optional[str] = None,
    ) -> Tuple[Dict, Dict, str]:
        """
        Merge multiple topics into one, with hierarchy rules:
        - Topics can only be merged if they are siblings (same parent object).
        - If any merged topics have children, those children are reparented to the new merged topic.
        - Only documents assigned directly to the merged topics (exclude descendants) are moved.
        """
        self._require_fitted()

        # normalize + validate ids
        topic_ids = [int(t) for t in topic_ids]
        topic_ids = [t for t in topic_ids if t in self.topics]
        topic_ids = sorted(set(topic_ids))

        if len(topic_ids) < 2:
            raise ValueError("Must merge at least 2 existing topics")

        # disallow outlier merges
        if self.OUTLIER_ID in topic_ids:
            raise ValueError("Cannot merge the outlier topic")

        # siblings-only constraint 
        parents = [self.topics[tid].parent for tid in topic_ids]
        parent0 = parents[0]
        if any(p is not parent0 for p in parents[1:]):
            raise ITMError("Can only merge sibling topics (topics must share the same parent).")

        # create merged topic under the shared parent 
        merged_id = self._new_topic_id()
        self._ensure_topic(merged_id, label=new_label, parent=parent0)
        merged_topic = self.topics[merged_id]

        # capture old state for undo 
        old_assignments: Dict[int, int] = {}

        # parent-relabel bookkeeping for children reparenting
        forward_topic_parents: Dict[int, object] = {merged_id: parent0}
        backward_topic_parents: Dict[int, object] = {}

        # move docs (direct only) + deactivate originals 
        moved_doc_ids: List[int] = []

        for tid in topic_ids:
            t = self.topics[tid]

            # move docs assigned *directly* to this topic (exclude descendants)
            doc_ids = t.get_doc_ids(include_descendants=False)
            for doc_id in doc_ids:
                doc_id = int(doc_id)
                pos = self._pos(doc_id)

                old_assignments[doc_id] = int(self._assignments[pos])

                self._assignments[pos] = int(merged_id)
                moved_doc_ids.append(doc_id)

            t.active = False

        # reparent any children of merged topics to the new merged topic 
        # (children are topics whose parent is one of the merged topics)
        for child in list(self.topics.values()):
            if not isinstance(child.parent, InteractiveTopic):
                continue
            if child.parent.topic_id in topic_ids:
                backward_topic_parents[child.topic_id] = child.parent
                child.parent = merged_topic
                forward_topic_parents[child.topic_id] = merged_topic

        # invalidate representations 
        for tid in topic_ids:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()
        merged_topic.invalidate_representations()

        # undo/redo payloads 
        forward = {
            "assignments": {doc_id: merged_id for doc_id in moved_doc_ids},
            "new_topic": merged_id,
            "deactivate_topics": topic_ids,
        }
        if forward_topic_parents:
            forward["topic_parents"] = forward_topic_parents

        backward = {
            "assignments": old_assignments,
            "remove_topic": merged_id,
            "reactivate_topics": topic_ids,
        }
        if backward_topic_parents:
            backward["topic_parents"] = backward_topic_parents

        if new_label is None:
            # generate label without creating a separate undo step
            old_label = merged_topic._label
            merged_topic.auto_label(_disable_tracking=True)

            # include label change in this merge's undo payload
            forward["topic_labels"] = {merged_id: merged_topic._label}
            backward["topic_labels"] = {merged_id: old_label}

        description = f"Merged {len(topic_ids)} sibling topics into topic {merged_id}"
        return forward, backward, description, merged_id

    @undoable
    def merge_similar_topics(
        self,
        threshold: float,
        *,
        parent: Optional[Union[int, InteractiveTopic, "InteractiveTopicModel"]] = None,
        mode: str = "embeddings",
        include_inactive: bool = True,
        linkagefun: Optional[Callable] = None,
        new_label_fn: Optional[Callable[[List[int]], str]] = None,
    ) -> Tuple[Dict, Dict, str, List[int]]:
        """
        Merge sibling clusters under `parent` whose linkage distances are <= threshold.

        Uses one hierarchy computation + one global cut:
        - if ((X Y) Z) merges at <= threshold, then {X,Y,Z} merges in one shot.
        """
        self._require_fitted()
        if threshold < 0:
            raise ValueError("threshold must be >= 0")

        # Expect: (linkage_matrix, labels, child_topic_ids)
        linkage_matrix, labels, child_topic_ids = self.get_topic_hierarchy(parent=parent,
                                                                           mode=mode,
                                                                           include_inactive=include_inactive,
                                                                           linkagefun=linkagefun)
        child_topic_ids = [int(t) for t in child_topic_ids]

        if len(child_topic_ids) < 2:
            return {}, {}, "No merge performed (fewer than 2 children).", []

        from scipy.cluster.hierarchy import fcluster

        # One cut at distance threshold
        cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")

        groups: Dict[int, List[int]] = {}
        for tid, cid in zip(child_topic_ids, cluster_ids):
            groups.setdefault(int(cid), []).append(int(tid))

        merge_groups = [sorted(g) for g in groups.values() if len(g) >= 2]
        if not merge_groups:
            return {}, {}, "No merge performed (no clusters below threshold).", []

        # compose undo states across multiple merge_topics calls
        forward_all: Dict[str, Any] = {}
        backward_all: Dict[str, Any] = {}
        merged_ids: List[int] = []

        def _merge_state(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            """Merge one edit-state dict into another, respecting _apply_edit keys."""
            for k, v in src.items():
                if k in ("assignments", "strengths", "validated", "topic_labels", "topic_parents", "topic_btms", "topic_active"):
                    dst.setdefault(k, {}).update(v)
                elif k in ("deactivate_topics", "reactivate_topics", "new_topics", "remove_topics", "deleted_topics"):
                    dst.setdefault(k, []).extend(list(v))
                elif k == "new_topic":
                    dst.setdefault("new_topics", []).append(v)
                elif k == "remove_topic":
                    dst.setdefault("remove_topics", []).append(v)
                else:
                    # Handle new keys here.
                    dst[k] = v

        # Disable tracking for inner merges; merge_similar_topics is the only undo step.
        with disable_tracking(self):
            for group in merge_groups:
                label = new_label_fn(group) if new_label_fn else None

                fwd, bwd, _desc, merged_id = self.merge_topics(group, new_label=label, _disable_tracking=True)
                merged_ids.append(int(merged_id))

                _merge_state(forward_all, fwd)
                _merge_state(backward_all, bwd)

        # Clean up duplicates / make deterministic
        if "new_topics" in forward_all:
            forward_all["new_topics"] = sorted(set(forward_all["new_topics"]))
        if "remove_topics" in backward_all:
            backward_all["remove_topics"] = sorted(set(backward_all["remove_topics"]))
        if "deactivate_topics" in forward_all:
            forward_all["deactivate_topics"] = sorted(set(forward_all["deactivate_topics"]))
        if "reactivate_topics" in backward_all:
            backward_all["reactivate_topics"] = sorted(set(backward_all["reactivate_topics"]))

        desc = (
            f"Merged {sum(len(g) for g in merge_groups)} topics across {len(merge_groups)} clusters "
            f"under parent {getattr(parent, 'topic_id', 'ITM')} (threshold={threshold})."
        )
        print(desc)

        return forward_all, backward_all, desc, merged_ids

    @undoable
    def merge_topics_by_label(
        self,
        exclude_inactive: bool = True,
    ) -> Tuple[Dict, Dict, str, List[int]]:
        """
        Merge topics that share the same label, but only when they are siblings.

        Args:
            exclude_inactive: If True, skip inactive topics.

        Returns:
            (forward_state, backward_state, description, merged_topic_ids)
        """
        self._require_fitted()

        # Group by (parent_object, label) so merges are always sibling-safe.
        groups: Dict[Tuple[object, str], List[int]] = {}

        for tid, topic in self.topics.items():
            if tid == self.OUTLIER_ID:
                continue
            if exclude_inactive and not topic.active:
                continue
            if not topic._label:
                continue

            key = (topic.parent, topic._label)
            groups.setdefault(key, []).append(int(tid))

        merge_groups: List[Tuple[object, str, List[int]]] = []
        for (parent_obj, label), tids in groups.items():
            tids = sorted(set(int(t) for t in tids if t in self.topics))
            if len(tids) >= 2:
                merge_groups.append((parent_obj, label, tids))

        if not merge_groups:
            return {}, {}, "No topics with duplicate labels found (among siblings).", []

        # Compose undo/redo states across multiple merge_topics calls (same as merge_similar_topics).
        forward_all: Dict[str, Any] = {}
        backward_all: Dict[str, Any] = {}
        merged_ids: List[int] = []

        def _merge_state(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            """Merge one edit-state dict into another, respecting _apply_edit keys."""
            for k, v in src.items():
                if k in ("assignments", "strengths", "validated", "topic_labels", "topic_parents", "topic_btms", "topic_active"):
                    dst.setdefault(k, {}).update(v)
                elif k in ("deactivate_topics", "reactivate_topics", "new_topics", "remove_topics", "deleted_topics"):
                    dst.setdefault(k, []).extend(list(v))
                elif k == "new_topic":
                    dst.setdefault("new_topics", []).append(v)
                elif k == "remove_topic":
                    dst.setdefault("remove_topics", []).append(v)
                else:
                    dst[k] = v

        # Disable tracking for inner merges; merge_topics_by_label is the only undo step.
        with disable_tracking(self):
            for _parent_obj, label, tids in merge_groups:
                # merge_topics enforces siblings-only itself, but grouping by parent prevents failures up front.
                fwd, bwd, _desc, merged_id = self.merge_topics(
                    tids,
                    new_label=label,
                    _disable_tracking=True,
                )
                merged_ids.append(int(merged_id))
                _merge_state(forward_all, fwd)
                _merge_state(backward_all, bwd)

        # Clean up duplicates / determinism
        if "new_topics" in forward_all:
            forward_all["new_topics"] = sorted(set(forward_all["new_topics"]))
        if "remove_topics" in backward_all:
            backward_all["remove_topics"] = sorted(set(backward_all["remove_topics"]))
        if "deactivate_topics" in forward_all:
            forward_all["deactivate_topics"] = sorted(set(forward_all["deactivate_topics"]))
        if "reactivate_topics" in backward_all:
            backward_all["reactivate_topics"] = sorted(set(backward_all["reactivate_topics"]))

        merged_count = sum(len(tids) for _, _, tids in merge_groups)
        desc = f"Merged {merged_count} topics across {len(merge_groups)} sibling-label groups."
        return forward_all, backward_all, desc, merged_ids

    @undoable
    def create_topic(
        self,
        label: Optional[str] = None,
        doc_ids: Optional[Sequence[int]] = None,
        *,
        parent_id: Optional[int] = None,
        validate: bool = False,
        confidence: Optional[float] = 1.0,
    ):
        """
        Create a new topic and optionally assign docs.

        Args:
            label: Optional topic label.
            doc_ids: Optional list of document IDs to assign.
            parent_id: Optional parent topic ID.
            validate: Whether to mark assigned docs as validated.
            confidence: Optional confidence score for assignments.

        Returns:
            (forward_state, backward_state, description, topic)
        """
        self._require_fitted()

        # --- Resolve parent object ---
        if parent_id is None:
            parent_obj = self
        else:
            parent_topic = self._ensure_topic(parent_id)
            parent_obj = parent_topic

        # --- Create topic ---
        topic_id = self._new_topic_id()
        self._ensure_topic(topic_id, label=label, parent=parent_obj)
        topic = self.topics[topic_id]

        # --- Undo/redo state ---
        forward: dict = {"new_topic": topic_id}
        backward: dict = {"remove_topic": topic_id}
        forward.setdefault("topic_parents", {})[topic_id] = parent_obj
        if label is not None:
            forward.setdefault("topic_labels", {})[topic_id] = label

        # --- Optional doc assignment ---
        if doc_ids is not None:
            old_assignments = {}
            old_strengths = {}
            old_validated = {}

            new_assignments = {}
            new_strengths = {}
            new_validated = {}

            for doc_id in doc_ids:
                doc_id = int(doc_id)
                pos = self._pos(doc_id)

                # capture previous
                old_assignments[doc_id] = int(self._assignments[pos])
                old_strengths[doc_id] = float(self._strengths[pos])
                old_validated[doc_id] = bool(self._validated[pos])

                # apply new
                self._assignments[pos] = int(topic_id)
                new_assignments[doc_id] = int(topic_id)

                if confidence is not None:
                    self._strengths[pos] = float(confidence)
                    new_strengths[doc_id] = float(confidence)

                if validate:
                    self._validated[pos] = True
                    new_validated[doc_id] = True

            forward["assignments"] = new_assignments
            backward["assignments"] = old_assignments

            if confidence is not None:
                forward["strengths"] = new_strengths
                backward["strengths"] = old_strengths

            if validate:
                forward["validated"] = new_validated
                backward["validated"] = old_validated

            # invalidate representations for affected topics (new + any previous topics)
            affected = {topic_id, *set(old_assignments.values())}
            for tid in affected:
                if tid in self.topics:
                    self.topics[tid].invalidate_representations()

        description = f"Created topic {topic_id}" + (f" ('{label}')" if label else "")

        # Extra return: topic object
        return forward, backward, description, topic


    @undoable
    def rename_topic(self, topic_id: int, label: str) -> Tuple[Dict, Dict, str]:
        """
        Rename a topic.
        
        Args:
            topic_id: Topic ID.
            label: New label.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        if topic_id not in self.topics:
            raise ValueError(f"Unknown topic_id: {topic_id}")
        
        topic = self.topics[topic_id]
        old_label = topic._label
        topic._label = label
        
        forward = {"topic_labels": {topic_id: label}}
        backward = {"topic_labels": {topic_id: old_label}}
        description = f"Renamed topic {topic_id} to '{label}'"
        
        return forward, backward, description
    
    # ----------------------------
    # Information retrieval
    # ----------------------------
    
    def get_doc_from_id(self, doc_id: int) -> str:
        """
        Return the raw text for a given document id.
        """
        doc_id = int(doc_id)
        try:
            pos = self._pos(doc_id)
        except Exception:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        return self._texts[pos]
    
    def get_topic_info(self,
                       top_words_n: int = 10, 
                       show_inactive: bool = False,
                       show_outliers: bool = True) -> pd.DataFrame:
        """
        Get summary information for topics.
        
        Args:
            top_words_n: Number of top words to include.
            show_inactive: Whether to include inactive topics.
            show_outliers: Whether to include the outlier topic.
        Returns:
            DataFrame with topic information.
        """
        self._require_fitted()
        
        rows = []
        for topic_id, topic in self.topics.items():
            if ((topic_id != self.OUTLIER_ID and not topic.active and not show_inactive) 
                or (topic_id == self.OUTLIER_ID and not show_outliers)):
                continue
            
            count = topic.get_count(include_descendants=True)
            top_terms = topic.get_top_terms(n=top_words_n)
            terms_str = ", ".join([term for term, _ in top_terms])
            
            rows.append(
                {
                    "topic_id": topic_id,
                    "label": topic.label,
                    "count": count,
                    "top_terms": terms_str,
                }
            )
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("topic_id")
        return df

    def get_representative_documents(
        self,
        topics: Optional[List[int]] = None,
    ) -> Dict[Tuple[int, str], List[str]]:
        """
        Get representative documents for topics.
        
        Args:
            topics: List of topic IDs (None = all active topics).
            
        Returns:
            Dict mapping (topic_id, label) to list of example texts.
        """
        self._require_fitted()
        
        if topics is None:
            topics = [tid for tid, t in self.topics.items() if t.active and tid >= 0]
        
        result = {}
        for topic_id in topics:
            if topic_id not in self.topics:
                continue
            
            topic = self.topics[topic_id]
            examples = topic.get_representative_docs()
            result[(topic_id, topic.label)] = examples
        
        return result

    def get_examples(
        self,
        topics: Optional[List[int]] = None,
        n: int = 5,
        include_descendants: bool = True,
        random_state: Optional[int] = None,
    ) -> Dict[Tuple[int, str], List[str]]:
        """
        Get example documents for topics.
        
        Args:
            topics: List of topic IDs (None = all active topics).
            n: Number of examples per topic.
            include_descendants: Whether to include descendant topics.
            random_state: Random seed.
            
        Returns:
            Dict mapping (topic_id, label) to list of example texts.
        """
        self._require_fitted()
        
        if topics is None:
            topics = [tid for tid, t in self.topics.items() if t.active and tid >= 0]
        
        result = {}
        for topic_id in topics:
            if topic_id not in self.topics:
                continue
            
            topic = self.topics[topic_id]
            examples = topic.get_examples(
                n=n, include_descendants=include_descendants, random_state=random_state
            )
            result[(topic_id, topic.label)] = examples
        
        return result
    
    def get_topic_hierarchy(
        self,
        *,
        parent: Optional[Union[int, "InteractiveTopic", "InteractiveTopicModel"]] = None,
        mode: str = "embeddings",
        include_inactive: bool = True,
        linkagefun: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[int]]:
        """
        Compute pairwise hierarchical clustering among the children of `parent`.

        Args:
            parent: Parent topic ID or object (None = root ITM).
            mode: Representation mode ("embeddings" or "ctfidf").
            include_inactive: Whether to include inactive topics.
            linkagefun: Optional custom linkage function.

        Returns
        -------
        linkage_matrix : ndarray (n-1, 4)
            SciPy linkage matrix.
        labels : list[str]
            Topic labels in leaf order.
        topic_ids : list[int]
            Topic IDs corresponding to labels.
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        self._require_fitted()

        # Resolve parent object
        if parent is None:
            parent_obj = self
        elif isinstance(parent, int):
            if parent not in self.topics:
                raise ITMError(f"Unknown parent topic_id: {parent}")
            parent_obj = self.topics[parent]
        else:
            parent_obj = parent

        # Collect children of parent
        topics = [
            t for t in self.topics.values()
            if t.topic_id >= 0
            and getattr(t, "parent", None) is parent_obj
            and (include_inactive or t.active)
            and t.get_count(include_descendants=True) > 0
        ]

        if len(topics) < 2:
            raise ITMError("Need at least two child topics to build a hierarchy.")

        # Build representations
        vectors = []
        labels = []
        topic_ids = []

        for t in topics:
            if mode == "embeddings":
                vec = t.get_embedding()
            elif mode == "ctfidf":
                vec = t.get_ctfidf()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if vec is None:
                continue

            if not np.all(np.isfinite(vec)):
                print(f"Skipping topic {t.topic_id} due to non-finite vector.")
                continue

            # Must be non-zero norm for cosine distance
            if np.linalg.norm(vec) == 0.0:
                print(f"Skipping topic {t.topic_id} due to zero vector.")
                continue

            vectors.append(vec)
            labels.append(t.label)
            topic_ids.append(t.topic_id)

        topic_matrix = np.vstack(vectors)
        distances = pdist(topic_matrix, metric="cosine")

        if linkagefun is None:
            linkage_matrix = linkage(distances, method="single", optimal_ordering=True)
        else:
            linkage_matrix = linkagefun(distances)

        return linkage_matrix, labels, topic_ids
    
    def get_active_topics(self) -> List[int]:
        """Get list of active topic IDs."""
        return [tid for tid, t in self.topics.items() if t.active and tid >= 0]
    
    def get_outlier_count(self) -> int:
        """Get number of documents assigned to outliers."""
        return int((self._assignments == self.OUTLIER_ID).sum())
    
    def get_validated_count(self) -> int:
        """Get number of validated documents."""
        return int(self._validated.sum())

    def similarity_search(
        self,
        search_texts: Union[str, List[str]],
        embedding_model: Optional[Union[str, Embedder]] = None,
    ) -> pd.DataFrame:
        """
        Compute cosine similarity between documents and one or more search strings.

        Each search string becomes its own similarity column.

        Args:
            search_texts: Single string or list of strings.
            embedding_model: Embedder instance or model name (uses default if None).

        Returns:
            DataFrame with columns:
                doc_id, text, topic_id, label,
                sim_0, sim_1, ..., sim_k
            where sim_i corresponds to search_texts[i].
        """
        self._require_fitted()

        # Normalize input
        if isinstance(search_texts, str):
            queries = [search_texts]
        else:
            queries = list(search_texts)

        if len(queries) == 0:
            raise ValueError("search_texts must contain at least one query string")

        # Resolve embedder
        if embedding_model is None:
            if self.btm.embedder is None:
                raise ValueError("No embedder available and none provided")
            embedder = self.btm.embedder
        elif isinstance(embedding_model, str):
            embedder = SentenceTransformerEmbedder(embedding_model)
        else:
            embedder = embedding_model

        # Embed queries (batch)
        query_embs = np.asarray(embedder.encode(queries))
        if query_embs.ndim == 1:
            query_embs = query_embs.reshape(1, -1)

        # Document embeddings (cached in BTM)
        doc_embs = np.asarray(self.btm._embeddings)
        if doc_embs.ndim != 2:
            raise ValueError("BTM embeddings have unexpected shape")

        # Compute similarity matrix: (n_docs, n_queries)
        sims = cosine_similarity(doc_embs, query_embs)

        # Build base DataFrame
        rows = []
        for pos in range(len(self._doc_ids)):
            doc_id = self._doc_ids[pos]
            topic_id = int(self._assignments[pos])
            label = self.topics[topic_id].label if topic_id in self.topics else str(topic_id)

            row = {
                "doc_id": doc_id,
                "text": self._texts[pos],
                "topic_id": topic_id,
                "label": label,
            }

            # Add similarity columns
            for i in range(len(queries)):
                row[f"sim_{i}"] = float(sims[pos, i])

            rows.append(row)

        df = pd.DataFrame(rows).sort_values(by=[f"sim_{i}" for i in range(len(queries))], ascending=[False]*len(queries))

        return df
    
    @undoable
    def label_topics(
        self, label_dict: Dict[int, str], skip_missing: bool = True
    ) -> Tuple[Dict, Dict, str]:
        """
        Bulk label topics.
        
        Args:
            label_dict: Mapping of topic_id to label.
            skip_missing: If False, raise error for unknown topic_ids.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        old_labels = {}
        new_labels = {}
        
        for tid, label in label_dict.items():
            if tid not in self.topics:
                if skip_missing:
                    continue
                raise ValueError(f"Unknown topic_id: {tid}")
            
            old_labels[tid] = self.topics[tid]._label
            new_labels[tid] = label
            self.topics[tid]._label = label
        
        forward = {"topic_labels": new_labels}
        backward = {"topic_labels": old_labels}
        description = f"Labeled {len(new_labels)} topics"
        
        return forward, backward, description
    
    @undoable
    def archive_topic(self, topic_id: int) -> Tuple[Dict, Dict, str]:
        """
        Deactivate a topic (mark as inactive).
        
        Docs remain assigned but topic won't appear in active lists.
        
        Args:
            topic_id: Topic ID to archive.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        if topic_id not in self.topics:
            raise ValueError(f"Unknown topic_id: {topic_id}")
        
        old_active = self.topics[topic_id].active
        self.topics[topic_id].active = False
        
        forward = {"topic_active": {topic_id: False}}
        backward = {"topic_active": {topic_id: old_active}}
        description = f"Archived topic {topic_id}"
        
        return forward, backward, description
    
    @undoable
    def group_topics(
        self, topic_ids: List[int], label: Optional[str] = None
    ) -> Tuple[Dict, Dict, str]:
        """
        Group existing topics under a new parent topic.
        
        Creates a new parent topic and makes the given topics its children.
        No documents are reassigned - hierarchical container only.
        
        Args:
            topic_ids: List of topic IDs to group.
            label: Optional label for the new parent topic.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        if len(topic_ids) < 2:
            raise ValueError("Need at least 2 topics to group")
        
        # Validate all topics exist
        for tid in topic_ids:
            if tid not in self.topics:
                raise ValueError(f"Unknown topic_id: {tid}")
        
        # Create parent topic
        parent_id = self._new_topic_id()
        self._ensure_topic(parent_id, label=label, parent=self)
        parent = self.topics[parent_id]
        
        # Save old parents
        old_parents = {}
        for tid in topic_ids:
            old_parents[tid] = self.topics[tid].parent
            self.topics[tid].parent = parent
        
        # Compute parent embedding from children
        child_embeddings = []
        for tid in topic_ids:
            try:
                emb = self.topics[tid].get_embedding()
                if emb is not None and len(emb) > 0:
                    child_embeddings.append(emb)
            except:
                pass
        
        if child_embeddings:
            parent._cached_embedding = np.mean(child_embeddings, axis=0)
        
        forward = {
            "new_topic": parent_id,
            "topic_parents": {tid: parent for tid in topic_ids},
        }
        backward = {
            "remove_topic": parent_id,
            "topic_parents": old_parents,
        }
        description = f"Grouped {len(topic_ids)} topics under parent {parent_id}"
        
        return forward, backward, description
    
    # ----------------------------
    # Undo/redo
    # ----------------------------
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    def undo(self, steps: int = 1) -> None:
        """
        Undo recent operations.
        
        Args:
            steps: Number of operations to undo.
        """
        for _ in range(steps):
            if not self.can_undo():
                print("Nothing to undo")
                return
            
            edit = self._undo_stack.pop()
            self._apply_edit(edit, direction="backward")
            self._redo_stack.append(edit)
    
    def redo(self, steps: int = 1) -> None:
        """
        Redo undone operations.
        
        Args:
            steps: Number of operations to redo.
        """
        for _ in range(steps):
            if not self.can_redo():
                print("Nothing to redo")
                return
            
            edit = self._redo_stack.pop()
            self._apply_edit(edit, direction="forward")
            self._undo_stack.append(edit)
    
    def _apply_edit(self, edit: Edit, *, direction: str) -> None:
        """
        Apply an edit in forward or backward direction.
        
        Args:
            edit: Edit to apply.
            direction: 'forward' or 'backward'.
        """
        state = edit.forward_state if direction == "forward" else edit.backward_state
        
        with disable_tracking(self):
            # Apply assignments
            if "assignments" in state:
                for doc_id, topic_id in state["assignments"].items():
                    pos = self._pos(doc_id)
                    self._assignments[pos] = topic_id
            
            # Apply strengths
            if "strengths" in state:
                for doc_id, strength in state["strengths"].items():
                    pos = self._pos(doc_id)
                    self._strengths[pos] = strength
            
            # Apply validated flags
            if "validated" in state:
                for doc_id, validated in state["validated"].items():
                    pos = self._pos(doc_id)
                    self._validated[pos] = validated
            
            # Apply topic labels
            if "topic_labels" in state:
                for topic_id, label in state["topic_labels"].items():
                    if topic_id in self.topics:
                        self.topics[topic_id]._label = label
            
            # Create new topics
            if "new_topic" in state:
                self._ensure_topic(state["new_topic"])
            
            if "new_topics" in state:
                for topic_id in state["new_topics"]:
                    self._ensure_topic(topic_id)
            
            # Remove topics
            if "remove_topic" in state:
                self.topics.pop(state["remove_topic"], None)
            
            if "remove_topics" in state:
                for topic_id in state["remove_topics"]:
                    self.topics.pop(topic_id, None)
            
            # Activate/deactivate topics
            if "deactivate_topics" in state:
                for topic_id in state["deactivate_topics"]:
                    if topic_id in self.topics:
                        self.topics[topic_id].active = False
            
            if "reactivate_topics" in state:
                for topic_id in state["reactivate_topics"]:
                    if topic_id in self.topics:
                        self.topics[topic_id].active = True
            
            # Apply topic active state
            if "topic_active" in state:
                for topic_id, active in state["topic_active"].items():
                    if topic_id in self.topics:
                        self.topics[topic_id].active = active
            
            # Apply topic parent relationships
            if "topic_parents" in state:
                for topic_id, parent in state["topic_parents"].items():
                    if topic_id in self.topics:
                        self.topics[topic_id].parent = parent

            # Apply topic BTMs (semantic space changes)
            if "topic_btms" in state:
                for topic_id, btm in state["topic_btms"].items():
                    if topic_id in self.topics:
                        self.topics[topic_id].btm = btm

            # --- Handle deletion / restoration of full topic snapshots ---
            # Forward key: "deleted_topics" -> remove the topic id(s)
            if "deleted_topics" in state:
                for tid in state["deleted_topics"]:
                    # remove if present
                    self.topics.pop(tid, None)

            # Backward/restore key: "restore_topics" -> re-insert deep-copied snapshots
            if "restore_topics" in state:
                for tid, meta in state["restore_topics"].items():
                    topic_snapshot = meta["topic_snapshot"]
                    parent_id = meta["parent_id"]
                    child_ids = meta.get("child_ids", [])

                    # Reattach snapshot to this ITM
                    topic_snapshot.itm = self
                    topic_snapshot.topic_id = tid
                    self.topics[tid] = topic_snapshot

                    # Restore parent
                    if parent_id is None:
                        # Root-level topic
                        topic_snapshot.parent = self
                    else:
                        parent_topic = self.topics.get(parent_id)
                        topic_snapshot.parent = parent_topic if parent_topic is not None else self

                    # Restore children
                    for cid in child_ids:
                        child = self.topics.get(cid)
                        if child is not None:
                            child.parent = topic_snapshot

                    # Ensure topic id counter stays monotonic
                    if tid >= self._next_topic_id:
                        self._next_topic_id = tid + 1

    
    def get_history(
        self, stack: str = "undo", reverse: bool = True, include_timestamp: bool = True
    ) -> List[str]:
        """
        Get edit history.
        
        Args:
            stack: 'undo' or 'redo'.
            reverse: Whether to reverse order (most recent first).
            include_timestamp: Whether to include timestamps.
            
        Returns:
            List of edit descriptions.
        """
        edits = self._undo_stack if stack == "undo" else self._redo_stack
        
        if reverse:
            edits = reversed(edits)
        
        if include_timestamp:
            return [str(edit) for edit in edits]
        else:
            return [edit.description for edit in edits]
    
    # ----------------------------
    # Visualization methods
    # ----------------------------
    
    def visualize_documents(
        self,
        *,
        use_reduced: bool = True,
        title: Optional[str] = None,
        reducer: Optional[Reducer] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize documents in 2D space.
        
        Args:
            title: Optional title for the plot.
            reducer: Optional 2D reducer (if None, uses UMAP with n_components=2).
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        from .visualization import visualize_documents as _visualize_documents
        return _visualize_documents(
            self, 
            plot_btm=self.btm, 
            doc_ids=list(self._doc_ids),
            use_reduced=use_reduced,
            title=title, 
            reducer=reducer,
            save_path=save_path
        )
    
    def visualize_documents_with_search(
        self,
        search_string: str,
        *,
        use_reduced: bool = True,
        title: Optional[str] = None,
        reducer: Optional[Reducer] = None,
        search_embedder: Optional[Union[str, Embedder]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize documents with search highlighting.
        
        Args:
            search_string: Text to search for similarity.
            title: Optional title for the plot.
            reducer: Optional 2D reducer (if None, uses UMAP with n_components=2).
            search_embedder: Embedder instance or model name for search.
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        from .visualization import visualize_documents_with_search as _visualize_documents_with_search
        return _visualize_documents_with_search(
            self,
            plot_btm=self.btm,
            doc_ids=list(self._doc_ids),
            search_string=search_string,
            use_reduced=use_reduced,
            title=title,
            reducer=reducer,
            search_embedder=search_embedder,
            save_path=save_path,
        )

    def visualize_topic_hierarchy(
        self,
        *,
        parent=None,
        mode: str = "embeddings",
        include_inactive: bool = True,
        linkagefun: Optional[Callable] = None,
        orientation: str = "right",
        title: str = "Topic Hierarchy",
        save_path: Optional[str] = None,
    ):
        linkage_matrix, labels, _ = self.get_topic_hierarchy(
            parent=parent,
            mode=mode,
            include_inactive=include_inactive,
            linkagefun=linkagefun,
        )

        from interactive_topic_model.visualization import visualize_topic_hierarchy as viz

        return viz(
            linkage_matrix=linkage_matrix,
            labels=labels,
            orientation=orientation,
            title=title,
            save_path=save_path,
        )

    @undoable
    def import_assignments(
        self,
        data: Union[str, pd.DataFrame],
        validate_imports: bool = True,
    ) -> Tuple[Dict, Dict, str]:
        """
        Import document reassignments from CSV or DataFrame.
        
        Args:
            data: CSV file path or pandas DataFrame.
            validate_imports: If True, mark reassigned documents as validated.
            
        Returns:
            Tuple of (forward_state, backward_state, description) for undo/redo.
        """
        self._require_fitted()
        
        # Load DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        # Validate required columns
        required_cols = ["doc_id", "reassigned_topic_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate doc_ids exist
        valid_doc_ids = set(self._doc_ids)
        invalid_doc_ids = df[~df["doc_id"].isin(valid_doc_ids)]["doc_id"].tolist()
        if invalid_doc_ids:
            raise ValueError(f"Unknown doc_ids: {invalid_doc_ids}")
        
        # Validate reassigned_topic_ids exist
        valid_topic_ids = set(self.topics.keys())
        non_null_reassigned = df["reassigned_topic_id"].dropna().astype(int)
        invalid_topic_ids = non_null_reassigned[~non_null_reassigned.isin(valid_topic_ids)].tolist()
        if invalid_topic_ids:
            raise ValueError(f"Unknown topic_ids: {invalid_topic_ids}")
        
        # Prepare state tracking
        old_assignments: Dict[int, int] = {}
        old_strengths: Dict[int, float] = {}
        old_validated: Dict[int, bool] = {}
        
        new_assignments: Dict[int, int] = {}
        new_strengths: Dict[int, float] = {}
        new_validated: Dict[int, bool] = {}
        
        # Apply reassignments
        for _, row in df.iterrows():
            doc_id = int(row["doc_id"])
            new_topic_id = int(row["reassigned_topic_id"])
            
            # Skip if reassigned_topic_id is NaN/None
            if pd.isna(new_topic_id):
                continue
            
            pos = self._pos(doc_id)
            
            # Store old values
            old_assignments[doc_id] = int(self._assignments[pos])
            old_strengths[doc_id] = float(self._strengths[pos])
            old_validated[doc_id] = bool(self._validated[pos])
            
            # Apply new assignment
            self._assignments[pos] = new_topic_id
            new_assignments[doc_id] = new_topic_id
            
            # Keep existing strength or set to 1.0 if not available
            strength = 1.0
            self._strengths[pos] = strength
            new_strengths[doc_id] = strength
            
            # Mark as validated if requested
            if validate_imports:
                self._validated[pos] = True
                new_validated[doc_id] = True
        
        # Invalidate representations for affected topics
        affected_topics = set(new_assignments.values()) | set(old_assignments.values())
        for tid in affected_topics:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()
        
        # Build forward/backward states
        forward = {"assignments": new_assignments, "strengths": new_strengths}
        backward = {"assignments": old_assignments, "strengths": old_strengths}
        
        if validate_imports and new_validated:
            forward["validated"] = new_validated
            backward["validated"] = old_validated
        
        description = f"Imported assignments for {len(new_assignments)} documents"
        
        return forward, backward, description
