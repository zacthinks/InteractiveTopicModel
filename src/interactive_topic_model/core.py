"""Core classes for Interactive Topic Model."""

from typing import Dict, List, Optional, Tuple, Callable, Any, Type, Union, Sequence, Iterable
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
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
        self.vectorizer = vectorizer
        
        self.n_representative_docs = n_representative_docs
        
        # Document-level caches (immutable once computed)
        # Maps doc_id to position in embedding matrices
        self._doc_id_to_pos: Dict[int, int] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._reduced_embeddings: Optional[np.ndarray] = None
        
        # Vocabulary (if this BTM has its own vectorizer)
        self._vocabulary: Optional[List[str]] = None
    
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
        
        # Generate embeddings
        self._embeddings = self.embedder.encode(texts)
        
        # Reduce dimensionality
        self._reduced_embeddings = self.reducer.fit_transform(self._embeddings)
        
        # Cluster
        self.clusterer.fit(self._reduced_embeddings)
        labels = self.clusterer.labels_
        strengths = self.clusterer.strengths_
        
        # Store vocabulary if using custom vectorizer
        if self.vectorizer is not self.itm._default_vectorizer:
            self._vocabulary = self.vectorizer.get_feature_names_out().tolist()
        
        return {"labels": labels, "strengths": strengths}

    def get_embeddings(self, doc_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for documents, computing and caching if needed.

        Args:
            doc_ids: Iterable of document IDs. Duplicates allowed and order is preserved.

        Returns:
            Tuple (embeddings, reduced_embeddings) where each is an ndarray with rows in same
            order as doc_ids. If `doc_ids` is empty, returns two arrays with shape (0, D) and (0, d).
        """
        # Fast path: empty input
        doc_ids = list(doc_ids)
        if not doc_ids:
            # Return empty arrays consistent with existing caches if present,
            # otherwise (0, 0) shaped arrays to avoid downstream exceptions.
            if getattr(self, "_embeddings", None) is not None:
                emb_dim = self._embeddings.shape[1]
                red_dim = self._reduced_embeddings.shape[1]
                return np.zeros((0, emb_dim)), np.zeros((0, red_dim))
            else:
                return np.zeros((0, 0)), np.zeros((0, 0))
        elif len(doc_ids) != len(set(doc_ids)):
            raise ValueError("get_embeddings() does not allow duplicate doc_ids")

        # Ensure fitted / caches present
        if getattr(self, "_embeddings", None) is None or getattr(self, "_reduced_embeddings", None) is None:
            raise RuntimeError("BTM not fitted yet")
        
        # Get caches
        d = self._doc_id_to_pos
        all_pos = []
        uncached_ids = []
        base = len(d)
        uncached_pos = []

        for doc_id in doc_ids:
            pos = d.get(doc_id)
            if pos is None:
                pos = base + len(uncached_pos)
                uncached_ids.append(doc_id)
                uncached_pos.append(pos)
            all_pos.append(pos)

        # Compute embeddings for uncached IDs (if any)
        if uncached_ids:
            # Fetch texts
            texts = [self.itm._texts[self.itm._pos(doc_id)] for doc_id in uncached_ids]

            # Compute embeddings and reduced embeddings
            new_embeddings = self.embedder.encode(texts)
            new_reduced = self.reducer.transform(new_embeddings)

            # Ensure 2D numpy arrays
            new_embeddings = np.asarray(new_embeddings)
            if new_embeddings.ndim == 1:
                new_embeddings = new_embeddings.reshape(1, -1)
            new_reduced = np.asarray(new_reduced)
            if new_reduced.ndim == 1:
                new_reduced = new_reduced.reshape(1, -1)

            # Append to caches efficiently
            try:
                self._embeddings = np.concatenate([self._embeddings, new_embeddings], axis=0)
                self._reduced_embeddings = np.concatenate([self._reduced_embeddings, new_reduced], axis=0)
            except ValueError:
                # Fallback to vstack if shapes peculiar
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
                self._reduced_embeddings = np.vstack([self._reduced_embeddings, new_reduced])

            # Update mapping and optional reverse list in one step
            for doc_id, pos in zip(uncached_ids, uncached_pos):
                d[doc_id] = pos

        # Index into cache arrays (numpy advanced indexing preserves order and duplicates)
        embeddings = self._embeddings[all_pos]
        reduced_embeddings = self._reduced_embeddings[all_pos]

        return embeddings, reduced_embeddings
    
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
        random_state: Optional[int] = None,
    ) -> List[str]:
        """
        Get example documents from this topic.
        
        Args:
            n: Number of examples to return.
            include_descendants: Whether to include descendant topics.
            random_state: Random seed for sampling.
            
        Returns:
            List of example document texts.
        """
        texts = self.get_texts(include_descendants=include_descendants)
        
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
        top_terms = self.get_top_terms(n=n)
        if not top_terms:
            return f"Topic {self.topic_id}"
        terms = [term for term, _ in top_terms[:n]]
        return f"{self.topic_id}_{'_'.join(terms)}"
    
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
        
        # Determine which topics to include in c-TF-IDF calculation
        sibling_topics = [t for t in self.itm.topics.values()
                          if t.parent == parent and t.topic_id >= 0]
        
        # Collect all doc positions and labels for siblings
        all_doc_positions = []
        all_labels = []

        for topic in sibling_topics:
            doc_ids = topic.get_doc_ids(include_descendants=True)
            all_doc_positions.extend(self.itm._pos(d) for d in doc_ids)
            all_labels.extend([topic.topic_id] * len(doc_ids))
        
        if all_doc_positions:
            # Get DTM for all siblings
            parent_dtm = self.itm._dtm[all_doc_positions]
            labels_array = np.array(all_labels, dtype=np.int32)
            
            # Apply ClassTfidfTransformer
            ctfidf_transformer = ClassTfidfTransformer(reduce_frequent_words=False)
            ctfidf_matrix, label_to_index = ctfidf_transformer.fit_transform(parent_dtm, labels_array)
            
            # Get this topic's row in the c-TF-IDF matrix
            if self.topic_id in label_to_index:
                topic_idx = label_to_index[self.topic_id]
                self._cached_ctfidf = ctfidf_matrix[topic_idx].toarray().ravel()
            else:
                # Fallback if topic not in labels (shouldn't happen)
                self._cached_ctfidf = np.zeros(self.itm._dtm.shape[1])
        else:
            self._cached_ctfidf = np.zeros(len(btm.get_vocabulary()))
        
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
        embedder: Optional[Union[str, Embedder]] = None,
        reducer: Optional[Reducer] = None,
        clusterer: Optional[Clusterer] = None,
        min_topic_size: int = 2,
        keep_outliers: bool = True,
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
            split_btm = BasicTopicModel(
                itm=self.itm,
                embedder=embedder,
                reducer=reducer,          # default reducer inside BTM if None
                clusterer=clusterer,      # use caller/default clusterer
                scorer=self.itm.btm.scorer,
                vectorizer=self.itm._default_vectorizer,
                n_representative_docs=self.itm.btm.n_representative_docs,
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
            if label == -1:
                # Outliers
                if keep_outliers:
                    label_to_new_id[-1] = self.topic_id  # Stay in parent
                else:
                    label_to_new_id[-1] = InteractiveTopicModel.OUTLIER_ID
            else:
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
    def commit_split(self) -> Tuple[Dict, Dict, str]:
        """
        Commit the previewed split.
        
        Returns:
            Tuple of (forward_state, backward_state, description) for undo/redo.
        """
        if self._split_preview is None:
            raise ITMError("No split preview to commit. Call preview_split() first.")
        
        preview = self._split_preview

        old_btm = self.btm
        new_btm = getattr(preview, "btm", None)
        
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
                self.itm._ensure_topic(new_real, parent=self)
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

        # Deactivate old topic
        self.active = False
        
        # Prepare undo/redo state
        forward = {
            "assignments": final_assignments.copy(),
            "strengths": preview.proposed_strengths.copy(),
            "new_topics": real_new_topic_ids.copy(),
            "deactivate_topics": [self.topic_id],
        }

        backward = {
            "assignments": old_assignments,
            "strengths": old_strengths,
            "remove_topics": real_new_topic_ids.copy(),
            "reactivate_topics": [self.topic_id],
        }

        if new_btm is not None or old_btm is not None:
            forward["topic_btms"] = {self.topic_id: new_btm}
            backward["topic_btms"] = {self.topic_id: old_btm}
        
        description = f"Split topic {self.topic_id} into {len(real_new_topic_ids)} new topics"
        
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

        # Create root BTM (semantic space)
        self.btm = BasicTopicModel(
            itm=self,
            embedder=embedder,
            reducer=reducer,
            clusterer=clusterer,
            scorer=scorer,
            vectorizer=self._default_vectorizer,
            n_representative_docs=n_representative_docs,
        )
        
        self._fitted = False
        
        # Master DTM and vocabulary (computed once during fit)
        self._dtm: Optional[sparse.csr_matrix] = None
        self._vocabulary: Optional[List[str]] = None
        
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
        self, doc_id: int, topic_id: int, strength: float = 1.0, validated: bool = False
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
        mode: Optional[Union[str, Callable]] = None,
        threshold: float = 0.0,
    ) -> Tuple[Optional[int], float]:
        """
        Suggest best topic for a document.
        
        Args:
            doc_id: Document ID.
            mode: Scoring mode ('embedding', 'tfidf', 'harmonic', or callable).
            threshold: Minimum score to return suggestion.
            
        Returns:
            Tuple of (suggested_topic_id, score) or (None, 0.0) if below threshold.
        """
        self._require_fitted()
        
        # Get active topics
        active_topics = [t for t in self.topics.values() if t.active and t.topic_id >= 0]
        
        if not active_topics:
            return None, 0.0
        
        # Get document position
        pos = self._pos(doc_id)
        
        # Get document representations
        doc_embedding = self.btm._embeddings[pos] if self.btm._embeddings is not None else None
        
        # Get document TF-IDF (would need to vectorize single doc, simplified for now)
        doc_tfidf = None
        
        # Determine scorer
        if mode is None or mode == "embedding":
            scorer = embedding_scorer
        elif mode == "tfidf":
            scorer = tfidf_scorer
        elif mode == "harmonic":
            scorer = harmonic_scorer
        elif callable(mode):
            scorer = mode
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")
        
        # Collect topic representations
        topic_ids = []
        topic_embeddings_list = []
        topic_ctfidfs_list = []
        
        for topic in active_topics:
            emb = topic.get_embedding()
            ctfidf = topic.get_ctfidf()
            
            topic_ids.append(topic.topic_id)
            topic_embeddings_list.append(emb)
            topic_ctfidfs_list.append(ctfidf)
        
        if not topic_ids:
            return None, 0.0
        
        topic_embeddings = np.array(topic_embeddings_list)
        topic_ctfidfs = np.array(topic_ctfidfs_list) if topic_ctfidfs_list else None
        
        # Score
        try:
            scores = scorer(doc_embedding, doc_tfidf, topic_embeddings, topic_ctfidfs)
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            if best_score >= threshold:
                return topic_ids[best_idx], float(best_score)
            else:
                return None, 0.0
        except Exception as e:
            print(f"Error in scoring: {e}")
            return None, 0.0
    
    # ----------------------------
    # Topic operations
    # ----------------------------
    
    @undoable
    def merge_topics(
        self, topic_ids: List[int], new_label: Optional[str] = None
    ) -> Tuple[Dict, Dict, str]:
        """
        Merge multiple topics into one.
        
        Args:
            topic_ids: List of topic IDs to merge.
            new_label: Optional label for merged topic.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        self._require_fitted()
        
        if len(topic_ids) < 2:
            raise ValueError("Must merge at least 2 topics")
        
        # Create new merged topic
        merged_id = self._new_topic_id()
        self._ensure_topic(merged_id, label=new_label, parent=self)
        
        # Collect old assignments
        old_assignments = {}
        old_strengths = {}
        
        for topic_id in topic_ids:
            if topic_id not in self.topics:
                continue
            
            topic = self.topics[topic_id]
            doc_ids = topic.get_doc_ids(include_descendants=False)
            
            for doc_id in doc_ids:
                pos = self._pos(doc_id)
                old_assignments[doc_id] = int(self._assignments[pos])
                old_strengths[doc_id] = float(self._strengths[pos])
                
                # Reassign to merged topic
                self._assignments[pos] = merged_id
                self._strengths[pos] = 1.0  # Reset strength
            
            # Deactivate old topic
            topic.active = False
        
        # Invalidate representations for affected topics
        for topic_id in topic_ids:
            if topic_id in self.topics:
                self.topics[topic_id].invalidate_representations()
        self.topics[merged_id].invalidate_representations()
        
        # Prepare undo/redo state
        forward = {
            "assignments": {doc_id: merged_id for doc_id in old_assignments.keys()},
            "strengths": {doc_id: 1.0 for doc_id in old_assignments.keys()},
            "new_topic": merged_id,
            "deactivate_topics": topic_ids,
        }
        
        backward = {
            "assignments": old_assignments,
            "strengths": old_strengths,
            "remove_topic": merged_id,
            "reactivate_topics": topic_ids,
        }
        
        description = f"Merged {len(topic_ids)} topics into topic {merged_id}"
        
        return forward, backward, description

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
    
    def get_topic_info(self, top_words_n: int = 10, show_inactive: bool = False, show_outliers: bool = True) -> pd.DataFrame:
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
            if (not topic.active and not show_inactive) or (topic_id < 0 and not show_outliers):
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
            df = df.sort_values("count", ascending=False)
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

        df = pd.DataFrame(rows)

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
    
    @undoable
    def merge_topics_by_label(
        self, exclude_inactive: bool = True, into: str = "first"
    ) -> Tuple[Dict, Dict, str]:
        """
        Merge topics that share the same label.
        
        Args:
            exclude_inactive: Skip inactive topics.
            into: "first" keeps the first topic ID, "lowest" keeps the lowest ID.
            
        Returns:
            Tuple of (forward_state, backward_state, description).
        """
        # Group topics by label
        label_to_tids: Dict[str, List[int]] = {}
        for tid, topic in self.topics.items():
            if tid == self.OUTLIER_ID:
                continue
            if exclude_inactive and not topic.active:
                continue
            if not topic._label:
                continue
            label_to_tids.setdefault(topic._label, []).append(tid)
        
        # Collect all merges
        all_old_assignments = {}
        all_old_strengths = {}
        merged_count = 0
        
        with disable_tracking(self):
            for label, tids in label_to_tids.items():
                if len(tids) < 2:
                    continue
                
                # Choose which topic to keep
                if into == "first":
                    keep_id = tids[0]
                elif into == "lowest":
                    keep_id = min(tids)
                else:
                    raise ValueError(f"Unknown 'into' mode: {into}")
                
                merge_ids = [t for t in tids if t != keep_id]
                
                # Collect old assignments
                for merge_id in merge_ids:
                    topic = self.topics[merge_id]
                    doc_ids = topic.get_doc_ids(include_descendants=False)
                    
                    for doc_id in doc_ids:
                        pos = self._pos(doc_id)
                        all_old_assignments[doc_id] = int(self._assignments[pos])
                        all_old_strengths[doc_id] = float(self._strengths[pos])
                        
                        # Reassign to keep_id
                        self._assignments[pos] = keep_id
                        self._strengths[pos] = 1.0
                    
                    # Deactivate merged topic
                    topic.active = False
                
                merged_count += len(merge_ids)
        
        if merged_count == 0:
            # No merges performed - return empty state
            return {}, {}, "No topics with duplicate labels found"
        
        # Invalidate affected topics
        affected = set()
        for doc_id in all_old_assignments.keys():
            pos = self._pos(doc_id)
            affected.add(int(self._assignments[pos]))
            affected.add(all_old_assignments[doc_id])
        
        for tid in affected:
            if tid in self.topics:
                self.topics[tid].invalidate_representations()
        
        forward = {
            "assignments": {doc_id: int(self._assignments[self._pos(doc_id)]) 
                          for doc_id in all_old_assignments.keys()},
            "strengths": {doc_id: 1.0 for doc_id in all_old_assignments.keys()},
        }
        backward = {
            "assignments": all_old_assignments,
            "strengths": all_old_strengths,
        }
        description = f"Merged {merged_count} topics by label"
        
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
    
    def visualize_hierarchy(
        self,
        *,
        similarity: str = "harmonic",
        include_inactive: bool = False,
        linkagefun: Optional[Callable] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize topic hierarchy as dendrogram.
        
        Args:
            similarity: Similarity metric ('embedding', 'tfidf', or 'harmonic').
            include_inactive: Whether to include inactive topics.
            linkagefun: Optional custom linkage function.
            save_path: Optional path to save figure.
            
        Returns:
            Plotly figure.
        """
        from .visualization import visualize_topic_hierarchy as _visualize_hierarchy
        return _visualize_hierarchy(
            self,
            similarity=similarity,
            include_inactive=include_inactive,
            linkagefun=linkagefun,
            save_path=save_path,
        )
