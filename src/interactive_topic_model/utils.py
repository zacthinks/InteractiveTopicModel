"""Utility functions and classes for Interactive Topic Model."""

from typing import Tuple, Union, Optional, Iterable
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class ClassTfidfTransformer:
    """
    Class-based TF-IDF transformer adapted from BERTopic.
    
    Converts per-document term counts into per-class aggregated TF-IDF vectors.
    Used to find representative terms and documents for each topic.
    """
    
    def __init__(self, reduce_frequent_words: bool = False):
        """
        Initialize c-TF-IDF transformer.
        
        Args:
            reduce_frequent_words: Whether to reduce impact of frequent words.
        """
        self.reduce_frequent_words = reduce_frequent_words
        self.idf_ = None
    
    def fit_transform(
        self, X: sparse.csr_matrix, labels: np.ndarray
    ) -> Tuple[sparse.csr_matrix, dict]:
        """
        Transform document-term matrix to class-term c-TF-IDF matrix.
        
        Args:
            X: Document-term matrix (n_docs, n_terms) - sparse.
            labels: Class labels for each document (n_docs,).
        
        Returns:
            Class-term matrix (n_classes, n_terms) with c-TF-IDF values and a mapper from the passed labels to row indices.
        """
        # Aggregate term counts per class
        unique_labels = sorted(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        n_classes = len(unique_labels)
        n_terms = X.shape[1]
        
        class_term_matrix = np.zeros((n_classes, n_terms))
        
        for label, idx in label_to_index.items():
            mask = labels == label
            class_term_matrix[idx] = X[mask].sum(axis=0).A1
        
        # Calculate class-based TF (term frequency within class)
        # TF: frequency of term in class / total terms in class
        class_sizes = class_term_matrix.sum(axis=1, keepdims=True)
        class_sizes[class_sizes == 0] = 1  # Avoid division by zero
        tf = class_term_matrix / class_sizes
        
        # Calculate IDF (inverse document frequency across classes)
        # IDF: log(total classes / classes containing term)
        n_containing = np.count_nonzero(class_term_matrix, axis=0)
        n_containing[n_containing == 0] = 1  # Avoid division by zero
        idf = np.log(n_classes / n_containing) + 1
        
        if self.reduce_frequent_words:
            # Reduce impact of words appearing in many classes
            idf = idf * (1.0 - (n_containing / n_classes))
        
        self.idf_ = idf
        
        # c-TF-IDF = TF * IDF
        ctfidf = tf * idf
        
        return sparse.csr_matrix(ctfidf), label_to_index


def default_vectorizer() -> CountVectorizer:
    """
    Default CountVectorizer for topic modeling / c-TF-IDF.
    Balanced and safe for general use.
    
    Returns:
        Configured CountVectorizer instance.
    """
    return CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words='english',
    )


def custom_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.9,
    use_en_stopwords: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    keepwords: Optional[Iterable[str]] = None,
    token_regex: Optional[str] = None,
) -> CountVectorizer:
    """
    Create a custom CountVectorizer with flexible configuration.
    
    Args:
        ngram_range: Range of n-gram sizes to extract.
        min_df: Minimum document frequency for terms.
        max_df: Maximum document frequency for terms.
        use_en_stopwords: Whether to use English stop words.
        stopwords: Additional stop words to exclude.
        keepwords: Words to keep even if they're stop words.
        token_regex: Custom tokenization regex pattern.
        
    Returns:
        Configured CountVectorizer instance.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    stop_words = set(ENGLISH_STOP_WORDS) if use_en_stopwords else set()
    
    if stopwords:
        stop_words.update(stopwords)
    if keepwords:
        stop_words.difference_update(keepwords)
    
    kwargs = dict(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words if stop_words else None,
    )
    
    if token_regex is not None:
        kwargs["token_pattern"] = token_regex
    
    return CountVectorizer(**kwargs)
