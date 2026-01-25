# Interactive Topic Model (ITM)

An interactive topic modeling framework designed for **social science research**, where interpretability, iterative refinement, and human-in-the-loop validation are made accessible via a user-friendly interface.

## Features

- **Single-line functions**: Perform complex operations with minimal coding skills required
- **Interactive visualizations**: Explore text data quickly and intuitively
- **Interactive topic refinement**: Split topics, merge topics, manually reassign documents
- **Hierarchical structures**: Create topic hierarchies via splitting or grouping operations with support for multiple embedding spaces
- **Semantic search**: Find documents with similar meanings without relying on them using the same words
- **Preview-then-commit workflow**: Preview topic splits before committing changes
- **Full undo/redo support**: Track and reverse all modeling operations
- **Document validation**: Mark high-confidence assignments and optionally freeze them during refitting
- **Representative documents**: Automatically identifies most representative documents for each topic
- **Rich topic metadata**: Auto-generated labels, top terms via c-TF-IDF, centroid embeddings
- **Flexible scoring**: Multiple modes for document-topic similarity (embedding, TF-IDF, harmonic mean)

## Quick Start

```python
import pandas as pd
from interactive_topic_model import InteractiveTopicModel

# Prepare your documents
texts = [
    "Machine learning is a subset of artificial intelligence...",
    "Deep neural networks have revolutionized computer vision...",
    # ... more documents
]

# Initialize model
itm = InteractiveTopicModel(texts)

# Fit initial model
itm.fit()

# View topic summary
print(itm.get_topic_info())

# Access individual topics
topic = itm.topics[0]
print(f"Topic {topic.topic_id}: {topic.label}")
print(f"Top terms: {topic.get_top_terms(n=5)}")
print(topic.get_examples(n=5))
print(topic.get_representative_docs(n=5))

# Interactive refinement
preview = topic.split()
print(preview)
topic.commit_split()

# Merge topics
itm.merge_topics([1, 2], into="new", new_label="Combined Topic")

# Reassign a document
itm.assign_doc(doc_id=42, topic_id=3, validated=True)

# Visualize
fig = visualize_documents(itm)
fig.show()

# Undo if needed
itm.undo()

# Check undo history
print(itm.get_history())
```

## Architecture

### Key Classes

#### `InteractiveTopicModel`
The main engine that orchestrates topic modeling operations:
- Manages document assignments, strengths, and validation status
- Coordinates semantic spaces via `BasicTopicModel` instances
- Provides operations: fit, assign, split, merge, group, archive
- Tracks undo/redo history

#### `InteractiveTopic`
Represents a single topic with:
- Access to documents: `get_doc_ids()`, `get_texts()`, `get_examples()`
- Representations: `get_embedding()`, `get_ctfidf()`, `get_top_terms()`
- Representative documents: `get_representative_doc_ids()`, `get_representative_docs()`
- Operations: `split()`, `commit_split()`
- Metadata: `label`, `parent`, `active` status

#### `BasicTopicModel`
Manages a semantic space:
- Embeds documents via `embedder` (e.g., sentence-transformers)
- Reduces dimensions via `reducer` (e.g., UMAP)
- Clusters via `clusterer` (e.g., HDBSCAN)
- Caches embeddings and reduced representations
- Can have its own vocabulary (for sub-splits with custom vectorizers)

### Topic Representations

Topics are characterized by multiple representations:

1. **Representative documents**: Documents with highest cluster membership strength
2. **c-TF-IDF vectors**: Class-based TF-IDF distinguishing each topic from siblings
3. **Centroid embeddings**: Mean embedding of representative documents
4. **Top terms**: Highest-weighted terms from c-TF-IDF
5. **Auto-generated labels**: Based on top terms

All representations are computed lazily and invalidated when assignments change.

## Component Customization

ITM provides default components but allows full customization:

```python
from interactive_topic_model import (
    InteractiveTopicModel,
    SentenceTransformerEmbedder,
    UMAPReducer,
    HDBSCANClusterer,
    custom_vectorizer,
    harmonic_scorer,
)

# Custom components
itm = InteractiveTopicModel(
    texts=texts,
    embedder=SentenceTransformerEmbedder("all-mpnet-base-v2"),
    reducer=UMAPReducer(n_components=5, n_neighbors=15),
    clusterer=HDBSCANClusterer(min_cluster_size=10),
    vectorizer=custom_vectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.85,
        use_en_stopwords=True,
    ),
    scorer=harmonic_scorer,
    n_representative_docs=5,
)
```

### Available Components

**Embedders**:
- `SentenceTransformerEmbedder` - sentence-transformers models
- `e5_base_embedder()` - Convenience function for E5-base-v2

**Reducers**:
- `UMAPReducer` - UMAP dimensionality reduction

**Clusterers**:
- `HDBSCANClusterer` - Hierarchical DBSCAN clustering

**Scorers**:
- `embedding_scorer` - Cosine similarity on embeddings
- `tfidf_scorer` - Cosine similarity on TF-IDF vectors
- `harmonic_scorer` - Harmonic mean of both (default)

**Vectorizers**:
- `default_vectorizer()` - Balanced defaults for general use
- `custom_vectorizer()` - Customizable n-grams, stopwords, etc.

## Workflow Patterns

Refer to [vignette](examples\dbpedia_vignette.ipynb) for more details.

### Iterative Refinement

```python
# 1. Fit initial model
itm.fit()

# 2. Explore topics
df = itm.get_topic_info()

# 3. Split heterogeneous topics
topic = itm.topics[5]
preview = topic.split()
# ... inspect preview ...
new_topic_ids = topic.commit_split()

# 4. Merge similar topics
itm.merge_topics([7, 8, 9], into="new", new_label="Sports")

# 5. Archive noise topics
itm.archive_topic(topic_id=12)

# 6. Validate good assignments
for doc_id in good_docs:
    itm.validate_doc(doc_id)
```

### Preview-Commit Pattern

```python
# Split returns a preview
topic = itm.topics[0]
preview = topic.split(
    clusterer=HDBSCANClusterer(min_cluster_size=5)
)

# Inspect before committing
print(f"Will create {len(preview.new_topic_ids)} new topics")

# Commit when satisfied
new_ids = topic.commit_split()

# Or undo if not satisfied
itm.undo()
```

### Validation and Refitting

```python
# Suggest assignments for unvalidated docs
itm.suggest_assignment(doc_id=42, mode="harmonic", threshold=0.5)

# Validate assignments
itm.validate_doc(doc_id=42)

# Auto-refit outliers
itm.refit(
    target=[itm.OUTLIER_ID],
    mode="harmonic",
    threshold=0.3,
    auto_reassign=True,
    validate_refits=True,
)
```

## Visualization

```python
from interactive_topic_model import visualize_documents, visualize_topic_hierarchy

# 2D scatter plot of documents colored by topic
fig = visualize_documents(itm, use_reduced=True)
fig.show()

# Topic hierarchy dendrogram
fig = visualize_topic_hierarchy(
    itm,
    similarity="harmonic",  # or "embedding", "tfidf"
    include_inactive=False,
)
fig.show()
```

## API Reference

### InteractiveTopicModel Methods

**Fitting**:
- `fit()` - Fit initial topic model

**Assignment Operations**:
- `assign_doc(doc_id, topic_id, strength, validated)` - Assign document to topic
- `validate_doc(doc_id)` - Mark document as validated
- `suggest_assignment(doc_id, mode, threshold)` - Suggest best topic

**Topic Operations**:
- `merge_topics(topic_ids, into, new_label, archive_sources, validate_refits)` - Merge topics
- `archive_topic(topic_id)` - Deactivate a topic
- `group_topics(topic_ids, label)` - Create hierarchical grouping

**Information Retrieval**:
- `get_topic_info(top_words_n, show_inactive, show_outliers)` - Summary DataFrame
- `get_representative_documents(topic_ids, n)` - Representative docs per topic
- `get_examples(topics, n, include_descendants, random_state)` - Sample documents
- `get_active_topics()` - List of active topic IDs
- `get_outlier_count()` - Count of outlier documents
- `get_validated_count()` - Count of validated documents

**Undo/Redo**:
- `undo(steps)` - Undo operations
- `redo(steps)` - Redo operations
- `can_undo()` - Check if undo is possible
- `can_redo()` - Check if redo is possible
- `get_history(stack, reverse, include_timestamp)` - View edit history

### InteractiveTopic Methods

**Document Access**:
- `get_count(include_descendants)` - Number of documents
- `get_doc_ids(include_descendants)` - List of document IDs
- `get_texts(include_descendants)` - List of document texts
- `get_examples(n, include_descendants, random_state)` - Sample documents
- `get_representative_doc_ids()` - IDs of representative documents
- `get_representative_docs()` - Texts of representative documents

**Representations**:
- `get_embedding()` - Centroid embedding
- `get_ctfidf()` - c-TF-IDF vector
- `get_top_terms(n)` - Top n terms with scores
- `get_auto_label(n)` - Generate label from top terms

**Operations**:
- `split(clusterer, ...)` - Preview topic split
- `commit_split()` - Commit the split preview

**Properties**:
- `label` - Topic label (get/set)
- `parent` - Parent topic or ITM
- `active` - Whether topic is active
- `semantic_space` - Associated BasicTopicModel

## To-do
- Update vignette to demonstrate more features and navigation
- Add serialization features
- Add advanced semantic features (idea training, etc.)
- Add assignment export/import features

## License

MIT License