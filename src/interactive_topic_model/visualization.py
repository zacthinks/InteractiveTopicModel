"""Visualization utilities for Interactive Topic Model."""

from typing import Optional, List, Callable, Union
import textwrap
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

from interactive_topic_model.components import SentenceTransformerEmbedder
from interactive_topic_model.core import BasicTopicModel, InteractiveTopicModel
from interactive_topic_model.protocols import Embedder

try:
    import plotly.figure_factory as ff
except ImportError:
    ff = None

def _wrap_text(s: str, width: int = 100) -> str:
    if not s:
        return ""
    return "<br>".join(textwrap.wrap(str(s), width=width))

def _truncate(s: str, max_len: int = 400) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."

def _get_doc_ids_and_plot_btm(
    itm: "InteractiveTopicModel",
    *,
    topic_id: Optional[int],
    include_descendants: bool,
):
    if topic_id is None:
        doc_ids = list(itm._doc_ids)
        plot_btm = itm.btm
    else:
        topic = itm.topics[topic_id]
        doc_ids = list(topic.get_doc_ids(include_descendants=include_descendants))
        plot_btm = topic.semantic_space
    return doc_ids, plot_btm

def _compute_2d_coords(
    embeddings_full: np.ndarray,
    embeddings_reduced: Optional[np.ndarray],
    *,
    use_reduced: bool = True,
    reducer: Optional[any],
):
    # exactly your existing behavior in visualize_documents_with_search
    to_reduce = embeddings_full
    if use_reduced and embeddings_reduced is not None and embeddings_reduced.shape[1] >= 2:
        if embeddings_reduced.shape[1] == 2:
            return embeddings_reduced
        else:
            to_reduce = embeddings_reduced

    if reducer is None:
        from .components import UMAPReducer
        reducer = UMAPReducer(n_components=2)
    return reducer.fit_transform(to_reduce)

def _build_doc_scatter_df(
    itm: InteractiveTopicModel,
    plot_btm: BasicTopicModel,
    doc_ids: List[int],
    use_reduced: bool = True,
    reducer: Optional[any] = None,
    custom_topic_labels: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, "BasicTopicModel", list, np.ndarray]:
    """
    Returns:
      df_plot with columns: x,y,doc_id,topic,hover_text
      plot_btm used for the layout space
      texts list aligned to df rows
    """

    if len(doc_ids) == 0:
        raise ValueError("No documents to visualize.")

    emb_full, emb_reduced = plot_btm.get_embeddings(doc_ids) 
    coords_2d = _compute_2d_coords(emb_full, emb_reduced, use_reduced=use_reduced, reducer=reducer)

    # per-doc topic assignment (global assignments live on ITM)
    topics = np.array(custom_topic_labels or [str(itm._assignments[itm._pos(d)]) for d in doc_ids])

    texts = [itm._texts[itm._pos(doc_id)] for doc_id in doc_ids]
    hover_text = [_wrap_text(_truncate(t)) for t in texts]

    df_plot = pd.DataFrame(
        {
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "doc_id": doc_ids,
            "topic": topics,
            "hover_text": hover_text,
        }
    )
    return df_plot, plot_btm, texts, emb_full

def _scatter_by_topic(
    df_plot: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    topic_order = sorted(
        df_plot["topic"].unique(),
        key=lambda x: (0, int(x)) if str(x).lstrip("-").isdigit() else (1, str(x))
    )

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="topic",
        custom_data=["hover_text", "doc_id"],
        title=title,
        category_orders={"topic": topic_order},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>doc_id:</b> %{customdata[1]}<br>"
            "<b>text:</b> %{customdata[0]}<extra></extra>"
        ),
        marker=dict(size=7, opacity=0.8),
    )
    fig.update_layout(legend_title_text="Topic ID", width=900, height=700)
    return fig

def _scatter_by_similarity(
    df_plot: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="similarity",
        custom_data=["hover_text", "doc_id", "topic", "similarity"],
        title=title,
        labels={"similarity": "Cosine similarity"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>doc_id:</b> %{customdata[1]}<br>"
            "<b>topic:</b> %{customdata[2]}<br>"
            "<b>similarity:</b> %{customdata[3]:.3f}<br>"
            "<b>text:</b> %{customdata[0]}<extra></extra>"
        ),
        marker=dict(size=7, opacity=0.8),
    )
    fig.update_layout(width=900, height=700)
    return fig

def visualize_documents(
    itm: "InteractiveTopicModel",
    plot_btm: BasicTopicModel,
    doc_ids: List[int],
    custom_topic_labels: Optional[List[str]] = None,
    use_reduced: bool = True,
    title: Optional[str] = None,
    reducer: Optional[any] = None,
    save_path: Optional[str] = None,
) -> go.Figure:
    df_plot, _plot_btm, _texts, _emb_full = _build_doc_scatter_df(itm, 
                                                                  plot_btm=plot_btm, 
                                                                  doc_ids=doc_ids,
                                                                  use_reduced=use_reduced,
                                                                  reducer=reducer, 
                                                                  custom_topic_labels=custom_topic_labels)
    fig = _scatter_by_topic(df_plot, title=title or "Document Scatter Plot Colored by Topic")
    if save_path:
        fig.write_html(save_path)
    return fig

def visualize_documents_with_search(
    itm: "InteractiveTopicModel",
    plot_btm: BasicTopicModel,
    doc_ids: List[int],
    search_string: str,
    use_reduced: bool = True,
    title: Optional[str] = None,
    reducer: Optional[any] = None,
    search_embedder: Optional[Union[str, Embedder]] = None,
    save_path: Optional[str] = None,
) -> go.Figure:
    df_plot, plot_btm, texts, emb_full = _build_doc_scatter_df(
        itm,
        plot_btm=plot_btm,
        doc_ids=doc_ids,
        use_reduced=use_reduced,
        reducer=reducer,
    )

    # similarity coloring
    if search_embedder is None:
        q_emb = plot_btm.embedder.encode([search_string])[0]
        sims = cosine_similarity([q_emb], emb_full).ravel()
    else:
        emb = search_embedder
        if isinstance(search_embedder, str):
            emb = SentenceTransformerEmbedder(model_name=search_embedder)
        doc_embs_new = emb.encode(texts)
        q_emb = emb.encode([search_string])[0]
        sims = cosine_similarity([q_emb], doc_embs_new).ravel()

    df_plot = df_plot.copy()
    df_plot["similarity"] = sims

    fig = _scatter_by_similarity(df_plot, title=title or f"Documents colored by similarity to: {search_string}")
    if save_path:
        fig.write_html(save_path)
    return fig


def visualize_topic_hierarchy(
    *,
    linkage_matrix,
    labels: List[str],
    title: str = "Topic Hierarchy",
    orientation: str = "right",
    figsize_per_leaf: float = 0.35,
    save_path: Optional[str] = None,
):
    """
    Render a dendrogram using matplotlib.

    Parameters
    ----------
    linkage_matrix : ndarray
        SciPy linkage matrix.
    labels : list[str]
        Leaf labels.
    orientation : str
        'left', 'right', 'top', or 'bottom'.
    figsize_per_leaf : float
        Vertical size per leaf (in inches).
    """
    from scipy.cluster.hierarchy import dendrogram

    n = len(labels)
    height = max(4, figsize_per_leaf * n)

    if orientation in ("left", "right"):
        figsize = (8, height)
    else:
        figsize = (height, 6)

    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation=orientation,
        ax=ax,
    )

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig, ax