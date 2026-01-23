"""Data structures for Interactive Topic Model."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class SplitPreview:
    """Preview of a topic split operation before committing."""
    
    parent_topic_id: int
    """ID of the topic being split."""
    
    proposed_assignments: Dict[int, int]
    """Mapping from doc_id to proposed new topic_id."""
    
    proposed_strengths: Dict[int, float]
    """Mapping from doc_id to cluster membership strength."""
    
    new_topic_ids: List[int]
    """List of newly created topic IDs."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """When this preview was created."""

    btm: Any = None
    
    def to_preview_dict(self, get_text_func) -> Dict[int, List[tuple]]:
        """
        Convert preview to a dict grouped by new topic_id.
        
        Returns:
            Dict mapping topic_id to list of (doc_id, text, strength) tuples.
        """
        result = {}
        for doc_id, topic_id in self.proposed_assignments.items():
            if topic_id not in result:
                result[topic_id] = []
            text = get_text_func(doc_id)
            strength = self.proposed_strengths.get(doc_id, 0.0)
            result[topic_id].append((doc_id, text, strength))
        
        # Sort by strength within each topic
        for topic_id in result:
            result[topic_id].sort(key=lambda x: x[2], reverse=True)
        
        return result


@dataclass(frozen=True)
class Edit:
    """Record of a reversible edit operation."""
    
    operation: str
    """Name of the operation (e.g., 'assign_doc', 'split_topic', 'merge_topics')."""
    
    timestamp: datetime
    """When this edit was made."""
    
    forward_state: Dict[str, Any]
    """State changes to apply when redoing."""
    
    backward_state: Dict[str, Any]
    """State changes to apply when undoing."""
    
    description: str = ""
    """Human-readable description of the edit."""
    
    def __repr__(self) -> str:
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if self.description:
            return f"Edit({self.operation}: {self.description} @ {time_str})"
        return f"Edit({self.operation} @ {time_str})"
