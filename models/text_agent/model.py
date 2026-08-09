"""Text Agent model — re-exports the full TextModel implementation."""
from models.text_agent.text_model import TextModel

# Backward-compatible alias
TextAgentModel = TextModel

__all__ = ["TextModel", "TextAgentModel"]
