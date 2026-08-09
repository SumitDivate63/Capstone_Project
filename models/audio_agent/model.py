"""Audio Agent model — re-exports the full AudioModel implementation."""
from models.audio_agent.audio_model import AudioModel

# Backward-compatible alias
AudioAgentModel = AudioModel

__all__ = ["AudioModel", "AudioAgentModel"]
