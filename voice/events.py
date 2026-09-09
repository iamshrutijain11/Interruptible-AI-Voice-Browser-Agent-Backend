from enum import Enum
from typing import Optional, Dict, Any

class VoiceEventType(str, Enum):
    VOICE_STARTED = "voice.started"
    VOICE_STOPPED = "voice.stopped"
    TRANSCRIPTION_COMPLETED = "transcription.completed"
    SPEECH_STARTED = "speech.started"
    SPEECH_STOPPED = "speech.stopped"
    SPEECH_INTERRUPTED = "speech.interrupted"
    VOICE_ERROR = "voice.error"
    STATE_CHANGED = "voice.state_changed"

class VoiceEvent:
    def __init__(self, event: VoiceEventType, state: str, task_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.event = event
        self.state = state
        self.task_id = task_id
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.value if isinstance(self.event, VoiceEventType) else str(self.event),
            "state": self.state,
            "task_id": self.task_id,
            "data": self.data
        }

def create_event(event_type: VoiceEventType, state: str, task_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Helper to generate standardized event dictionary payloads for WebSockets or JSON APIs."""
    return VoiceEvent(
        event=event_type,
        state=state,
        task_id=task_id,
        data=kwargs
    ).to_dict()
