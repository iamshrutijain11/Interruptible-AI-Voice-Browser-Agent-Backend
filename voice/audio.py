import asyncio
import logging
from enum import Enum
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)

class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"

class AudioManager:
    """
    Manages the global voice lifecycle, active task tracking, state transitions,
    and instantaneous interruption signals.
    """
    def __init__(self):
        self._state: VoiceState = VoiceState.IDLE
        self._active_task_id: Optional[str] = None
        self._listeners: List[Callable[[VoiceState, Optional[str], Optional[Dict[str, Any]]], None]] = []
        self._invalidated_tasks: set = set()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def active_task_id(self) -> Optional[str]:
        return self._active_task_id

    def add_listener(self, listener: Callable[[VoiceState, Optional[str], Optional[Dict[str, Any]]], None]):
        """Register a callback for state changes."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[VoiceState, Optional[str], Optional[Dict[str, Any]]], None]):
        """Unregister a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, extra_data: Optional[Dict[str, Any]] = None):
        """Notify subscribers of state change."""
        for listener in list(self._listeners):
            try:
                listener(self._state, self._active_task_id, extra_data or {})
            except Exception as e:
                logger.error(f"Error notifying voice state listener: {e}")

    def transition_to(self, new_state: VoiceState, task_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Transition voice state safely."""
        logger.info(f"[VoiceState] {self._state.value} -> {new_state.value} (task_id={task_id or self._active_task_id})")
        self._state = new_state
        if task_id is not None:
            self._active_task_id = task_id
        self._notify(extra_data)

    def is_task_valid(self, task_id: str) -> bool:
        """Check if task_id has been invalidated due to interruption."""
        if not task_id:
            return False
        if task_id in self._invalidated_tasks:
            return False
        # If currently speaking a different task_id, task_id is superseded
        if self._active_task_id and self._active_task_id != task_id and self._state == VoiceState.SPEAKING:
            return False
        return True

    def interrupt_current_task(self, new_task_id: Optional[str] = None, reason: str = "User interrupted") -> Optional[str]:
        """
        Immediately interrupt active speech and invalidate active task_id.
        Transitions state to INTERRUPTED.
        """
        interrupted_task = self._active_task_id
        if interrupted_task:
            self._invalidated_tasks.add(interrupted_task)
            logger.warning(f"[Interruption] Task '{interrupted_task}' invalidated. Reason: {reason}")
        
        self._state = VoiceState.INTERRUPTED
        self._active_task_id = new_task_id
        self._notify({
            "reason": reason,
            "interrupted_task_id": interrupted_task,
            "new_task_id": new_task_id
        })
        return interrupted_task

    def start_listening(self) -> None:
        """Start listening for user input."""
        # If currently speaking, auto-interrupt
        if self._state == VoiceState.SPEAKING:
            self.interrupt_current_task(reason="User started speaking")
        self.transition_to(VoiceState.LISTENING)

    def start_transcribing(self) -> None:
        """Indicate transcription in progress."""
        self.transition_to(VoiceState.TRANSCRIBING)

    def start_speaking(self, task_id: str) -> bool:
        """
        Start TTS playback for task_id.
        Returns False if task was invalidated before starting.
        """
        if task_id in self._invalidated_tasks:
            logger.warning(f"Attempted to speak for invalidated task_id '{task_id}'. Ignoring.")
            return False
        self.transition_to(VoiceState.SPEAKING, task_id=task_id)
        return True

    def stop_speaking(self, task_id: Optional[str] = None, reason: str = "Finished speaking") -> None:
        """Stop speaking for a given task or general stop request."""
        target_task = task_id or self._active_task_id
        if target_task:
            self._invalidated_tasks.add(target_task)
        
        if self._state == VoiceState.SPEAKING:
            self.transition_to(VoiceState.IDLE, extra_data={"reason": reason})

    def set_error(self, error_message: str) -> None:
        """Set state to ERROR with message."""
        logger.error(f"[VoiceError] {error_message}")
        self.transition_to(VoiceState.ERROR, extra_data={"error": error_message})

    def reset(self) -> None:
        """Reset state to IDLE."""
        self._state = VoiceState.IDLE
        self._active_task_id = None


# Global singleton instance
voice_manager = AudioManager()
