import os
import io
import wave
import math
import struct
import logging
import requests
from typing import Optional, Generator, Dict, Any
from .audio import voice_manager

logger = logging.getLogger(__name__)

class RimeTTSService:
    """
    Rime Text-to-Speech API Wrapper.
    All Rime API payload structures, headers, endpoints, and fallback controls are isolated here.
    """
    def __init__(self):
        self.api_key = os.getenv("RIME_API_KEY", "")
        self.speaker = os.getenv("RIME_SPEAKER", "marsh")
        self.model_id = os.getenv("RIME_MODEL_ID", "v1")
        self.api_url = os.getenv("RIME_API_URL", "https://users.rime.ai/v1/rime-tts")
        self.audio_format = os.getenv("RIME_AUDIO_FORMAT", "mp3") # 'mp3' or 'wav'

    def speak(self, text: str, task_id: str) -> Optional[bytes]:
        """
        Generate audio bytes for `text` associated with `task_id`.
        Validates if `task_id` is active before & during operation.

        :param text: Text string for TTS conversion
        :param task_id: Unique task identifier
        :return: Audio bytes (MP3/WAV) or None if interrupted/failed
        """
        if not text or not text.strip():
            logger.warning("Rime TTS received empty text payload.")
            return None

        # Check if task is valid before executing
        if not voice_manager.is_task_valid(task_id):
            logger.info(f"Task '{task_id}' was invalidated before Rime TTS started. Aborting.")
            return None

        voice_manager.start_speaking(task_id)

        # If Rime API key is missing or mock mode active, use mock audio generator
        if not self.api_key or self.api_key == "your_rime_api_key_here":
            logger.info(f"[Rime TTS] No API key detected. Generating mock audio buffer for task '{task_id}'.")
            return self._generate_mock_audio(text, task_id)

        try:
            audio_bytes = self._call_rime_api(text, task_id)
            
            # Re-check task validity after network call
            if not voice_manager.is_task_valid(task_id):
                logger.info(f"Task '{task_id}' was interrupted during Rime API generation. Discarding payload.")
                return None

            return audio_bytes

        except Exception as e:
            logger.error(f"[Rime TTS Error] Failed to generate speech via Rime API: {e}")
            voice_manager.set_error(f"Rime API Error: {str(e)}")
            return self._generate_mock_audio(text, task_id)

    def stop_speaking(self, task_id: Optional[str] = None) -> None:
        """
        Stop speaking and invalidate current task.
        """
        logger.info(f"[Rime TTS] Explicit stop requested for task_id: {task_id}")
        voice_manager.stop_speaking(task_id=task_id, reason="Explicit Rime stop_speaking call")

    def _call_rime_api(self, text: str, task_id: str) -> bytes:
        """
        Performs HTTP POST request to Rime API.
        
        Isolates Rime API endpoint parameters for easy custom updates.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": f"audio/{self.audio_format}"
        }

        # Customizable Rime API Payload Schema
        payload: Dict[str, Any] = {
            "speaker": self.speaker,
            "text": text,
            "modelId": self.model_id,
            "samplingRate": 22050,
            "speedAlpha": 1.0,
            "audioFormat": self.audio_format
        }

        logger.info(f"Sending request to Rime API at {self.api_url} for task '{task_id}'...")
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(f"Rime API returned HTTP status {response.status_code}: {response.text}")

        return response.content

    def _generate_mock_audio(self, text: str, task_id: str) -> bytes:
        """
        Generates a lightweight valid WAV audio byte buffer for testing without live API keys.
        Duration proportional to text length (~1-3 seconds).
        """
        sample_rate = 16000
        words_count = max(1, len(text.split()))
        duration = min(5.0, max(1.0, words_count * 0.25)) # duration in seconds
        total_samples = int(sample_rate * duration)

        num_channels = 1
        sampwidth = 2 # 16-bit PCM

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)

            # Generate pleasant soft sine audio tone sequence
            freq = 440.0 # A4 tone
            data = bytearray()
            for i in range(total_samples):
                # Fade out tone towards the end
                envelope = max(0.0, 1.0 - (i / total_samples))
                t = i / sample_rate
                # Modulate slightly to simulate speech-like envelope
                mod = math.sin(2 * math.pi * 5 * t)
                sample = int(16000 * envelope * (0.8 * math.sin(2 * math.pi * freq * t) + 0.2 * mod))
                data.extend(struct.pack('<h', max(-32768, min(32767, sample))))
            
            wav_file.writeframes(data)

        return buf.getvalue()


# Default singleton instance
rime_service = RimeTTSService()
