import os
import io
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy model loading for Whisper
_whisper_model = None

class STTService:
    """
    Speech-to-Text service utilizing Whisper or API fallback.
    Exposes transcribe(audio_bytes) -> clean text string.
    """
    def __init__(self, engine: Optional[str] = None):
        self.engine = engine or os.getenv("STT_ENGINE", "whisper_local")

    def _get_whisper_model(self):
        global _whisper_model
        if _whisper_model is None:
            try:
                import whisper
                logger.info("Loading local Whisper model ('base')...")
                _whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully.")
            except ImportError:
                logger.warning("`openai-whisper` package not installed. Falling back to mock engine.")
                _whisper_model = False
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                _whisper_model = False
        return _whisper_model

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribes raw audio bytes into clean plain text.

        :param audio_bytes: Raw binary audio payload (WebM, WAV, MP3, OGG, etc.)
        :param mime_type: Audio MIME type hint
        :return: Transcribed text string
        """
        if not audio_bytes or len(audio_bytes) == 0:
            logger.warning("STT received empty audio bytes.")
            return ""

        # Check for mock engine explicitly or if local Whisper model fails
        if self.engine == "mock":
            return "[Mock Transcription] Open hackathon main page and search for laptop deals"

        if self.engine in ("gemini_api", "openai_api"):
            return self._transcribe_gemini_api(audio_bytes, mime_type)

        # Default local Whisper transcription
        return self._transcribe_local_whisper(audio_bytes, mime_type)

    def _transcribe_local_whisper(self, audio_bytes: bytes, mime_type: str) -> str:
        model = self._get_whisper_model()
        if not model:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_gemini_api_key_here":
                return self._transcribe_gemini_api(audio_bytes, mime_type)
            logger.info("Local Whisper model not loaded; returning clean audio notification.")
            return ""

        # Determine file extension based on mime_type
        ext = ".webm"
        if "wav" in mime_type:
            ext = ".wav"
        elif "mp3" in mime_type or "mpeg" in mime_type:
            ext = ".mp3"
        elif "ogg" in mime_type:
            ext = ".ogg"

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_file_path = temp_audio.name

            logger.info(f"Transcribing audio file ({len(audio_bytes)} bytes) with Whisper...")
            result = model.transcribe(temp_file_path, fp16=False)
            text = result.get("text", "").strip()
            logger.info(f"Whisper Transcription result: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            return ""
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    def _transcribe_gemini_api(self, audio_bytes: bytes, mime_type: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY missing for gemini_api STT engine, falling back to local whisper.")
            return self._transcribe_local_whisper(audio_bytes, mime_type)
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            model_name = os.getenv("LLM_MODEL")
            if not model_name or "2.5" in model_name:
                model_name = "gemini-3.6-flash"
            clean_mime = mime_type.split(";")[0].strip() if mime_type else "audio/webm"

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
                    "Transcribe this audio clip accurately. Output only the verbatim transcribed text, nothing else.",
                ],
            )
            text = response.text.strip() if response.text else ""
            logger.info(f"Gemini STT result: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Gemini STT API error ({e}), falling back to local Whisper...")
            return self._transcribe_local_whisper(audio_bytes, mime_type)


# Default singleton instance
stt_service = STTService()

def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Convenience helper function for transcription."""
    return stt_service.transcribe(audio_bytes, mime_type)
