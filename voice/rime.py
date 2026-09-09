"""
rime.py
-------
Rime Text-to-Speech API wrapper.

Multilingual support (verified from https://docs.rime.ai/docs/voices, September 2026):
  - Model: "coda" — Rime's flagship model, 9 languages, 253 voices.
    The legacy "v1" model is absent from current Rime docs; do NOT use it.
  - Each Coda voice serves EXACTLY ONE language; pairing a voice with a
    mismatched `lang` is unsupported and may produce errors or wrong output.
  - A per-language default voice mapping is maintained in _SPEAKER_BY_LANG.
  - For languages not in Rime's supported list (e.g. "zh" Chinese), we fall
    back to English voice + English TTS rather than sending an invalid
    language code to the API and getting an opaque runtime error.

Supported `lang` values for Coda (BCP-47 tags):
  ar  Arabic    (6 voices)   — Coda only, not Mist v3
  en  English   (162 voices)
  fr  French    (8 voices)
  de  German    (9 voices)
  hi  Hindi     (2 voices)   — Coda only, not Mist v3
  it  Italian   (2 voices)   — Coda only, not Mist v3
  ja  Japanese  (13 voices)  — Coda only, not Mist v3
  pt  Portuguese(11 voices)  — Coda only, not Mist v3
  es  Spanish   (40 voices)
"""

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

# ---------------------------------------------------------------------------
# Default voice per language — all confirmed in the live Coda voice catalog.
# Each entry: (speaker_name, verified_coda_lang_tag)
# Override any entry by setting RIME_SPEAKER_<LANG_UPPER> in .env,
# e.g. RIME_SPEAKER_HI=taru  to switch the Hindi default to the male voice.
# ---------------------------------------------------------------------------
_DEFAULT_SPEAKER_BY_LANG: Dict[str, str] = {
    "en": "astra",    # Female, Young Adult, US  — 162 en voices available
    "es": "brisa",    # Female, Young Adult, US  — 40 es voices available
    "ja": "akari",    # Female, Adult, JP         — 13 ja voices available
    "pt": "rio",      # Female, Young Adult, BR   — 11 pt voices available
    "de": "aura",     # Female, Adult, DE         — 9 de voices available
    "fr": "aurelie",  # Female, Young Adult, FR   — 8 fr voices available
    "ar": "layla",    # Female, Adult, EG         — 6 ar voices available (Coda only)
    "hi": "nadi",     # Female, Young Adult, IN   — 2 hi voices available (Coda only)
    "it": "livia",    # Female, Young Adult, IT   — 2 it voices available (Coda only)
}

# Languages supported by the Coda model (confirmed from live docs August 2026)
_CODA_SUPPORTED_LANGS = frozenset(_DEFAULT_SPEAKER_BY_LANG.keys())


def _get_speaker_for_lang(lang: str, default_speaker: str) -> tuple[str, str]:
    """
    Returns (speaker, effective_lang) for a given language code.

    Falls back to (default_speaker, "en") if `lang` is not supported by
    Coda, logging a warning rather than propagating an unsupported combination
    to the API (which would cause an opaque runtime error mid-demo).
    """
    # Normalise: strip region tag for lookup (e.g. "es-MX" -> "es")
    base_lang = lang.split("-")[0].lower() if lang else "en"

    if base_lang not in _CODA_SUPPORTED_LANGS:
        logger.warning(
            f"[Rime TTS] Language '{lang}' is not supported by the Coda model "
            f"(supported: {sorted(_CODA_SUPPORTED_LANGS)}). "
            "Falling back to English voice and English speech. "
            "The spoken response will be in English."
        )
        return (default_speaker, "en")

    # Allow per-language override via environment variable, e.g. RIME_SPEAKER_HI=taru
    env_key = f"RIME_SPEAKER_{base_lang.upper()}"
    speaker = os.getenv(env_key) or _DEFAULT_SPEAKER_BY_LANG[base_lang]
    return (speaker, base_lang)


class RimeTTSService:
    """
    Rime Text-to-Speech API Wrapper.
    All Rime API payload structures, headers, endpoints, and fallback controls are isolated here.
    """
    def __init__(self):
        self.api_key = os.getenv("RIME_API_KEY", "")
        # Default English speaker (used when no per-language mapping matches)
        self.speaker = os.getenv("RIME_SPEAKER", "astra")
        # coda: 9 languages, 253 voices — current Rime flagship model.
        # v1 is a legacy model absent from current docs; do NOT revert to it.
        self.model_id = os.getenv("RIME_MODEL_ID", "coda")
        self.api_url = os.getenv("RIME_API_URL", "https://users.rime.ai/v1/rime-tts")
        self.audio_format = os.getenv("RIME_AUDIO_FORMAT", "mp3") # 'mp3' or 'wav'

    def speak(self, text: str, task_id: str, lang: str = "en") -> Optional[bytes]:
        """
        Generate audio bytes for `text` associated with `task_id`, spoken in `lang`.
        Validates if `task_id` is active before & during operation.

        :param text: Text string for TTS conversion (should be in `lang`'s language)
        :param task_id: Unique task identifier
        :param lang: ISO 639-1 language code (e.g. "en", "hi", "es").
                     Determines both the `lang` API parameter and the speaker voice.
                     Unsupported languages fall back to English gracefully.
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
            audio_bytes = self._call_rime_api(text, task_id, lang=lang)

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

    def _call_rime_api(self, text: str, task_id: str, lang: str = "en") -> bytes:
        """
        Performs HTTP POST request to Rime API.

        Selects the correct speaker voice for `lang` from the per-language
        mapping, then adds `lang` to the payload. Falls back to English if
        `lang` is not in Coda's supported language set.
        """
        # Resolve speaker and effective lang (may fall back to en)
        speaker, effective_lang = _get_speaker_for_lang(lang, self.speaker)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": f"audio/{self.audio_format}"
        }

        # Customizable Rime API Payload Schema
        payload: Dict[str, Any] = {
            "speaker": speaker,
            "text": text,
            "modelId": self.model_id,
            "lang": effective_lang,          # BCP-47 tag, required for Coda multilingual
            "samplingRate": 22050,
            "speedAlpha": 1.0,
            "audioFormat": self.audio_format
        }

        logger.info(
            f"[Rime TTS] Sending request for task '{task_id}': "
            f"speaker='{speaker}' lang='{effective_lang}' model='{self.model_id}'"
        )
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
