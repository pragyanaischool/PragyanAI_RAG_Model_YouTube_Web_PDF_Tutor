"""
Voice and Multilingual Service Module for NCET GenAI Multimodal RAG.
Handles high-speed Speech-to-Text (STT) via Groq Whisper API,
multilingual text translation with chunk-safe batching,
and Text-to-Speech (TTS) audio synthesis using gTTS.
"""

import os
import tempfile
from typing import Optional
from groq import Groq
from googletrans import Translator
from gtts import gTTS
from core.config import GROQ_API_KEY, GROQ_WHISPER_MODEL


def get_groq_client() -> Groq:
    """Instantiates and returns the Groq client using configured credentials."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .streamlit/secrets.toml or environment.")
    return Groq(api_key=GROQ_API_KEY)


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes (e.g. from Streamlit microphone recording) into text
    using Groq's high-throughput whisper-large-v3 API.

    Args:
        audio_bytes (bytes): Raw audio buffer in WAV/MP3 format.

    Returns:
        str: Transcribed text string. Returns empty string on failure.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return ""

    tmp_audio_path = None
    try:
        client = get_groq_client()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name

        with open(tmp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_audio_path), file.read()),
                model=GROQ_WHISPER_MODEL,
                response_format="text",
                temperature=0.0
            )

        return transcription.strip() if transcription else ""

    except Exception as e:
        print(f"[Error in voice_lang_service]: Groq STT Transcription failed: {e}")
        return ""

    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except Exception:
                pass


def translate_content(text: str, target_lang: str) -> str:
    """
    Translates input text into the destination language code (e.g. 'hi', 'kn', 'te', 'es', 'de').
    Splits long content into chunks under 2,000 characters to prevent API rate/timeout errors.

    Args:
        text (str): Source text to translate.
        target_lang (str): ISO destination language code.

    Returns:
        str: Translated text string, or original text if translation fails or target is 'en'.
    """
    if not text or not text.strip():
        return ""

    clean_target = (target_lang or "en").strip().lower()
    if clean_target == "en":
        return text

    try:
        translator = Translator()
        # Chunk text on paragraph/newline boundaries where possible, or slice <= 2000 chars
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        translated_segments = []

        for chunk in chunks:
            if not chunk.strip():
                continue
            res = translator.translate(chunk, dest=clean_target)
            translated_segments.append(res.text if hasattr(res, "text") else str(res))

        return " ".join(translated_segments) if translated_segments else text

    except Exception as e:
        print(f"[Warning in voice_lang_service]: Translation to '{clean_target}' failed: {e}")
        return text


def text_to_speech(text: str, lang_code: str = "en") -> Optional[str]:
    """
    Synthesizes speech audio from text using gTTS and writes to a temporary MP3 file.
    Caps synthesis to the first 1,000 characters for responsive playback in the UI.

    Args:
        text (str): Text content to convert to voice.
        lang_code (str): Language code for voice synthesis.

    Returns:
        Optional[str]: Absolute path to the generated temporary MP3 file.
                       Returns None if synthesis fails or text is empty.
    """
    if not text or not text.strip():
        return None

    clean_lang = (lang_code or "en").strip().lower()
    # Normalize complex locale codes to 2-letter codes supported by gTTS
    base_lang = clean_lang.split("-")[0].split("_")[0]

    try:
        # Synthesize audio summary (max 1000 chars for real-time speed)
        payload_text = text[:1000].strip()
        tts = gTTS(text=payload_text, lang=base_lang, slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts.save(fp.name)
            return fp.name

    except Exception as e:
        # Fallback to English voice if regional language voice engine is unavailable
        if base_lang != "en":
            try:
                tts = gTTS(text=text[:1000].strip(), lang="en", slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    tts.save(fp.name)
                    return fp.name
            except Exception:
                pass

        print(f"[Warning in voice_lang_service]: TTS Audio generation failed: {e}")
        return None
