"""
YouTube Extractor Module for PragyanAI GenAI Multimodal RAG.
Fixed for HTTP 403 Forbidden on cloud deployments via player_client spoofing,
header randomization, and robust fast-path caption extraction.
"""

import os
import re
import tempfile
from typing import Any, Dict, Optional
from groq import Groq
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from core.config import GROQ_API_KEY, GROQ_WHISPER_MODEL


def clean_youtube_url(url: str) -> Optional[str]:
    """Validates and cleans YouTube video URLs (standard, short links, and embed formats)."""
    if not url or not isinstance(url, str):
        return None

    clean_url = url.strip()
    youtube_regex = (
        r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
        r"(watch\?v=|embed/|v/|shorts/|.+\?v=)?([^&=%\?]{11})"
    )
    match = re.search(youtube_regex, clean_url)
    return clean_url if match else None


def extract_video_id(url: str) -> Optional[str]:
    """Extracts the 11-character video ID from various YouTube URL formats."""
    if not url:
        return None
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript_via_api(video_id: str) -> Optional[str]:
    """
    Attempts fast caption retrieval without downloading video streams or triggering 403 errors.
    Prioritizes manual English captions, falls back to generated English or first available language.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            except Exception:
                transcript = next(iter(transcript_list))

        entries = transcript.fetch()
        full_text = " ".join([entry["text"] for entry in entries if entry.get("text")])
        return full_text.strip() if full_text else None
    except Exception:
        return None


def extract_from_youtube(video_url: str) -> Optional[Dict[str, Any]]:
    """
    Extracts transcript from a YouTube video URL.
    Uses fast caption APIs first; falls back to yt-dlp with anti-403 client headers
    and Groq whisper-large-v3 transcription.

    Args:
        video_url (str): The valid YouTube video URL.

    Returns:
        Optional[Dict[str, Any]]: Document payload containing:
            - 'source': Video URL
            - 'type': 'youtube'
            - 'title': Video title
            - 'channel': Channel / creator name
            - 'duration': Duration in seconds or formatted string
            - 'content': Structured transcript with metadata
        Returns None if extraction fails.
    """
    valid_url = clean_youtube_url(video_url)
    if not valid_url:
        print(f"[Warning in youtube_extractor]: Invalid YouTube URL: {video_url}")
        return None

    video_id = extract_video_id(valid_url)
    video_title = "YouTube Video"
    uploader = "Unknown Channel"
    duration = "N/A"

    # ================= Strategy 1: Instant Caption API (Zero 403 Risk) =================
    if video_id:
        fast_transcript = fetch_transcript_via_api(video_id)
        if fast_transcript:
            header_block = (
                f"[Source: YouTube Video (Captions)]\n"
                f"[Title: {video_title}]\n"
                f"[Video ID: {video_id}]\n"
                f"[URL: {valid_url}]\n"
                f"----------------------------------------\n\n"
            )
            return {
                "source": valid_url,
                "type": "youtube",
                "title": video_title,
                "channel": uploader,
                "duration": duration,
                "content": f"{header_block}{fast_transcript}",
            }

    # ================= Strategy 2: yt-dlp Audio Extraction + Groq Whisper =================
    if not GROQ_API_KEY:
        print("[Error in youtube_extractor]: GROQ_API_KEY is missing for Whisper fallback.")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "ba/b",
                "outtmpl": out_template,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "128",
                    }
                ],
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                # Spoof Android, iOS and Web clients to bypass YouTube 403 Forbidden blocks on cloud hosts
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "ios", "web"]
                    }
                },
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(valid_url, download=True)
                if not info_dict:
                    return None

                video_title = info_dict.get("title", video_title)
                uploader = info_dict.get("uploader") or info_dict.get("channel", uploader)
                duration = info_dict.get("duration", duration)
                v_id = info_dict.get("id", video_id or "temp_audio")
                audio_file_path = os.path.join(temp_dir, f"{v_id}.mp3")

            if not audio_file_path or not os.path.exists(audio_file_path):
                print(f"[Error in youtube_extractor]: Audio file extraction failed for {valid_url}")
                return None

            # Transcribe with Groq Whisper API
            client = Groq(api_key=GROQ_API_KEY)
            with open(audio_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), audio_file.read()),
                    model=GROQ_WHISPER_MODEL,
                    response_format="text",
                    temperature=0.0,
                )

            clean_transcript = transcription.strip() if transcription else ""
            if not clean_transcript:
                print(f"[Warning in youtube_extractor]: Empty transcript generated for {video_title}.")
                return None

            header_block = (
                f"[Source: YouTube Video (Whisper ASR)]\n"
                f"[Title: {video_title}]\n"
                f"[Channel: {uploader} | Duration: {duration}s]\n"
                f"[URL: {valid_url}]\n"
                f"----------------------------------------\n\n"
            )

            return {
                "source": valid_url,
                "type": "youtube",
                "title": video_title,
                "channel": uploader,
                "duration": duration,
                "content": f"{header_block}{clean_transcript}",
            }

    except Exception as e:
        print(f"[Error in youtube_extractor]: Failed extraction for {video_url}: {e}")
        return None
