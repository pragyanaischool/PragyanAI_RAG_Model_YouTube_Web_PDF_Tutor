"""
YouTube Extractor Module for PragyanAI GenAI Multimodal RAG.
Handles audio downloading from YouTube URLs using yt-dlp, downsampling to lightweight
mono audio, and executing fast speech-to-text transcription via Groq's whisper-large-v3 API.
"""

import os
import re
import tempfile
from typing import Any, Dict, Optional
from groq import Groq
import yt_dlp
from core.config import GROQ_API_KEY, GROQ_WHISPER_MODEL


def clean_youtube_url(url: str) -> Optional[str]:
    """
    Validates and cleans YouTube video URLs (standard, shortened, and shorts).
    """
    if not url or not isinstance(url, str):
        return None
    
    clean_url = url.strip()
    youtube_regex = (
        r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
        r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )
    match = re.search(youtube_regex, clean_url)
    return clean_url if match else None


def extract_from_youtube(video_url: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the audio stream from a YouTube video URL using yt-dlp,
    transcribes it with Groq's Whisper API, and formats the transcript payload
    for ChromaDB vector storage.

    Args:
        video_url (str): The valid YouTube video link.

    Returns:
        Optional[Dict[str, Any]]: Standard document payload containing:
            - 'source': clean YouTube URL
            - 'type': 'youtube'
            - 'title': extracted video title
            - 'channel': channel or uploader name
            - 'duration': video duration in seconds
            - 'content': structured transcript with video metadata
        Returns None if download or transcription fails.
    """
    valid_url = clean_youtube_url(video_url)
    if not valid_url:
        print(f"[Warning in youtube_extractor]: Invalid YouTube URL format: {video_url}")
        return None

    if not GROQ_API_KEY:
        print("[Error in youtube_extractor]: GROQ_API_KEY is not configured.")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # yt-dlp options: download lowest-overhead mono MP3 audio
            out_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
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
            }

            video_title = "YouTube Video"
            uploader = "Unknown"
            duration = "N/A"
            audio_file_path = None

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(valid_url, download=True)
                if not info_dict:
                    return None

                video_title = info_dict.get("title", "YouTube Video")
                uploader = info_dict.get("uploader", "Unknown Channel")
                duration = info_dict.get("duration", "N/A")
                video_id = info_dict.get("id", "temp_audio")
                audio_file_path = os.path.join(temp_dir, f"{video_id}.mp3")

            if not audio_file_path or not os.path.exists(audio_file_path):
                print(f"[Error in youtube_extractor]: Audio file extraction failed for {valid_url}")
                return None

            # Transcribe audio using Groq Whisper API
            client = Groq(api_key=GROQ_API_KEY)
            with open(audio_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), audio_file.read()),
                    model=GROQ_WHISPER_MODEL,
                    response_format="text",
                    temperature=0.0
                )

            clean_transcript = transcription.strip() if transcription else ""
            if not clean_transcript:
                print(f"[Warning in youtube_extractor]: No transcription generated for {video_title}.")
                return None

            # Build metadata header for semantic retrieval
            header_block = (
                f"[Source: YouTube Video]\n"
                f"[Title: {video_title}]\n"
                f"[Channel: {uploader} | Duration: {duration}s]\n"
                f"[URL: {valid_url}]\n"
                f"----------------------------------------\n\n"
            )

            full_content = f"{header_block}{clean_transcript}"

            return {
                "source": valid_url,
                "type": "youtube",
                "title": video_title,
                "channel": uploader,
                "duration": duration,
                "content": full_content,
            }

    except Exception as e:
        print(f"[Error in youtube_extractor]: Failed processing {video_url}: {e}")
        return None
