"""
Core Package for NCET GenAI Multimodal RAG & Intelligence Suite.
Exposes database setup, search services, voice processing, notes synthesis, and exam evaluation.
"""

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_WHISPER_MODEL,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    LANGUAGE_CODES,
)
from .vector_db import index_documents_to_chroma, get_embeddings
from .search_service import search_multiple_youtube_videos, search_and_read_web_articles
from .voice_lang_service import transcribe_audio_bytes, translate_content, text_to_speech
from .note_synthesizer import generate_single_doc_notes, generate_combined_master_notes
from .exam_solver import solve_multiformat_questions, export_assessment_to_pdf

__all__ = [
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_WHISPER_MODEL",
    "CHROMA_PERSIST_DIR",
    "EMBEDDING_MODEL_NAME",
    "LANGUAGE_CODES",
    "index_documents_to_chroma",
    "get_embeddings",
    "search_multiple_youtube_videos",
    "search_and_read_web_articles",
    "transcribe_audio_bytes",
    "translate_content",
    "text_to_speech",
    "generate_single_doc_notes",
    "generate_combined_master_notes",
    "solve_multiformat_questions",
    "export_assessment_to_pdf",
]
