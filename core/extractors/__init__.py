"""
Extractors package for NCET GenAI Multimodal RAG.
Exports clean ingestion utilities for Web, PDF, DOCX, PPTX, and YouTube sources.
"""

from .web_extractor import extract_from_url
from .pdf_extractor import extract_from_pdf
from .docx_extractor import extract_from_docx
from .pptx_extractor import extract_from_pptx
from .youtube_extractor import extract_from_youtube

__all__ = [
    "extract_from_url",
    "extract_from_pdf",
    "extract_from_docx",
    "extract_from_pptx",
    "extract_from_youtube",
]
