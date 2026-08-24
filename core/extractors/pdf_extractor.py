"""
PDF Extractor Module for NCET GenAI Multimodal RAG.
Handles uploaded PDF streams, per-page text extraction, 
whitespace normalization, and structured metadata formatting for Vector DB indexing.
"""

import io
import re
from typing import Any, Dict, Optional
from pypdf import PdfReader


def clean_page_text(raw_text: str) -> str:
    """
    Normalizes excessive whitespace, fixes broken line breaks,
    and strips non-printable control characters from extracted text.
    """
    if not raw_text:
        return ""
    # Replace carriage returns with standard newlines
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse consecutive spaces or tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_from_pdf(uploaded_file: Any) -> Optional[Dict[str, Any]]:
    """
    Parses an uploaded PDF file stream or file-like binary object,
    extracts formatted text page by page with explicit page markers,
    and constructs a structured payload ready for ChromaDB chunking.

    Args:
        uploaded_file: Streamlit UploadedFile, file-like object, or bytes buffer.

    Returns:
        Optional[Dict[str, Any]]: Document payload containing:
            - 'source': Document filename
            - 'type': 'pdf'
            - 'title': Document title or filename
            - 'page_count': Total number of pages extracted
            - 'content': Complete per-page structured text
        Returns None if extraction fails, file is encrypted, or PDF contains no readable text.
    """
    if uploaded_file is None:
        return None

    file_name = getattr(uploaded_file, "name", "Uploaded_Document.pdf")

    try:
        # Support Streamlit UploadedFile, BytesIO, or raw byte streams
        if hasattr(uploaded_file, "getvalue"):
            pdf_stream = io.BytesIO(uploaded_file.getvalue())
        elif isinstance(uploaded_file, bytes):
            pdf_stream = io.BytesIO(uploaded_file)
        else:
            pdf_stream = uploaded_file

        reader = PdfReader(pdf_stream)

        # Check for password-protected/encrypted PDFs
        if reader.is_encrypted:
            try:
                # Attempt default empty password decryption
                reader.decrypt("")
            except Exception:
                print(f"[Error in pdf_extractor]: {file_name} is password-protected and cannot be read.")
                return None

        total_pages = len(reader.pages)
        if total_pages == 0:
            return None

        pages_extracted = []
        valid_page_count = 0

        for page_idx, page in enumerate(reader.pages):
            try:
                raw_page_text = page.extract_text() or ""
                cleaned_text = clean_page_text(raw_page_text)

                if cleaned_text:
                    pages_extracted.append(
                        f"--- Page {page_idx + 1} of {total_pages} ---\n{cleaned_text}"
                    )
                    valid_page_count += 1
            except Exception as page_err:
                print(f"[Warning in pdf_extractor]: Could not extract text from Page {page_idx + 1} of {file_name}: {page_err}")
                continue

        # If no readable text was found (e.g. pure scanned image PDF without OCR)
        if not pages_extracted:
            print(f"[Warning in pdf_extractor]: No extractable text found in {file_name}.")
            return None

        full_document_text = "\n\n".join(pages_extracted)

        # Build clean header prefix
        header_block = (
            f"[Source: PDF Document]\n"
            f"[Title: {file_name}]\n"
            f"[Total Pages: {total_pages} | Extracted Pages: {valid_page_count}]\n"
            f"----------------------------------------\n\n"
        )

        full_content = f"{header_block}{full_document_text}"

        return {
            "source": file_name,
            "type": "pdf",
            "title": file_name,
            "page_count": total_pages,
            "content": full_content,
        }

    except Exception as e:
        print(f"[Error in pdf_extractor]: Failed parsing {file_name}: {e}")
        return None
