"""
Web Extractor Module for NCET GenAI Multimodal RAG.
Handles fetching, clean boiler-plate removal, metadata extraction,
and formatting of live web articles and URLs.
"""

from typing import Any, Dict, Optional
import trafilatura
from trafilatura.settings import use_config


def extract_from_url(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """
    Fetches raw HTML from a target web URL, isolates the readable article body,
    removes boilerplate (navigation bars, ads, footers, sidebars), and creates a
    structured payload for vector indexing.

    Args:
        url (str): The target website or article URL.
        timeout (int): Network request timeout in seconds.

    Returns:
        Optional[Dict[str, Any]]: Standard document payload containing:
            - 'source': clean target URL
            - 'type': 'web'
            - 'title': extracted webpage or article title
            - 'content': structured, readable text representation
        Returns None if extraction fails, timeouts, or yields empty content.
    """
    if not url or not isinstance(url, str):
        return None

    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        return None

    try:
        # Configure trafilatura settings for optimized extraction
        config = use_config()
        config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))

        # Fetch HTML content
        downloaded_html = trafilatura.fetch_url(clean_url, config=config)
        if not downloaded_html:
            return None

        # Extract primary semantic content
        extracted_text = trafilatura.extract(
            downloaded_html,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            no_fallback=False,
            favor_precision=True,
        )

        if not extracted_text or not extracted_text.strip():
            return None

        # Extract metadata (Title, Author, Date)
        metadata = trafilatura.extract_metadata(downloaded_html)
        page_title = (
            metadata.title
            if (metadata and metadata.title)
            else clean_url.split("//")[-1].split("/")[0]
        )
        author = (
            metadata.author if (metadata and metadata.author) else "Unknown"
        )
        published_date = (
            metadata.date if (metadata and metadata.date) else "N/A"
        )

        # Build clean contextual payload for Chroma vector DB
        header_block = (
            f"[Source: Web Article]\n"
            f"[Title: {page_title}]\n"
            f"[URL: {clean_url}]\n"
            f"[Author: {author} | Date: {published_date}]\n"
            f"----------------------------------------\n\n"
        )

        full_content = f"{header_block}{extracted_text.strip()}"

        return {
            "source": clean_url,
            "type": "web",
            "title": page_title,
            "content": full_content,
        }

    except Exception as e:
        print(
            f"[Error in web_extractor]: Failed to extract content from {clean_url}: {e}"
        )
        return None
