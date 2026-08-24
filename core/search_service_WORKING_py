"""
Search Service Module for PragyanAI GenAI Multimodal RAG.
Robust multi-video YouTube search using yt-dlp native query search,
paired with DuckDuckGo live web search and automated article extraction.
"""

from typing import Any, Dict, List
import yt_dlp
from duckduckgo_search import DDGS
from core.extractors.web_extractor import extract_from_url


def search_multiple_youtube_videos(
    topic: str, max_results: int = 6
) -> List[Dict[str, Any]]:
    """
    Searches YouTube using yt-dlp's built-in ytsearch engine without downloading audio/video.
    Completely avoids third-party scraper proxy/httpx incompatibilities.

    Args:
        topic (str): The search query or educational topic.
        max_results (int): Maximum number of video records to return.

    Returns:
        List[Dict[str, Any]]: List of video records with keys:
            - 'id': YouTube video ID
            - 'title': Video title
            - 'url': Direct YouTube URL
            - 'duration': Formatted duration string (e.g., '12:45')
            - 'channel': Channel / creator name
            - 'views': View count string
            - 'thumbnail': Primary thumbnail image URL
    """
    if not topic or not topic.strip():
        return []

    video_list: List[Dict[str, Any]] = []
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # Metadata-only retrieval without triggering downloads
        "skip_download": True,
    }

    try:
        query_str = f"ytsearch{max_results}:{topic.strip()}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(query_str, download=False)
            entries = result.get("entries", []) if result else []

            for entry in entries:
                if not entry:
                    continue

                duration_sec = entry.get("duration")
                if isinstance(duration_sec, (int, float)):
                    mins, secs = divmod(int(duration_sec), 60)
                    duration_str = f"{mins}:{secs:02d}"
                else:
                    duration_str = "N/A"

                vid_id = entry.get("id", "")
                video_url = entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}"

                thumbnails = entry.get("thumbnails", [])
                thumb_url = (
                    thumbnails[-1].get("url", "")
                    if thumbnails
                    else f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
                )

                video_list.append(
                    {
                        "id": vid_id,
                        "title": entry.get("title", "Untitled Video"),
                        "url": video_url,
                        "duration": duration_str,
                        "channel": entry.get("uploader")
                        or entry.get("channel", "Unknown Channel"),
                        "views": str(entry.get("view_count", "N/A")),
                        "thumbnail": thumb_url,
                    }
                )

        return video_list

    except Exception as e:
        print(f"[Error in search_service]: YouTube search failed for '{topic}': {e}")
        return []


def search_and_read_web_articles(
    query: str, max_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Performs live web search via DuckDuckGo, retrieves the top search URLs,
    and automatically extracts full readable text from each URL using the web extractor.

    Args:
        query (str): The search term or topic.
        max_results (int): Number of top search hits to crawl.

    Returns:
        List[Dict[str, Any]]: List of document payloads ready for vector indexing,
                              each containing 'source', 'type', 'title', and 'content'.
    """
    if not query or not query.strip():
        return []

    scraped_documents: List[Dict[str, Any]] = []

    try:
        with DDGS() as ddgs:
            search_hits = list(ddgs.text(query.strip(), max_results=max_results))

            for hit in search_hits:
                target_url = hit.get("href")
                if not target_url:
                    continue

                snippet_body = hit.get("body", "")
                hit_title = hit.get("title", target_url)

                # Fetch and extract clean article content
                extracted_data = extract_from_url(target_url)

                if extracted_data and extracted_data.get("content"):
                    if (
                        not extracted_data.get("title")
                        or extracted_data["title"] == target_url
                    ):
                        extracted_data["title"] = hit_title
                    scraped_documents.append(extracted_data)
                elif snippet_body:
                    # Fallback to search snippet if full webpage extraction was blocked
                    scraped_documents.append(
                        {
                            "source": target_url,
                            "type": "web",
                            "title": hit_title,
                            "content": (
                                f"[Source: Web Search Snippet]\n"
                                f"[Title: {hit_title}]\n"
                                f"[URL: {target_url}]\n"
                                f"----------------------------------------\n\n"
                                f"{snippet_body.strip()}"
                            ),
                        }
                    )

    except Exception as e:
        print(
            f"[Error in search_service]: Web search & read failed for query '{query}': {e}"
        )

    return scraped_documents
