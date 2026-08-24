"""
Search Service Module for NCET GenAI Multimodal RAG.
Provides multi-video YouTube search with rich metadata extraction and 
DuckDuckGo live web search paired with deep article text scraping.
"""

from typing import Any, Dict, List
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch
from core.extractors.web_extractor import extract_from_url


def search_multiple_youtube_videos(
    topic: str, max_results: int = 6
) -> List[Dict[str, Any]]:
    """
    Searches YouTube for videos matching a given topic or query and extracts
    structured metadata including video IDs, titles, thumbnails, durations, and channels.

    Args:
        topic (str): The search query or educational topic.
        max_results (int): Maximum number of video records to return.

    Returns:
        List[Dict[str, Any]]: List of video records with keys:
            - 'id': YouTube video ID
            - 'title': Video title
            - 'url': Direct YouTube URL
            - 'duration': Video duration string (e.g. '12:45')
            - 'channel': Channel / creator name
            - 'views': View count string
            - 'thumbnail': Primary thumbnail image URL
    """
    if not topic or not topic.strip():
        return []

    try:
        videos_search = VideosSearch(topic.strip(), limit=max_results)
        raw_result = videos_search.result() or {}
        raw_videos = raw_result.get("result", [])

        video_list: List[Dict[str, Any]] = []
        for vid in raw_videos:
            thumbnails = vid.get("thumbnails", [])
            thumb_url = thumbnails[0].get("url", "") if thumbnails else ""
            channel_info = vid.get("channel", {})
            channel_name = channel_info.get("name", "Unknown Channel") if isinstance(channel_info, dict) else "Unknown Channel"
            view_info = vid.get("viewCount", {})
            views = view_info.get("short", "N/A") if isinstance(view_info, dict) else "N/A"

            video_list.append(
                {
                    "id": vid.get("id", ""),
                    "title": vid.get("title", "Untitled Video"),
                    "url": vid.get("link", f"https://www.youtube.com/watch?v={vid.get('id', '')}"),
                    "duration": vid.get("duration", "N/A"),
                    "channel": channel_name,
                    "views": views,
                    "thumbnail": thumb_url,
                }
            )

        return video_list

    except Exception as e:
        print(f"[Error in search_service]: YouTube video search failed for query '{topic}': {e}")
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
            # Fetch search hits from DuckDuckGo
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
                    # Use search title if extraction produced an empty title
                    if not extracted_data.get("title") or extracted_data["title"] == target_url:
                        extracted_data["title"] = hit_title
                    scraped_documents.append(extracted_data)
                else:
                    # Fallback to search snippet if full webpage extraction was blocked
                    if snippet_body:
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
        print(f"[Error in search_service]: Web search & read failed for query '{query}': {e}")

    return scraped_documents
