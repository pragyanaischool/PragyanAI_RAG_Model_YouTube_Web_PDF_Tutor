"""
Multi-Source Search & Research Engine for PragyanAI GenAI Multimodal RAG.
Supports arXiv Research Papers, Wikipedia Encyclopedic Pages,
Tech Blogs/Articles, Google Links, and YouTube Video Discovery.
"""

from typing import Any, Dict, List, Optional
import arxiv
import wikipedia
import yt_dlp
from duckduckgo_search import DDGS
from core.extractors.web_extractor import extract_from_url


# =====================================================================
# 1. arXiv Research Paper Search & Extraction
# =====================================================================

def search_arxiv_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches arXiv for academic research papers, returns metadata and summary abstracts.
    """
    if not query or not query.strip():
        return []

    papers: List[Dict[str, Any]] = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query.strip(),
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            authors_str = ", ".join([a.name for a in result.authors[:4]])
            if len(result.authors) > 4:
                authors_str += " et al."

            published_date = result.published.strftime("%Y-%m-%d") if result.published else "N/A"
            categories_str = ", ".join(result.categories[:3])

            header_block = (
                f"[Source: arXiv Research Paper]\n"
                f"[Title: {result.title}]\n"
                f"[Authors: {authors_str}]\n"
                f"[Published: {published_date} | Categories: {categories_str}]\n"
                f"[arXiv ID: {result.entry_id} | PDF: {result.pdf_url}]\n"
                f"----------------------------------------\n\n"
            )

            papers.append({
                "title": result.title,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "authors": authors_str,
                "date": published_date,
                "snippet": result.summary.replace("\n", " ")[:350] + "...",
                "source_type": "arxiv",
                "badge": "🔬 arXiv Paper",
                "content": f"{header_block}Abstract:\n{result.summary.strip()}",
            })
    except Exception as e:
        print(f"[Error in search_service]: arXiv search failed for '{query}': {e}")

    return papers


# =====================================================================
# 2. Wikipedia Encyclopedic Search & Extraction
# =====================================================================

def search_wikipedia_articles(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Searches Wikipedia for matching encyclopedic topic pages and extracts content.
    """
    if not query or not query.strip():
        return []

    wiki_results: List[Dict[str, Any]] = []
    try:
        search_titles = wikipedia.search(query.strip(), results=max_results)

        for title in search_titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                
                header_block = (
                    f"[Source: Wikipedia Knowledge Base]\n"
                    f"[Title: {page.title}]\n"
                    f"[URL: {page.url}]\n"
                    f"----------------------------------------\n\n"
                )

                # Use up to first 6,000 characters of full page content
                body_content = page.content[:6000] if hasattr(page, "content") else summary

                wiki_results.append({
                    "title": page.title,
                    "url": page.url,
                    "snippet": summary,
                    "source_type": "wikipedia",
                    "badge": "📖 Wikipedia",
                    "content": f"{header_block}{body_content}",
                })
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
    except Exception as e:
        print(f"[Error in search_service]: Wikipedia search failed for '{query}': {e}")

    return wiki_results


# =====================================================================
# 3. Google Links, Tech Articles & Engineering Blogs
# =====================================================================

def search_google_and_blogs(
    query: str, target_type: str = "all", max_results: int = 6
) -> List[Dict[str, Any]]:
    """
    Discovers Google search links, Medium/Substack blogs, documentation, and web articles.
    """
    if not query or not query.strip():
        return []

    # Augment query for targeted blog / technical discovery
    refined_query = query.strip()
    badge_label = "🌐 Google Link"
    
    if target_type == "blogs":
        refined_query += " (site:medium.com OR site:towardsdatascience.com OR site:substack.com OR blog)"
        badge_label = "✍️ Tech Blog"
    elif target_type == "articles":
        refined_query += " (article OR tutorial OR guide OR documentation)"
        badge_label = "📰 Web Article"

    results: List[Dict[str, Any]] = []

    # Attempt 1: Google Search
    try:
        from googlesearch import search as gsearch
        raw_results = list(gsearch(refined_query, num_results=max_results, advanced=True))
        for res in raw_results:
            results.append({
                "title": getattr(res, "title", None) or res.url,
                "url": res.url,
                "snippet": getattr(res, "description", "") or "",
                "source_type": target_type if target_type != "all" else "google",
                "badge": badge_label,
            })
        if results:
            return results
    except Exception:
        pass

    # Attempt 2: DuckDuckGo Fallback
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(refined_query, max_results=max_results))
            for hit in hits:
                results.append({
                    "title": hit.get("title", hit.get("href")),
                    "url": hit.get("href"),
                    "snippet": hit.get("body", ""),
                    "source_type": target_type if target_type != "all" else "google",
                    "badge": badge_label,
                })
    except Exception as e:
        print(f"[Error in search_service]: Web search discovery failed for '{query}': {e}")

    return results


def search_and_read_web_articles(
    query: str, max_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Performs live web search and extracts clean readable text from top hits.
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

                extracted_data = extract_from_url(target_url)

                if extracted_data and extracted_data.get("content"):
                    if not extracted_data.get("title") or extracted_data["title"] == target_url:
                        extracted_data["title"] = hit_title
                    scraped_documents.append(extracted_data)
                elif snippet_body:
                    scraped_documents.append({
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
                    })
    except Exception as e:
        print(f"[Error in search_service]: Web search & read failed for query '{query}': {e}")

    return scraped_documents


# =====================================================================
# 4. YouTube Video Discovery Engine
# =====================================================================

def search_multiple_youtube_videos(
    topic: str, max_results: int = 6
) -> List[Dict[str, Any]]:
    """
    Searches YouTube using yt-dlp's built-in ytsearch engine without downloading media files.
    """
    if not topic or not topic.strip():
        return []

    video_list: List[Dict[str, Any]] = []
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
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

                video_list.append({
                    "id": vid_id,
                    "title": entry.get("title", "Untitled Video"),
                    "url": video_url,
                    "duration": duration_str,
                    "channel": entry.get("uploader") or entry.get("channel", "Unknown Channel"),
                    "views": str(entry.get("view_count", "N/A")),
                    "thumbnail": thumb_url,
                })

        return video_list

    except Exception as e:
        print(f"[Error in search_service]: YouTube search failed for '{topic}': {e}")
        return []
