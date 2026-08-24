"""
Note Synthesizer Module for NCET GenAI Multimodal RAG.
Transforms raw extracted documents (single or multi-source corpora) into structured,
pedagogical slide-by-slide interactive decks using Groq LLM (e.g. openai/gpt-oss-120b).
"""

import re
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Instantiates the ChatGroq model instance using the configured model and API key.

    Args:
        temperature (float): Sampling temperature for generation (default 0.2 for structured clarity).

    Returns:
        ChatGroq: Initialized LangChain ChatGroq model wrapper.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .streamlit/secrets.toml or environment.")

    return ChatGroq(
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def generate_single_doc_notes(
    doc_title: str, doc_type: str, content: str
) -> List[str]:
    """
    Synthesizes a single document's content into 3 to 5 discrete, highly readable
    slide/page sections delimited for carousel/page-by-page display.

    Args:
        doc_title (str): Title or file name of the document.
        doc_type (str): Type of document (pdf, docx, pptx, web, youtube).
        content (str): Full or truncated text content of the document.

    Returns:
        List[str]: List of formatted markdown strings, each representing one presentation slide.
    """
    if not content or not content.strip():
        return ["### No Content Available\n\nThe selected document has no extractable text."]

    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        "You are a Distinguished University Professor and Academic Author. "
        "Synthesize the provided {doc_type} text into concise, high-yield academic study notes.\n\n"
        "Instructions:\n"
        "1. Structure the entire output into exactly 3 to 5 discrete Slide/Page sections.\n"
        "2. You MUST separate each slide using the exact delimiter `=== SLIDE: <Slide Title> ===` at the beginning of each slide.\n"
        "3. Recommended Slide Layout:\n"
        "   - Slide 1: Core Concept Map & Executive Summary (High-level thesis, key taxonomy, main objectives).\n"
        "   - Slide 2 to 4: In-Depth Technical Breakdown (Core mechanisms, formulas/equations, architectural logic, practical case studies).\n"
        "   - Final Slide: Key Takeaways & Exam Revision (Important definitions, common pitfalls, top practice questions).\n"
        "4. Use bullet points, bold key terms, and markdown formatting within each slide.\n\n"
        "Document Title: {title}\n"
        "----------------------------------------\n"
        "Content Corpus:\n"
        "{content}\n"
        "----------------------------------------"
    )

    # Use up to 7,000 characters to stay within prompt limits while preserving context
    trimmed_content = content[:7000].strip()

    try:
        chain = prompt | llm
        raw_response = chain.invoke(
            {
                "doc_type": doc_type,
                "title": doc_title,
                "content": trimmed_content,
            }
        ).content

        # Split on the slide delimiter
        raw_slides = re.split(r"=== SLIDE:\s*", raw_response)
        cleaned_slides = [s.strip() for s in raw_slides if s.strip()]

        if not cleaned_slides:
            return [raw_response.strip()]

        # Re-attach slide title headers nicely if stripped by regex
        formatted_slides = []
        for slide in cleaned_slides:
            lines = slide.split("\n", 1)
            header = lines[0].replace("===", "").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            formatted_slides.append(f"### {header}\n\n{body}")

        return formatted_slides

    except Exception as e:
        print(f"[Error in note_synthesizer]: Single document note generation failed: {e}")
        return [f"### Generation Error\n\nFailed to synthesize notes: {str(e)}"]


def generate_combined_master_notes(all_docs: List[Dict[str, Any]]) -> List[str]:
    """
    Synthesizes heterogeneous multi-source learning materials (Web articles, PDFs,
    DOCX, PPTX presentations, YouTube transcripts) into an integrated Master Curriculum Slide Deck.

    Args:
        all_docs (List[Dict[str, Any]]): List of ingested document dictionaries.

    Returns:
        List[str]: List of formatted markdown strings representing the master curriculum slides.
    """
    if not all_docs:
        return ["### No Knowledge Sources Ingested\n\nPlease ingest documents, web links, or YouTube videos first."]

    llm = get_llm(temperature=0.2)

    # Aggregate excerpts across all sources
    corpus_blocks = []
    for idx, doc in enumerate(all_docs):
        title = doc.get("title", f"Source {idx + 1}")
        dtype = doc.get("type", "generic")
        snippet = doc.get("content", "")[:1800].strip()
        corpus_blocks.append(f"--- [Source {idx + 1}: {title} ({dtype})] ---\n{snippet}")

    combined_corpus = "\n\n".join(corpus_blocks)

    prompt = ChatPromptTemplate.from_template(
        "You are an Academic Director and Curriculum Designer. "
        "Synthesize all provided multi-source learning materials into an integrated Master Study & Revision Curriculum.\n\n"
        "Instructions:\n"
        "1. Structure the entire output into exactly 4 to 6 discrete Master Slides.\n"
        "2. You MUST separate each slide using the exact delimiter `=== SLIDE: <Slide Title> ===` at the beginning of each slide.\n"
        "3. Recommended Structure:\n"
        "   - Slide 1: Global Architectural Overview & Problem Context.\n"
        "   - Slide 2 to 4: Unified Theoretical Foundations & Cross-Source Synthesis (highlighting how different sources connect).\n"
        "   - Slide 5: Real-World Applications, Industry Standards & Practical Implementation.\n"
        "   - Slide 6: Comprehensive Review Matrix, High-Yield Formulas & Exam Checklist.\n"
        "4. Format each slide cleanly using bullet points, tables where appropriate, and bold highlights.\n\n"
        "Ingested Multi-Source Corpus:\n"
        "----------------------------------------\n"
        "{corpus}\n"
        "----------------------------------------"
    )

    try:
        chain = prompt | llm
        raw_response = chain.invoke({"corpus": combined_corpus}).content

        raw_slides = re.split(r"=== SLIDE:\s*", raw_response)
        cleaned_slides = [s.strip() for s in raw_slides if s.strip()]

        if not cleaned_slides:
            return [raw_response.strip()]

        formatted_master_slides = []
        for slide in cleaned_slides:
            lines = slide.split("\n", 1)
            header = lines[0].replace("===", "").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            formatted_master_slides.append(f"### 📘 {header}\n\n{body}")

        return formatted_master_slides

    except Exception as e:
        print(f"[Error in note_synthesizer]: Master curriculum note generation failed: {e}")
        return [f"### Master Generation Error\n\nFailed to synthesize master notes: {str(e)}"]
