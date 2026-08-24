"""
Note Synthesizer Module for PragyanAI GenAI Multimodal RAG.
Transforms raw extracted documents and multi-source corpora into exhaustive,
pedagogical markdown study guides with progressive learning tiers (Beginner to Expert),
analogies, code implementations, mathematical formulations, and downloadable PDF/MD reports.
"""

import tempfile
from typing import Any, Dict, List
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.25) -> ChatGroq:
    """Instantiates the ChatGroq model instance using configured credentials."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in secrets or environment.")

    return ChatGroq(
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def generate_single_doc_notes(
    doc_title: str, doc_type: str, content: str
) -> str:
    """
    Expands an individual source document into a comprehensive, multi-tiered
    pedagogical technical handbook.

    Args:
        doc_title (str): Title or filename of the document.
        doc_type (str): Type of source (pdf, docx, pptx, web, youtube).
        content (str): Full or truncated raw text.

    Returns:
        str: Exhaustive markdown text.
    """
    if not content or not content.strip():
        return "### No Content Available\n\nThe selected document has no extractable text."

    llm = get_llm(temperature=0.25)

    prompt = ChatPromptTemplate.from_template(
        "You are a Distinguished AI Research Scientist and Principal Systems Architect. "
        "Your mission is to read the raw excerpt from '{title}' ({doc_type}) and produce an "
        "EXHAUSTIVE, authoritative technical study guide and reference handbook in clean Markdown.\n\n"
        "Ensure the notes provide extreme pedagogical depth, progressing naturally from intuitive foundations "
        "to production-level engineering.\n\n"
        "Structure the markdown guide with these exact sections:\n\n"
        "# 📘 Deep-Dive Study Guide: {title}\n"
        "> **Source Type:** {doc_type} | **Target Audience:** Engineering Students, Researchers, Practitioners\n\n"
        "## 1. 🌐 Executive Summary & High-Level Architecture\n"
        "- The core thesis, problem statement, and why this topic matters.\n"
        "- Mental model and intuitive high-level overview.\n\n"
        "## 2. 💡 Conceptual Analogies (The 'Explain-Like-I'm-5' Layer)\n"
        "- 2 distinct real-world intuitive analogies that demystify abstract concepts.\n\n"
        "## 3. 🪜 Progressive Learning Spectrum (Beginner to Expert)\n"
        "- **Beginner Level:** Fundamental vocabulary, taxonomy, foundational concepts.\n"
        "- **Intermediate Level:** Operational mechanics, mathematical equations/formulations, dataflow.\n"
        "- **Advanced/Expert Level:** Edge-case behavior, hardware/memory bottlenecks, asymptotic complexity, trade-offs.\n\n"
        "## 4. 💻 Practical Implementations & Code Snippets\n"
        "- Well-commented, robust Python/PyTorch/NumPy or system code snippets illustrating the core algorithms.\n\n"
        "## 5. 🚀 Real-World Industrial Applications & Case Studies\n"
        "- 2 to 3 concrete industry use cases showing how leading tech companies or labs deploy this in production.\n\n"
        "## 6. 🔍 Extra Key Concepts & Peripheral Discoveries\n"
        "- Underlying nuances, hidden dependencies, failure modes, and related paradigms discussed in the source.\n\n"
        "## 7. 🎯 High-Yield Exam & Interview Revision Checklist\n"
        "- Top 5 critical questions with crisp, point-blank model answers.\n\n"
        "Raw Source Content:\n"
        "----------------------------------------\n"
        "{content}\n"
        "----------------------------------------"
    )

    trimmed_content = content[:8000].strip()

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "doc_type": doc_type,
                "title": doc_title,
                "content": trimmed_content,
            }
        )
        return response.content

    except Exception as e:
        print(f"[Error in note_synthesizer]: Single document note expansion failed: {e}")
        return f"### Generation Error\n\nFailed to synthesize notes: {str(e)}"


def generate_combined_master_notes(all_docs: List[Dict[str, Any]]) -> str:
    """
    Synthesizes multiple heterogeneous sources into a unified, cross-cutting Master Curriculum.
    """
    if not all_docs:
        return "### No Knowledge Sources Ingested\n\nPlease ingest documents, web links, or YouTube videos first."

    llm = get_llm(temperature=0.25)

    corpus_blocks = []
    for idx, doc in enumerate(all_docs):
        title = doc.get("title", f"Source {idx + 1}")
        dtype = doc.get("type", "generic")
        snippet = doc.get("content", "")[:1800].strip()
        corpus_blocks.append(f"--- [Source {idx + 1}: {title} ({dtype})] ---\n{snippet}")

    combined_corpus = "\n\n".join(corpus_blocks)

    prompt = ChatPromptTemplate.from_template(
        "You are an Academic Director and Principal Curriculum Architect. "
        "Synthesize all provided multi-source learning materials into an integrated, end-to-end Master Curriculum Guide in clean Markdown.\n\n"
        "Structure the comprehensive notes with these sections:\n\n"
        "# 📚 Master Technical Curriculum & Unified Knowledge Base\n\n"
        "## 1. 🌐 Global Synthesis & Cross-Source Synergy\n"
        "- Unified paradigm connecting all ingested sources into a cohesive technology stack.\n\n"
        "## 2. 🧩 Integrated Architectural Pipeline\n"
        "- Step-by-step end-to-end operational pipeline combining concepts from all documents.\n\n"
        "## 3. 🔬 Deep Technical Matrix (Comparative Analysis)\n"
        "- Markdown table comparing techniques, performance tradeoffs, compute requirements, and use cases.\n\n"
        "## 4. 🛠️ End-to-End Implementation Blueprint\n"
        "- Complete illustrative code framework connecting components together.\n\n"
        "## 5. 💡 Real-World System Design & Production Considerations\n"
        "- Scalability, latency, deployment architectures, failure recovery.\n\n"
        "## 6. 🎓 Master Comprehensive Review & Viva-Voce Questions\n"
        "- 5 deep analytical questions testing cross-source conceptual mastery.\n\n"
        "Ingested Multi-Source Corpus:\n"
        "----------------------------------------\n"
        "{corpus}\n"
        "----------------------------------------"
    )

    try:
        chain = prompt | llm
        response = chain.invoke({"corpus": combined_corpus})
        return response.content

    except Exception as e:
        print(f"[Error in note_synthesizer]: Master notes generation failed: {e}")
        return f"### Master Generation Error\n\nFailed to synthesize master notes: {str(e)}"


class NotesPDFReport(FPDF):
    """Custom FPDF class for generating formatted academic study notes."""

    def __init__(self, title_text: str = "Technical Notes"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = title_text
        self.set_margins(left=15, top=15, right=15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, "PragyanAI / NCET GenAI Intelligence Suite | Deep-Dive Notes", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="C")


def export_notes_to_pdf(content: str, title: str = "Deep Dive Study Notes") -> str:
    """Exports generated Markdown study notes into a cleanly formatted PDF file."""
    pdf = NotesPDFReport(title_text=title)
    pdf.add_page()

    # Title Banner
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 8, title, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, "Comprehensive Technical Reference & Learning Blueprint", align="C")
    pdf.ln(8)

    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # Body
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(30, 41, 59)

    sanitized_text = (
        content.replace("—", "-")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("•", "-")
        .replace("`", "'")
    )
    ascii_safe_text = sanitized_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, ascii_safe_text)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name
