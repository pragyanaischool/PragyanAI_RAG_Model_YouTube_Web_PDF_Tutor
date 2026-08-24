"""
Note Synthesizer Module for PragyanAI GenAI Multimodal RAG.
Transforms raw extracted documents and multi-source corpora into clean,
pedagogically structured Markdown guides and beautifully styled PDF reports.
"""

import os
import re
import tempfile
from typing import Any, Dict, List
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """Instantiates ChatGroq model instance using configured credentials."""
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
    pedagogical technical handbook in clean Markdown.
    """
    if not content or not content.strip():
        return "### No Content Available\n\nThe selected document has no extractable text."

    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        "You are a Distinguished AI Research Scientist and Principal Systems Architect. "
        "Read the raw excerpt from '{title}' ({doc_type}) and produce an EXHAUSTIVE, "
        "authoritative technical study guide and reference handbook in clean, standard Markdown.\n\n"
        "Formatting Rules:\n"
        "- Do NOT output ASCII box diagrams. Use clean bulleted workflows or numbered steps instead.\n"
        "- Format code strictly within standard triple-backtick markdown blocks (```python ... ```).\n"
        "- Keep mathematical formulations readable using standard unicode or clear text formulas.\n"
        "- Do not use obscure unicode symbols that degrade into question marks.\n\n"
        "Structure the markdown guide with these exact sections:\n\n"
        "# Deep-Dive Study Guide: {title}\n"
        "> **Source Type:** {doc_type} | **Audience:** Engineering Students, Researchers, Practitioners\n\n"
        "## 1. Executive Summary & Core Foundations\n"
        "- Core thesis, problem statement, and why this topic matters.\n"
        "- Intuitive mental model and high-level architecture.\n\n"
        "## 2. Conceptual Analogies\n"
        "- 2 distinct real-world intuitive analogies that demystify abstract concepts.\n\n"
        "## 3. Progressive Learning Spectrum\n"
        "- **Beginner Level:** Fundamental vocabulary, taxonomy, and foundational concepts.\n"
        "- **Intermediate Level:** Operational mechanics, mathematical formulations, and data flow.\n"
        "- **Advanced / Expert Level:** Edge cases, compute/memory bottlenecks, complexity, and trade-offs.\n\n"
        "## 4. Practical Implementation & Code Snippets\n"
        "- Fully-commented, working Python / PyTorch / NumPy implementation illustrating the core algorithms.\n\n"
        "## 5. Real-World Industrial Applications\n"
        "- 2 to 3 concrete production case studies showing practical industry deployment.\n\n"
        "## 6. Extra Key Concepts & Nuances\n"
        "- Hidden dependencies, failure modes, regularisation, and subtle technical trade-offs.\n\n"
        "## 7. High-Yield Exam & Interview Revision Checklist\n"
        "- Top 5 critical questions with direct, high-scoring model answers.\n\n"
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

    llm = get_llm(temperature=0.2)

    corpus_blocks = []
    for idx, doc in enumerate(all_docs):
        title = doc.get("title", f"Source {idx + 1}")
        dtype = doc.get("type", "generic")
        snippet = doc.get("content", "")[:1800].strip()
        corpus_blocks.append(f"--- [Source {idx + 1}: {title} ({dtype})] ---\n{snippet}")

    combined_corpus = "\n\n".join(corpus_blocks)

    prompt = ChatPromptTemplate.from_template(
        "You are an Academic Director and Principal Curriculum Architect. "
        "Synthesize all provided multi-source learning materials into an integrated, "
        "end-to-end Master Curriculum Guide in clean Markdown.\n\n"
        "Formatting Rules:\n"
        "- Do NOT output ASCII box diagrams.\n"
        "- Format code strictly within standard triple-backtick markdown blocks (```python ... ```).\n"
        "- Format comparative matrices using clean Markdown tables.\n\n"
        "Structure the comprehensive notes with these sections:\n\n"
        "# Master Technical Curriculum & Unified Knowledge Base\n\n"
        "## 1. Global Synthesis & Cross-Source Synergy\n"
        "- Unified paradigm connecting all ingested sources into a cohesive technology stack.\n\n"
        "## 2. Integrated Architectural Pipeline\n"
        "- Step-by-step end-to-end operational pipeline combining concepts from all documents.\n\n"
        "## 3. Deep Technical Comparative Matrix\n"
        "- Markdown table comparing techniques, performance trade-offs, compute requirements, and use cases.\n\n"
        "## 4. End-to-End Implementation Blueprint\n"
        "- Complete illustrative code framework connecting components together.\n\n"
        "## 5. Real-World System Design & Production Considerations\n"
        "- Scalability, latency, deployment architectures, failure recovery.\n\n"
        "## 6. Master Comprehensive Review & Viva-Voce Questions\n"
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


# =====================================================================
# Clean & Robust PDF Generation Engine
# =====================================================================

def sanitize_for_pdf(text: str) -> str:
    """
    Cleans LaTeX formulas, replaces broken Unicode symbols and typographical
    quotes to ensure zero character corruption ('?' artifacts) in PDFs.
    """
    if not text:
        return ""

    replacements = {
        # Quotes and dashes
        "“": '"', "”": '"', "‘": "'", "’": "'", "`": "'",
        "—": " - ", "–": " - ", "…": "...",
        # Emojis & decorative characters
        "📘": "[Guide]", "💡": "[Concept]", "🪜": "[Level]", "💻": "[Code]",
        "🚀": "[Application]", "🔍": "[Nuance]", "🎯": "[Exam]", "📚": "[Master]",
        "🧩": "[Architecture]", "🔬": "[Matrix]", "🛠️": "[Blueprint]", "🎓": "[Viva]",
        "🌐": "[Overview]", "📝": "[Notes]", "🎬": "[Video]", "▶️": "[Play]",
        "✅": "[OK]", "❌": "[Error]", "⏱️": "[Duration]", "🔑": "[Key]",
        "•": "*", "→": "->", "←": "<-", "⇒": "=>", "≤": "<=", "≥": ">=",
        "≠": "!=", "≈": "~=", "×": "*", "÷": "/", "±": "+/-",
        # Common LaTeX variables to readable text
        "\\theta^{*}": "theta*", "\\theta": "theta", "\\nabla": "grad",
        "\\mathcal{L}": "L(Loss)", "\\sum": "SUM", "\\prod": "PROD",
        "\\arg\\min": "argmin", "\\arg\\max": "argmax", "\\frac": "",
        "\\text": "", "\\left": "", "\\right": "", "\\bigl": "", "\\bigr": "",
        "\\epsilon": "epsilon", "\\delta": "delta", "\\sigma": "sigma",
        "\\eta": "eta", "\\le": "<=", "\\ge": ">=", "\\in": "in",
    }

    cleaned = text
    for target, sub in replacements.items():
        cleaned = cleaned.replace(target, sub)

    # Clean leftover LaTeX markup patterns
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = re.sub(r"[{}\^_\$]", "", cleaned)

    # Convert to pure Latin-1 safe bytes
    return cleaned.encode("latin-1", "replace").decode("latin-1")


class ModernAcademicPDF(FPDF):
    """Custom academic PDF renderer with styled typography, headers, and footers."""

    def __init__(self, title_text: str = "Deep-Dive Study Guide"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = title_text
        self.set_margins(left=18, top=18, right=18)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "PRAGYANAI / NCET GENAI INTELLIGENCE SUITE | TECHNICAL STUDY GUIDE", align="L")
        self.ln(6)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.2)
        self.line(18, 14, 192, 14)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.2)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="C")


def export_notes_to_pdf(markdown_content: str, title: str = "Technical Study Guide") -> str:
    """
    Parses Markdown content and generates a publication-quality PDF report
    with syntax boxes for code, callouts, and distinct typography hierarchies.
    """
    clean_title = sanitize_for_pdf(title)
    pdf = ModernAcademicPDF(title_text=clean_title)
    pdf.add_page()

    # Document Header Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 8, clean_title)
    pdf.ln(2)

    # Subtitle
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, "Comprehensive Technical Reference, Implementations & Exam Blueprint")
    pdf.ln(8)

    # Decorative Divider
    pdf.set_draw_color(14, 165, 233)
    pdf.set_line_width(0.8)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(6)

    # Parse markdown line by line
    raw_lines = markdown_content.split("\n")
    in_code_block = False
    code_buffer: List[str] = []

    for line in raw_lines:
        clean_l = sanitize_for_pdf(line)

        # Code Block Delimiter
        if clean_l.strip().startswith("```"):
            if in_code_block:
                # Flush code block buffer
                pdf.set_font("Courier", size=8.5)
                pdf.set_text_color(30, 41, 59)
                pdf.set_fill_color(241, 245, 249)
                pdf.set_draw_color(203, 213, 225)
                
                code_text = "\n".join(code_buffer)
                pdf.multi_cell(0, 4.5, code_text, fill=True, border=1)
                pdf.ln(3)
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buffer.append(clean_l)
            continue

        # Blank Line
        if not clean_l.strip():
            pdf.ln(2)
            continue

        # Heading 1 (# ...)
        if clean_l.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(15, 23, 42)
            pdf.multi_cell(0, 7, clean_l.replace("# ", "").strip())
            pdf.ln(2)

        # Heading 2 (## ...)
        elif clean_l.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(2, 132, 199)
            pdf.multi_cell(0, 6, clean_l.replace("## ", "").strip())
            pdf.ln(1.5)

        # Heading 3 (### ...)
        elif clean_l.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(0, 5.5, clean_l.replace("### ", "").strip())
            pdf.ln(1)

        # Blockquotes (> ...)
        elif clean_l.startswith(">"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(71, 85, 105)
            pdf.set_fill_color(248, 250, 252)
            quote_body = clean_l.replace(">", "").strip()
            pdf.multi_cell(0, 5, quote_body, fill=True, border="L")
            pdf.ln(2)

        # Bullet Points (* or -)
        elif clean_l.strip().startswith(("- ", "* ")):
            pdf.set_font("Helvetica", size=9.5)
            pdf.set_text_color(51, 65, 85)
            bullet_body = clean_l.strip()[2:].strip()
            pdf.cell(5, 5, chr(149), align="R")  # Clean bullet point symbol
            pdf.multi_cell(0, 5, f" {bullet_body}")
            pdf.ln(1)

        # Standard Paragraph Text
        else:
            pdf.set_font("Helvetica", size=9.5)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 5, clean_l)
            pdf.ln(1.5)

    # Output to Temporary File
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
        return tmp.name
