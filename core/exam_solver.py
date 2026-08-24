"""
Exam Solver & Evaluation Module for NCET GenAI Multimodal RAG.
Parses mixed questions (MCQs, Short-Answer, Long/Analytical/Essay),
retrieves grounded context from ChromaDB, constructs marks-calibrated model answers,
provides deep pedagogical explanations, distractor analysis, verifiable citations,
and generates downloadable PDF assessment reports using fpdf2.
"""

import os
import tempfile
from typing import Any, List
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL


def get_evaluator_llm(temperature: float = 0.1) -> ChatGroq:
    """
    Instantiates ChatGroq with deterministic low temperature for rigorous academic evaluation.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .streamlit/secrets.toml or environment.")

    return ChatGroq(
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def solve_multiformat_questions(retriever: Any, questions_raw_text: str) -> str:
    """
    Evaluates questions against vector chunks, providing model solutions,
    MCQ distractor breakdowns, grading rubrics, and explicit inline citations.

    Args:
        retriever: LangChain Chroma vectorstore retriever.
        questions_raw_text (str): Raw string containing question items.

    Returns:
        str: Comprehensive formatted Markdown solution report.
    """
    if not questions_raw_text or not questions_raw_text.strip():
        return "### No Questions Provided\n\nPlease upload a question paper or paste questions to solve."

    llm = get_evaluator_llm(temperature=0.1)

    # Retrieve relevant background knowledge from indexed ChromaDB
    retrieved_docs = retriever.invoke(questions_raw_text[:2500])

    # Build context with citation markers
    context_blocks: List[str] = []
    for idx, doc in enumerate(retrieved_docs):
        src_name = doc.metadata.get("title", doc.metadata.get("source", "Document"))
        src_type = doc.metadata.get("type", "source")
        context_blocks.append(
            f"[[CITATION_{idx + 1} | Source: {src_name} ({src_type})]]\n{doc.page_content}"
        )
    formatted_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a Distinguished Academic Examiner, University Professor, and Master Evaluator. "
        "Your task is to analyze the provided question set (which may contain MCQs, Short-Answer Questions, "
        "and Long/Essay Questions) and generate authoritative, high-scoring model answers grounded strictly "
        "in the reference context provided.\n\n"
        "Follow these structural rubrics for every question:\n\n"
        "### 1. For Multiple Choice Questions (MCQs):\n"
        "- **Question Header:** Identify question number and MCQ tag.\n"
        "- **Correct Option & Key:** State the exact letter and option text clearly.\n"
        "- **Concept & Rationale:** 2-3 sentences explaining the underlying theory and why this option is correct.\n"
        "- **Distractor Analysis:** Explain why each incorrect option is invalid or irrelevant.\n"
        "- **Citation:** Cite the exact `[[CITATION_X]]` tag supporting the answer.\n\n"
        "### 2. For Short-Answer Questions (2 to 5 Marks):\n"
        "- **Core Thesis / Definition:** Direct, high-precision definition in 1-2 sentences.\n"
        "- **Key Points:** 3 to 5 bullet points breaking down core mechanisms or steps.\n"
        "- **Detailed Technical Summary:** 1 clear paragraph synthesizing the explanation.\n"
        "- **Citation:** Cite the supporting `[[CITATION_X]]` tags.\n\n"
        "### 3. For Long / Essay / Analytical Questions (6+ Marks):\n"
        "- **Executive Overview & Scope:** Conceptual summary and foundational theory.\n"
        "- **In-Depth Technical Breakdown:** Thorough theoretical analysis, equations, architecture, or workflows.\n"
        "- **Practical Example / Case Study / Industry Application:** Concrete implementation context.\n"
        "- **Examiner Marking Rubric:** 2 to 3 point breakdown showing how an examiner should score this answer.\n"
        "- **Citations:** Cite all applicable `[[CITATION_X]]` tags.\n\n"
        "Rules:\n"
        "- Always cite explicitly using the provided `[[CITATION_X]]` anchors.\n"
        "- Do not fabricate citations; only reference provided context blocks."
    )

    user_prompt = (
        "Reference Knowledge Base with Citations:\n"
        "-----------------------------------------\n"
        "{context}\n"
        "-----------------------------------------\n\n"
        "Questions to Evaluate & Solve:\n"
        "-------------------------------\n"
        "{questions}\n"
        "-------------------------------"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({
            "context": formatted_context,
            "questions": questions_raw_text,
        })
        return response.content

    except Exception as e:
        print(f"[Error in exam_solver]: Failed solving questions: {e}")
        return f"### Evaluation Error\n\nFailed to generate answers: {str(e)}"


class AcademicPDFReport(FPDF):
    """Custom FPDF class with header, footer, and page numbering."""

    def __init__(self, title_text: str = "Assessment Solutions"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = title_text
        self.set_margins(left=15, top=15, right=15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, "NCET GenAI Intelligence Suite | Model Solutions & Citations", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        page_str = f"Page {self.page_no()}/{{nb}}"
        self.cell(0, 6, page_str, align="C")


def export_assessment_to_pdf(
    content: str, title: str = "Academic Assessment & Model Answers"
) -> str:
    """
    Generates a cleanly formatted academic PDF document from markdown text.

    Args:
        content (str): Text content with question answers and citations.
        title (str): Document title for the header.

    Returns:
        str: Filepath to the generated temporary PDF file.
    """
    pdf = AcademicPDFReport(title_text=title)
    pdf.add_page()

    # Title Banner
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 8, title, align="C")
    pdf.ln(2)

    # Sub-header
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, "Marks-Calibrated Solutions & Verified Citation Matrix", align="C")
    pdf.ln(8)

    # Horizontal Rule
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # Body Content
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(30, 41, 59)

    # Clean non-latin-1 characters for safe standard PDF rendering
    sanitized_text = (
        content.replace("—", "-")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("•", "-")
    )
    ascii_safe_text = sanitized_text.encode("latin-1", "replace").decode("latin-1")

    # Render lines with proper line spacing
    pdf.multi_cell(0, 6, ascii_safe_text)

    # Output to Temporary File
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name
