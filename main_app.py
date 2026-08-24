"""
Main Streamlit Application for NCET / PragyanAI GenAI Multimodal RAG Suite.
Features native yt-dlp search, Deep-Dive Markdown Notes Studio with PDF export, Exam Solver, and Voice RAG.
"""

import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Safe cross-version LangChain chain import with fallback
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    LEGACY_CHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        LEGACY_CHAIN_AVAILABLE = True
    except ImportError:
        LEGACY_CHAIN_AVAILABLE = False

from core.config import LANGUAGE_CODES, GROQ_MODEL, GROQ_API_KEY
from core.vector_db import index_documents_to_chroma
from core.extractors import (
    extract_from_url,
    extract_from_pdf,
    extract_from_docx,
    extract_from_pptx,
    extract_from_youtube,
)
from core.search_service import search_multiple_youtube_videos, search_and_read_web_articles
from core.voice_lang_service import transcribe_audio_bytes, text_to_speech, translate_content
from core.note_synthesizer import (
    generate_single_doc_notes,
    generate_combined_master_notes,
    export_notes_to_pdf,
)
from core.exam_solver import solve_multiformat_questions, export_assessment_to_pdf

# Page configuration
st.set_page_config(
    page_title="NCET GenAI Intelligence Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State Variables
for state_key, default_val in [
    ("extracted_docs", []),
    ("doc_notes", {}),
    ("master_notes", ""),
    ("yt_search_results", []),
    ("selected_video", None),
    ("current_trans", None),
    ("trans_text", ""),
]:
    if state_key not in st.session_state:
        st.session_state[state_key] = default_val

# Helper to normalize notes to string for downloads
def notes_to_str(content) -> str:
    if isinstance(content, list):
        return "\n\n---\n\n".join(content)
    return str(content or "")

# App Header
st.markdown(
    f"""
<div class="main-header">
    <h1>🎓 NCET GenAI: Multimodal RAG & Deep-Dive Notes Studio</h1>
    <p>Powered by Groq <code>{GROQ_MODEL}</code> & <code>whisper-large-v3</code> • In-Depth Notes Synthesizer • Exam Solver & Citations</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar Configuration & Ingestion
with st.sidebar:
    st.header("⚙️ Configuration & Ingestion")
    if GROQ_API_KEY:
        st.success("🔑 Groq API Key Active")
    else:
        key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input

    st.caption(f"Active Model: `{GROQ_MODEL}`")
    selected_lang = st.selectbox("🌐 Interface Language", list(LANGUAGE_CODES.keys()))
    lang_code = LANGUAGE_CODES[selected_lang]

    st.divider()
    st.subheader("📥 Ingest Sources")
    web_topic = st.text_input("Search Web Topic", placeholder="e.g. Backpropagation")
    direct_url = st.text_input("Direct WebLink URL", placeholder="https://example.com")
    yt_url = st.text_input("Direct YouTube URL", placeholder="https://youtube.com/watch?v=...")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, PPTX)",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Ingest & Generate Notes", use_container_width=True):
        new_docs = []
        with st.spinner("Extracting multi-source documents and vectorizing..."):
            if web_topic:
                new_docs.extend(search_and_read_web_articles(web_topic, max_results=2))
            if direct_url:
                web_doc = extract_from_url(direct_url)
                if web_doc:
                    new_docs.append(web_doc)
            if yt_url:
                yt_doc = extract_from_youtube(yt_url)
                if yt_doc:
                    new_docs.append(yt_doc)
            if uploaded_files:
                for f in uploaded_files:
                    fname = f.name.lower()
                    if fname.endswith(".pdf"):
                        p_doc = extract_from_pdf(f)
                        if p_doc:
                            new_docs.append(p_doc)
                    elif fname.endswith(".docx"):
                        d_doc = extract_from_docx(f)
                        if d_doc:
                            new_docs.append(d_doc)
                    elif fname.endswith(".pptx"):
                        pp_doc = extract_from_pptx(f)
                        if pp_doc:
                            new_docs.append(pp_doc)

            if new_docs:
                st.session_state["extracted_docs"].extend(new_docs)
                st.session_state["vectorstore"] = index_documents_to_chroma(
                    st.session_state["extracted_docs"]
                )

                # Generate comprehensive individual notes
                for doc in new_docs:
                    st.session_state["doc_notes"][doc["title"]] = generate_single_doc_notes(
                        doc["title"], doc["type"], doc["content"]
                    )

                # Generate comprehensive combined master notes
                st.session_state["master_notes"] = generate_combined_master_notes(
                    st.session_state["extracted_docs"]
                )
                st.success(f"✅ Ingested {len(new_docs)} source(s) and generated comprehensive notes!")
            else:
                st.warning("Please provide at least one input source.")

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎥 YouTube Studio & Player",
        "📖 Deep-Dive Notes Studio",
        "📄 Exam Solver & Citations",
        "🎙️ Voice & Text RAG Chat",
    ]
)

# ----------------- TAB 1: YOUTUBE STUDIO -----------------
with tab1:
    st.subheader("🎥 Multi-Video Topic Discovery & Multilingual Player")
    col1, col2 = st.columns([3, 1])
    with col1:
        yt_search_query = st.text_input(
            "Search topic for YouTube videos:",
            placeholder="e.g. Transformer Attention Mechanisms",
        )
    with col2:
        st.write("")
        if st.button("🔍 Search Videos", use_container_width=True) and yt_search_query:
            with st.spinner("Searching YouTube via yt-dlp engine..."):
                st.session_state["yt_search_results"] = search_multiple_youtube_videos(
                    yt_search_query, max_results=6
                )

    if st.session_state["yt_search_results"]:
        cols = st.columns(3)
        for i, vid in enumerate(st.session_state["yt_search_results"]):
            with cols[i % 3]:
                if vid.get("thumbnail"):
                    st.image(vid["thumbnail"], use_container_width=True)
                st.markdown(f"**{vid['title'][:45]}...**")
                st.caption(f"⏱️ {vid['duration']} | {vid['channel']}")
                if st.button(
                    "🎬 Select & Transcribe",
                    key=f"yt_card_{vid['id']}",
                    use_container_width=True,
                ):
                    st.session_state["selected_video"] = vid
                    with st.spinner("Transcribing and synthesizing deep-dive notes..."):
                        trans_data = extract_from_youtube(vid["url"])
                        if trans_data:
                            st.session_state["current_trans"] = trans_data
                            st.session_state["trans_text"] = translate_content(
                                trans_data["content"], lang_code
                            )
                            st.session_state["extracted_docs"].append(trans_data)
                            st.session_state["vectorstore"] = index_documents_to_chroma(
                                st.session_state["extracted_docs"]
                            )
                            st.session_state["doc_notes"][trans_data["title"]] = (
                                generate_single_doc_notes(
                                    trans_data["title"],
                                    trans_data["type"],
                                    trans_data["content"],
                                )
                            )
                            st.success("✅ Transcribed and synthesized notes!")

    if st.session_state["selected_video"] and st.session_state["current_trans"]:
        st.markdown("---")
        col_p, col_t = st.columns(2)
        with col_p:
            st.markdown(f"**Now Playing:** {st.session_state['selected_video']['title']}")
            st.video(st.session_state["selected_video"]["url"])
        with col_t:
            st.markdown(f"**Transcript ({selected_lang})**")
            audio_file = text_to_speech(st.session_state["trans_text"], lang_code)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
            st.text_area(
                "Transcript Content",
                st.session_state["trans_text"],
                height=260,
            )

# ----------------- TAB 2: DEEP-DIVE NOTES STUDIO -----------------
with tab2:
    st.subheader("📖 Deep-Dive Source Notes & Enhanced Learning Guides")
    sub1, sub2 = st.tabs(["Individual Source Handbook", "Unified Master Curriculum"])

    with sub1:
        if not st.session_state["doc_notes"]:
            st.info("Ingest sources from the sidebar to generate comprehensive technical notes.")
        else:
            chosen_doc = st.selectbox(
                "Select Ingested Document / Source:",
                list(st.session_state["doc_notes"].keys()),
            )
            raw_notes_content = st.session_state["doc_notes"][chosen_doc]
            notes_str = notes_to_str(raw_notes_content)

            # Download Action Bar
            col_d1, col_d2, col_d3 = st.columns([1.5, 1.5, 4])
            with col_d1:
                st.download_button(
                    label="📥 Download as Markdown (.md)",
                    data=notes_str,
                    file_name=f"{chosen_doc[:30].replace(' ', '_')}_Notes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                pdf_p = export_notes_to_pdf(notes_str, title=f"Study Guide: {chosen_doc[:40]}")
                with open(pdf_p, "rb") as pdf_f:
                    st.download_button(
                        label="📥 Download as PDF (.pdf)",
                        data=pdf_f.read(),
                        file_name=f"{chosen_doc[:30].replace(' ', '_')}_Notes.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            st.divider()
            st.markdown(notes_str)

    with sub2:
        if not st.session_state["master_notes"]:
            st.info("Unified master curriculum will appear here once sources are ingested.")
        else:
            master_str = notes_to_str(st.session_state["master_notes"])

            col_m1, col_m2, col_m3 = st.columns([1.5, 1.5, 4])
            with col_m1:
                st.download_button(
                    label="📥 Download Master Notes (.md)",
                    data=master_str,
                    file_name="Unified_Master_Curriculum_Notes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_m2:
                master_pdf = export_notes_to_pdf(master_str, title="Unified Master Curriculum")
                with open(master_pdf, "rb") as m_pdf_f:
                    st.download_button(
                        label="📥 Download Master PDF (.pdf)",
                        data=m_pdf_f.read(),
                        file_name="Unified_Master_Curriculum_Notes.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            st.divider()
            st.markdown(master_str)

# ----------------- TAB 3: EXAM SOLVER & CITATIONS -----------------
with tab3:
    st.subheader("📄 Exam Paper Solver with Marks Rubrics & Grounded Citations")
    q_file = st.file_uploader(
        "Upload Question Paper (PDF, DOCX)",
        type=["pdf", "docx"],
        key="q_paper_file",
    )
    q_txt = st.text_area(
        "Or Paste Questions Directly:",
        placeholder=(
            "1. [MCQ] What is the primary purpose of layer normalization in Transformers?\n"
            "   A) Reduce parameters  B) Stabilize hidden state dynamics  C) Increase depth\n\n"
            "2. [Short - 3M] Explain the bottleneck of recurrent neural network architectures.\n\n"
            "3. [Long - 10M] Derive and explain Multi-Head Attention mechanisms."
        ),
        height=180,
    )

    raw_questions = ""
    if q_file:
        fname = q_file.name.lower()
        if fname.endswith(".pdf"):
            res = extract_from_pdf(q_file)
            if res:
                raw_questions = res["content"]
        elif fname.endswith(".docx"):
            res = extract_from_docx(q_file)
            if res:
                raw_questions = res["content"]
    elif q_txt:
        raw_questions = q_txt

    if st.button("🚀 Solve Questions with Marks & Citations", use_container_width=True):
        if "vectorstore" not in st.session_state:
            st.error("Please ingest sources in the sidebar first to build the Vector DB.")
        elif not raw_questions:
            st.warning("Please provide questions by uploading a file or typing above.")
        else:
            with st.spinner("Analyzing questions and generating grounded solutions..."):
                retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6})
                solved_output = solve_multiformat_questions(retriever, raw_questions)
                st.session_state["solved_exam"] = solved_output

    if "solved_exam" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["solved_exam"])
        exam_pdf = export_assessment_to_pdf(st.session_state["solved_exam"])
        with open(exam_pdf, "rb") as f:
            st.download_button(
                "📥 Download Model Solutions (PDF)",
                f.read(),
                file_name="Model_Solutions_With_Citations.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ----------------- TAB 4: VOICE & TEXT RAG -----------------
with tab4:
    st.subheader("🎙️ Multilingual Voice & Text Assistant")
    col_mic, col_txt = st.columns([1, 4])
    with col_mic:
        st.markdown("**Speak Query**")
        audio_record = audio_recorder(text="", recording_color="#e11d48", neutral_color="#0284c7")
    with col_txt:
        text_query = st.text_input("Or enter your question:", placeholder="Ask anything from your ingested corpus...")

    active_query = ""
    if audio_record:
        with st.spinner("Transcribing speech via Groq Whisper..."):
            active_query = transcribe_audio_bytes(audio_record)
            st.info(f"🗣️ **Voice Input:** {active_query}")
    elif text_query:
        active_query = text_query

    if active_query:
        if "vectorstore" not in st.session_state:
            st.warning("Please ingest data sources in the sidebar first.")
        else:
            with st.spinner("Searching Vector DB and formulating response..."):
                retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 4})
                llm = ChatGroq(model_name=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are an academic researcher. Answer truthfully based on context:\n\nContext:\n{context}",
                        ),
                        ("human", "{input}"),
                    ]
                )

                if LEGACY_CHAIN_AVAILABLE:
                    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                    result = chain.invoke({"input": active_query})
                    raw_answer = result["answer"]
                    retrieved_chunks = result["context"]
                else:
                    def format_docs(docs):
                        return "\n\n".join(d.page_content for d in docs)

                    retrieved_chunks = retriever.invoke(active_query)
                    rag_lcel_chain = (
                        {"context": lambda x: format_docs(retrieved_chunks), "input": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    raw_answer = rag_lcel_chain.invoke(active_query)

                final_answer = translate_content(raw_answer, lang_code)
                st.markdown(f"#### 💡 Response ({selected_lang})")
                st.write(final_answer)
                tts_audio = text_to_speech(final_answer, lang_code)
                if tts_audio:
                    st.audio(tts_audio, format="audio/mp3")

                with st.expander("📚 Source Chunks Used"):
                    for idx, doc in enumerate(retrieved_chunks):
                        src = doc.metadata.get("source", "Unknown")
                        st.markdown(f"<span class='source-pill'>Chunk {idx+1}</span> `{src}`", unsafe_allow_html=True)
                        st.caption(doc.page_content[:300] + "...")
