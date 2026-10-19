"""DocuMind RAG: Streamlit UI and orchestration only.

All local-file storage, document processing, retrieval, and LLM code live in
backend/. Do not place LangChain/Ollama implementations in this file.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import streamlit as st

from backend.config import ALLOWED_EXTENSIONS, DEMO_MODE, UPLOAD_ROOT
from backend.file_storage import save_uploaded_files
from backend.rag_pipeline import create_rag_chain, generate_answer, process_documents


st.set_page_config(
    page_title="DocuMind RAG | Chat with Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def file_fingerprint(files) -> str | None:
    """Identify the current upload selection (names + file contents).

    This is a UI change-detection helper, NOT a vector-database operation.
    The fingerprint is recalculated on reruns, but files are SAVED only when
    this fingerprint differs from the last saved selection.
    """
    if not files:
        return None

    digest = hashlib.sha256()
    for uploaded_file in files:
        name = uploaded_file.name.encode("utf-8")
        content = uploaded_file.getvalue()
        # Include lengths so adjacent name/content fields cannot be confused.
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


# Streamlit reruns this script on UI interactions. Session state persists the
# upload selection, the saved paths, the retriever and the latest answer.
initial_state = {
    "upload_session_id": None,
    "uploaded_signature": None,
    "processed_signature": None,
    "saved_file_paths": [],
    "uploaded_names": [],
    "rag_retriever": None,
    "answer": None,
    "contexts": [],
    "answered_question": None,
}
for key, initial_value in initial_state.items():
    if key not in st.session_state:
        st.session_state[key] = initial_value

if st.session_state["upload_session_id"] is None:
    # Separate local folders for separate Streamlit browser sessions.
    st.session_state["upload_session_id"] = uuid4().hex


# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("📁 Your Documents")
    st.write("Upload files to build your document knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
        help="Supported: PDF, TXT, DOC, DOCX, PNG, JPG, JPEG.",
    )

    current_signature = file_fingerprint(uploaded_files)

    # IMPORTANT: This block runs only when the upload SELECTION CHANGES.
    # Clicking Get Answer, typing a question, or clicking Process does not
    # change the fingerprint, so the files will NOT be saved again.
    if current_signature != st.session_state["uploaded_signature"]:
        # Invalidate any index/results for the previous file selection.
        st.session_state["processed_signature"] = None
        st.session_state["rag_retriever"] = None
        st.session_state["rag_chain"] = None
        st.session_state["answer"] = None
        st.session_state["contexts"] = []
        st.session_state["answered_question"] = None
        st.session_state["uploaded_names"] = [f.name for f in uploaded_files]
        st.session_state["saved_file_paths"] = []

        if not uploaded_files:
            # Files were removed; there is nothing to save.
            st.session_state["uploaded_signature"] = None
        else:
            try:
                session_upload_dir = (
                    UPLOAD_ROOT / st.session_state["upload_session_id"]
                )
                # Filesystem-related code lives in backend/file_storage.py.
                saved_paths = save_uploaded_files(
                    uploaded_files, destination_dir=session_upload_dir
                )
                st.session_state["saved_file_paths"] = [
                    str(path) for path in saved_paths
                ]
                # Mark selection as saved ONLY after all writes succeed.
                st.session_state["uploaded_signature"] = current_signature
                st.toast("Uploaded files saved locally.", icon="📁")
            except (OSError, ValueError) as error:
                # Keep the old signature so the next rerun can retry saving.
                st.error(f"Failed to save uploaded files: {error}")

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} document(s) selected")
    else:
        st.caption("No documents selected yet")

    files_saved = (
        bool(uploaded_files)
        and current_signature == st.session_state["uploaded_signature"]
        and len(st.session_state["saved_file_paths"]) == len(uploaded_files)
    )

    process_clicked = st.button(
        "Process",
        type="primary",
        disabled=not files_saved,
        use_container_width=True,
    )

    if process_clicked:
        processing_notice = st.empty()
        # The previous index shouldn't be considered usable while reprocessing.
        st.session_state["processed_signature"] = None
        st.session_state["rag_retriever"] = None
        st.session_state["rag_chain"] = None
        st.session_state["answer"] = None
        st.session_state["contexts"] = []
        st.session_state["answered_question"] = None
        try:
            processing_notice.info(
                "Processing documents. This may take a few minutes. Please wait..."
            )
            with st.spinner("Extracting text, splitting chunks and storing vectors..."):
                # Convert paths to Path objects for LangChain loaders.
                paths = [Path(p) for p in st.session_state["saved_file_paths"]]
                # TODO implementation is in backend/rag_pipeline.py.
                retriever = process_documents(paths)
                print("NEW Retriever")
                print(retriever)
                rag_chain = create_rag_chain(retriever)

            # Record success only if backend returned without raising an error.
            st.session_state["rag_retriever"] = retriever
            st.session_state["rag_chain"] = rag_chain
            st.session_state["processed_signature"] = current_signature
            processing_notice.empty()
            if DEMO_MODE:
                st.toast("UI demo processing complete. You can test the question flow.", icon="✅")
            else:
                st.toast("Documents processed successfully! You can now ask questions.", icon="✅")
        except Exception as error:
            processing_notice.empty()
            st.session_state["processed_signature"] = None
            st.session_state["rag_retriever"] = None
            st.session_state["rag_chain"] = None
            st.error(f"Document processing failed: {error}")

    is_processed = (
        files_saved
        and st.session_state["processed_signature"] == current_signature
        and st.session_state["rag_retriever"] is not None
        and st.session_state["rag_chain"] is not None
    )

    if is_processed:
        st.success("✅ Documents ready for questions")
    elif uploaded_files and files_saved:
        st.warning("Click Process to prepare your documents.")
    elif uploaded_files:
        st.warning("Could not save files. Upload again or retry.")
    else:
        st.info("Start by uploading one or more documents.")

    st.divider()
    if DEMO_MODE:
        st.caption("🧪 UI demo mode — no RAG backend connected")
    st.caption("Powered by Streamlit · LangChain · Ollama")


# ------------------------- MAIN SCREEN -------------------------
st.title("📚 DocuMind RAG")
st.markdown(
    "**Chat with your documents using AI.** Upload PDFs, Word files, "
    "text files, or images in the sidebar, process them, and ask questions. "
    "Once you connect your RAG pipeline, answers will be grounded in "
    "relevant content retrieved from your files."
)
st.divider()

st.subheader("Ask Your Question")
question = st.text_area(
    "Your question",
    placeholder="e.g., Summarize the key findings from my uploaded report...",
    height=130,
    label_visibility="collapsed",
    key="question_input",
)

ask_clicked = st.button("Get Answer", type="primary", disabled=not is_processed)
if not is_processed:
    st.caption("🔒 Upload and process documents to enable Get Answer.")


# ------------------------ ANSWER SECTION ------------------------
if ask_clicked:
    if not question.strip():
        st.warning("Please type a question before clicking Get Answer.")
    else:
        try:
            with st.spinner("Thinking... retrieving context and generating your answer..."):
                # TODO implementation is in backend/rag_pipeline.py.
                answer, contexts = generate_answer(
                    question.strip(), st.session_state["rag_retriever"],st.session_state["rag_chain"]
                )
                print(f"ANSWER : {answer}" )
                print(f"CONTEXT : {contexts}")
            st.session_state["answer"] = answer
            st.session_state["contexts"] = contexts or []
            st.session_state["answered_question"] = question.strip()
        except Exception as error:
            st.session_state["answer"] = None
            st.session_state["contexts"] = []
            st.session_state["answered_question"] = None
            st.error(f"Unable to generate an answer: {error}")

if st.session_state["answer"] is not None:
    st.divider()
    st.subheader("Answer")
    st.markdown(st.session_state["answer"])

    with st.expander("Show Context", expanded=False):
        contexts = st.session_state["contexts"]
        if not contexts:
            st.info("No context chunks were returned.")
        else:
            st.caption(f"{len(contexts)} retrieved chunk(s)")
            for chunk_number, chunk in enumerate(contexts, start=1):
                st.markdown(f"**Chunk {chunk_number}**")
                source = chunk.get("source", "Unknown source")
                page = chunk.get("page")
                page_label = f" · Page: {page}" if page is not None else ""
                st.caption(f"Source: {source}{page_label}")
                st.write(chunk.get("content", ""))
                if chunk_number < len(contexts):
                    st.divider()
