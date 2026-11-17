from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.ingestion.document_loader import SUPPORTED_EXTENSIONS
from app.ingestion.document_registry import (
    calculate_file_hash,
    is_duplicate,
    list_documents,
    load_registry,
    register_document,
    save_registry,
)
from app.ingestion.pipeline import (
    delete_uploaded_document,
    ingest_uploaded_document,
)
from app.orchestration.rag_graph import rag_app


st.set_page_config(
    page_title="Evaluated RAG Assistant",
    page_icon="📚",
    layout="wide",
)

UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024


def ask_assistant(question: str) -> dict:
    return rag_app.invoke(
        {
            "question": question,
            "documents": [],
            "evidence_status": "",
            "answer": "",
            "sources": [],
            "faithfulness_status": "",
            "final_status": "",
        }
    )


def upload_document(uploaded_file) -> dict:
    extension = Path(uploaded_file.name).suffix.lower()
    content = uploaded_file.getvalue()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, DOCX and TXT files are supported.")
    if not content:
        raise ValueError("The selected file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Maximum upload size is 10 MB.")

    file_hash = calculate_file_hash(content)
    if is_duplicate(file_hash):
        raise ValueError("This document has already been uploaded.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(content)
        result = ingest_uploaded_document(file_path)
        register_document(
            file_hash=file_hash,
            filename=uploaded_file.name,
            stored_filename=stored_filename,
            chunks_created=result["chunks_created"],
        )
        return result
    except (OSError, ValueError, RuntimeError):
        file_path.unlink(missing_ok=True)
        raise


def remove_document(file_hash: str) -> str:
    registry = load_registry()
    document = registry.get(file_hash)

    if document is None:
        raise ValueError("Document not found.")

    file_path = UPLOAD_DIR / document["stored_filename"]
    if not file_path.is_file():
        raise ValueError("Uploaded file is missing.")

    result = delete_uploaded_document(file_path)
    file_path.unlink()
    registry.pop(file_hash)
    save_registry(registry)
    return str(result["records_requested_for_deletion"])


st.title("Evaluated RAG Assistant")
st.caption("Ask questions against the indexed employee policy collection.")

with st.sidebar:
    st.header("Document management")
    uploaded_file = st.file_uploader(
        "Upload a policy document",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    )

    if st.button("Upload and index", disabled=uploaded_file is None):
        try:
            result = upload_document(uploaded_file)
            st.success(
                f"Indexed {uploaded_file.name} ({result['chunks_created']} chunks)."
            )
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))

    st.divider()
    st.subheader("Uploaded documents")
    documents = list_documents()
    if not documents:
        st.info("No uploaded documents yet.")
    else:
        for document in documents:
            st.write(document["original_filename"])
            if st.button(
                "Delete",
                key=f"delete-{document['file_hash']}",
            ):
                try:
                    records_deleted = remove_document(document["file_hash"])
                    st.success(f"Deleted {records_deleted} indexed records.")
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(str(error))

question = st.text_area(
    "Question",
    placeholder="How many annual leave days can an employee take?",
    height=100,
)

if st.button("Ask assistant", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving and verifying an answer..."):
        try:
            result = ask_assistant(question.strip())
            st.subheader("Answer")
            st.write(result.get("answer", "No answer was returned."))
            st.caption(f"Status: {result.get('final_status', 'unknown')}")

            sources = result.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(
                            f"{source.get('reference', '')} "
                            f"{source.get('source', '')}"
                        )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Unable to answer the question: {error}")
