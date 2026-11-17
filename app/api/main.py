
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, File
from app.ingestion.pipeline import (
    ingest_uploaded_document,
    delete_uploaded_document as delete_document_from_pinecone,
)
from app.ingestion.document_loader import SUPPORTED_EXTENSIONS
from app.orchestration.rag_graph import rag_app
from app.ingestion.document_registry import (
    calculate_file_hash,
    is_duplicate,
    register_document,
    list_documents,
    get_document,
    load_registry,
    save_registry,
)


api = FastAPI(
    title="Evaluated RAG Assistant",
    description="Production-style RAG API powered by LangGraph, Pinecone and OpenRouter",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG assistant",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    sources: list


@api.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "evaluated-rag-assistant",
    }


UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@api.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    extension = Path(file.filename or "").suffix.lower()

    # Validate file type
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are supported.",
        )

    # Read uploaded file
    content = await file.read()

    # Generate SHA-256 checksum
    file_hash = calculate_file_hash(content)

    # Check for duplicate document
    if is_duplicate(file_hash):
        raise HTTPException(
            status_code=409,
            detail="This document has already been uploaded.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 10 MB.",
        )

    # Generate a unique filename to prevent overwrites
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / safe_filename

    try:
        file_path.write_bytes(content)

        result = ingest_uploaded_document(file_path)

        # Register successfully indexed document
        register_document(
            file_hash=file_hash,
            filename=file.filename,
            stored_filename=safe_filename,
            chunks_created=result["chunks_created"],
        )

        return {
            "status": "success",
            "message": "Document uploaded and indexed successfully.",
            "original_filename": file.filename,
            "file_hash": file_hash,
            "ingestion": result,
        }

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from error

    finally:
        await file.close()



@api.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    try:
        result = rag_app.invoke({
            "question": request.question,
            "documents": [],
            "evidence_status": "",
            "answer": "",
            "sources": [],
            "faithfulness_status": "",
            "final_status": "",
        })

        return AskResponse(
            question=request.question,
            answer=result.get("answer", ""),
            status=result.get("final_status", "unknown"),
            sources=result.get("sources", []),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        ) from error

@api.get("/documents")
def get_uploaded_documents():
    documents = list_documents()

    return {
        "total_documents": len(documents),
        "documents": documents,
    }


@api.get("/documents/{file_hash}")
def get_uploaded_document(file_hash: str):
    document = get_document(file_hash)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document


@api.delete("/documents/{file_hash}")
def delete_uploaded_document_api(file_hash: str):
    registry = load_registry()

    document = registry.get(file_hash)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    stored_filename = document["stored_filename"]
    file_path = UPLOAD_DIR / stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file is missing. Cannot safely regenerate Pinecone IDs."
        )

    try:
        # Delete document chunks from Pinecone
        result = delete_document_from_pinecone(file_path)

        # Delete the local uploaded file
        file_path.unlink()

        # Remove document from registry
        registry.pop(file_hash)
        save_registry(registry)

        return {
            "status": "success",
            "message": "Document deletion requested successfully.",
            "original_filename": document["original_filename"],
            "records_requested_for_deletion": result[
                "records_requested_for_deletion"
            ],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed."
        ) from error

