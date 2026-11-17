
from pathlib import Path

from app.ingestion.document_loader import load_single_document
from app.ingestion.chunker import split_documents
from app.ingestion.embeddings import (
    get_embedding_model,
    generate_document_embeddings,
)
from app.retrieval.vector_store import (
    get_pinecone_index,
    prepare_records,
    upload_records,
    generate_record_id,
)


def ingest_uploaded_document(file_path: Path):
    """
    Load, chunk, embed and index one uploaded document.
    """

    # 1. Load document
    documents = load_single_document(file_path)

    if not documents:
        raise ValueError("No readable content found in document")

    # 2. Split into chunks
    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("No chunks generated from document")

    # 3. Generate embeddings
    embedding_model = get_embedding_model()

    vectors = generate_document_embeddings(
        chunks,
        embedding_model,
    )

    # 4. Validate embedding dimensions
    for vector in vectors:
        if len(vector) != 384:
            raise ValueError(
                f"Expected 384 dimensions, got {len(vector)}"
            )

    # 5. Prepare Pinecone records
    records = prepare_records(chunks, vectors)

    # 6. Upload to Pinecone
    index = get_pinecone_index()

    response = upload_records(index, records)

    return {
        "filename": file_path.name,
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "vectors_generated": len(vectors),
        "records_uploaded": response.upserted_count,
        "namespace": "employee-policy-v2",
    }



def delete_uploaded_document(file_path: Path):
    """Delete all Pinecone records belonging to an uploaded document."""

    documents = load_single_document(file_path)

    if not documents:
        raise ValueError("Could not load the uploaded document.")

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("No chunks found for this document.")

    record_ids = []

    for chunk in chunks:
        source = chunk.metadata["source"]
        start_index = chunk.metadata["start_index"]
        content = chunk.page_content

        record_id = generate_record_id(
            source,
            start_index,
            content,
        )

        record_ids.append(record_id)

    index = get_pinecone_index()

    response = index.documents.delete(
        namespace="employee-policy-v2",
        ids=record_ids,
    )

    return {
        "records_requested_for_deletion": len(record_ids),
        "namespace": "employee-policy-v2",
    }
