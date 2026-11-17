
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from app.ingestion.document_loader import load_all_documents
from app.ingestion.chunker import split_documents
from app.ingestion.embeddings import (
    get_embedding_model,
    generate_document_embeddings,
)


load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
API_KEY = os.getenv("PINECONE_API_KEY")

NAMESPACE = "employee-policy-v2"


def get_pinecone_index():

    if not API_KEY:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    if not INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is missing from .env")

    pc = Pinecone(api_key=API_KEY)

    return pc.index(INDEX_NAME)


def generate_record_id(source, start_index, content):

    unique_text = f"{source}:{start_index}:{content}"

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


def prepare_records(chunks, vectors):

    records = []

    if len(chunks) != len(vectors):
        raise ValueError("Number of chunks and vectors does not match")

    for chunk, vector in zip(chunks, vectors):

        source = chunk.metadata.get("source", "unknown")
        start_index = chunk.metadata.get("start_index", 0)

        record = {
            "_id": generate_record_id(
                source,
                start_index,
                chunk.page_content
            ),
            "chunk_text": chunk.page_content,
            "title": Path(source).stem,
            "embedding": [float(value) for value in vector],
            "source": source,
            "file_type": chunk.metadata.get("file_type", "unknown"),
            "start_index": int(start_index or 0),
        }

        records.append(record)

    return records


def upload_records(index, records):

    response = index.documents.upsert(
        namespace=NAMESPACE,
        documents=records,
    )

    return response


def main():

    print("Loading documents...")

    documents = load_all_documents()

    print(f"Loaded documents: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    print(f"Generated chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    embedding_model = get_embedding_model()

    print("\nGenerating embeddings...")

    vectors = generate_document_embeddings(
        chunks,
        embedding_model,
    )

    print(f"Generated vectors: {len(vectors)}")

    for vector in vectors:
        if len(vector) != 384:
            raise ValueError(
                f"Expected 384 dimensions, got {len(vector)}"
            )

    print("\nPreparing Pinecone records...")

    records = prepare_records(chunks, vectors)

    print(f"Records prepared: {len(records)}")

    print("\nConnecting to Pinecone...")

    index = get_pinecone_index()

    print(f"Connected to index: {INDEX_NAME}")

    print("\nUploading records...")

    response = upload_records(index, records)

    print("\n========== PINECONE UPLOAD SUMMARY ==========")

    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Records uploaded: {response.upserted_count}")

    print("\nUpload request completed.")


if __name__ == "__main__":
    main()
