
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingestion.document_loader import load_all_documents
from app.ingestion.chunker import split_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def generate_document_embeddings(chunks, embedding_model):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embedding_model.embed_documents(texts)

    return vectors


def main():

    # Step 1: Load documents
    documents = load_all_documents()

    # Step 2: Split documents
    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    # Step 3: Initialize embedding model
    embedding_model = get_embedding_model()

    # Step 4: Generate embeddings
    vectors = generate_document_embeddings(
        chunks,
        embedding_model
    )

    print("\n========== EMBEDDING SUMMARY ==========")

    print(f"Total chunks: {len(chunks)}")
    print(f"Total vectors: {len(vectors)}")
    print(f"Embedding dimensions: {len(vectors[0])}")

    for index, vector in enumerate(vectors):

        print(f"\nChunk {index + 1}")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 10 values: {vector[:10]}")

    # Step 5: Embed a sample query
    query = "How many annual leave days can an employee take?"

    query_vector = embedding_model.embed_query(query)

    print("\n========== QUERY EMBEDDING ==========")
    print(f"Query: {query}")
    print(f"Query vector dimensions: {len(query_vector)}")

    # Step 6: Compare query against document chunks
    print("\n========== SIMILARITY RESULTS ==========")

    for index, vector in enumerate(vectors):

        # Both vectors are normalized
        similarity = sum(
            a * b
            for a, b in zip(query_vector, vector)
        )

        print(
            f"Chunk {index + 1}: "
            f"Similarity = {similarity:.4f}"
        )


if __name__ == "__main__":
    main()
