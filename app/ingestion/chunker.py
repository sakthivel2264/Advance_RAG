
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.document_loader import load_all_documents


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    documents = load_all_documents()

    chunks = split_documents(
        documents,
        chunk_size=300,
        chunk_overlap=50,
    )

    print("\n========== CHUNKING SUMMARY ==========")

    print(f"Original document objects: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")

    for index, chunk in enumerate(chunks):

        print(f"\n========== CHUNK {index + 1} ==========")

        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Start index: {chunk.metadata.get('start_index')}")
        print(f"Character count: {len(chunk.page_content)}")

        print("\nContent:")
        print(chunk.page_content)
