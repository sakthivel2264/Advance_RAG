
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

DATA_DIR = Path("data/raw")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


def load_single_document(file_path: Path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    elif extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    # Attach consistent source metadata
    for document in documents:
        document.metadata["source"] = str(
            file_path.as_posix()
        )
        document.metadata["file_type"] = extension

    return documents


def load_all_documents():

    all_documents = []

    for file_path in sorted(DATA_DIR.rglob("*")):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        print(f"Loading: {file_path}")

        documents = load_single_document(file_path)

        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    documents = load_all_documents()

    print("\n========== INGESTION SUMMARY ==========")

    print(f"Total document objects: {len(documents)}")

    for index, document in enumerate(documents):

        print(f"\nDocument object: {index + 1}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Metadata: {document.metadata}")
        print(f"Text preview: {document.page_content[:300]}")
