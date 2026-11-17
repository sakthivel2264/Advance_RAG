
import hashlib
import json

from pathlib import Path


REGISTRY_PATH = Path("data/metadata/document_registry.json")

def list_documents() -> list:
    """Return all registered uploaded documents."""

    registry = load_registry()

    documents = []

    for file_hash, details in registry.items():
        documents.append({
            "file_hash": file_hash,
            **details,
        })

    return documents


def calculate_file_hash(content: bytes) -> str:
    """Generate SHA-256 checksum of the original file."""

    return hashlib.sha256(content).hexdigest()


def load_registry() -> dict:
    """Load previously uploaded document records."""

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PATH.exists():
        return {}

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry: dict):
    """Persist document records."""

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


def is_duplicate(file_hash: str) -> bool:
    registry = load_registry()
    return file_hash in registry


def register_document(
    file_hash: str,
    filename: str,
    stored_filename: str,
    chunks_created: int,
):
    registry = load_registry()

    registry[file_hash] = {
        "original_filename": filename,
        "stored_filename": stored_filename,
        "chunks_created": chunks_created,
    }

    save_registry(registry)


def get_document(file_hash: str) -> dict | None:
    """Retrieve a document using its SHA-256 hash."""

    registry = load_registry()

    document = registry.get(file_hash)

    if document is None:
        return None

    return {
        "file_hash": file_hash,
        **document,
    }

