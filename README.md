# Evaluated RAG Assistant

An evidence-grounded employee-policy question-answering system built with
LangGraph, LangChain, OpenRouter, Pinecone, and Streamlit. The project
combines dense semantic retrieval with BM25 keyword retrieval, evaluates
whether the retrieved evidence is sufficient, and verifies generated answers
before returning them.

The policy files in this repository are synthetic documents for engineering
and RAG evaluation experiments.

## Features

- PDF, DOCX, and TXT document loading
- Configurable document chunking and local embedding generation
- Pinecone vector and full-text indexing
- Hybrid retrieval using dense search, BM25, and Reciprocal Rank Fusion
- Evidence grading before answer generation
- Faithfulness verification after answer generation
- Safe abstention when evidence is missing or an answer cannot be verified
- FastAPI endpoints for questions and document management
- Streamlit UI for questions, uploads, indexing, and deletion
- Retrieval evaluation with Precision, Recall, Hit Rate, and MRR
- RAGAS evaluation for faithfulness, relevancy, correctness, context precision,
  and context recall

## Architecture

```text
Documents
   |
   v
Loaders (PDF / DOCX / TXT)
   |
   v
Chunking (500 characters, 100-character overlap)
   |
   v
all-MiniLM-L6-v2 embeddings (384 dimensions)
   |
   v
Pinecone namespace: employee-policy-v2
   |
   +--> Dense semantic retrieval
   |
   +--> BM25 full-text retrieval
             |
             v
      Reciprocal Rank Fusion
             |
             v
      Evidence grading
        /         \
  insufficient   sufficient
      |             |
   Abstain       Generate answer
                      |
                      v
              Faithfulness grading
                /             \
           faithful       unfaithful
              |               |
            Answer          Abstain
```

The default retriever returns the top three fused results. Dense and BM25
searches each retrieve up to ten candidates before fusion.

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)
- An OpenRouter API key
- A Pinecone API key
- A Pinecone index configured for the project

The project currently requires Python `>=3.14`, as specified in
[`pyproject.toml`](./pyproject.toml).

## Installation

Clone the repository and install the locked dependencies:

```bash
uv sync
```

Create a local `.env` file in the project root with your own credentials:

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index-name
```

Never commit `.env` or API keys. Use placeholders in shared configuration
examples and rotate any credential that may have been exposed.

## Policy documents

The synthetic employee-policy collection is organized as follows:

```text
data/raw/employee_policies/
├── finance/
│   ├── payroll.txt
│   └── travel_reimbursement.txt
├── hr/
│   ├── holidays.txt
│   └── resignation.txt
├── leave/
│   ├── annual_leave.txt
│   ├── casual_leave.txt
│   └── sick_leave.txt
└── workplace/
    ├── attendance.txt
    ├── work_from_home.txt
    └── working_hours.txt
```

Additional supported documents can be placed under `data/raw/`. The loader
recursively discovers `.pdf`, `.docx`, and `.txt` files.

## Ingesting the policy collection

To inspect documents discovered by the loader:

```bash
uv run python -m app.ingestion.document_loader
```

To chunk, embed, and upload all documents under `data/raw/` to Pinecone:

```bash
uv run python -m app.retrieval.vector_store
```

The indexing process uses the `employee-policy-v2` namespace and validates that
each generated embedding has 384 dimensions.

## Running the Streamlit UI

Start the user interface from the project root:

```bash
uv run streamlit run streamlitui.py
```

The UI provides:

1. A question box that runs the full RAG graph.
2. The verified answer and final pipeline status.
3. Source references for accepted answers.
4. A sidebar for uploading and indexing PDF, DOCX, and TXT files.
5. A list of uploaded documents and controls to delete them.

Uploaded files are stored in `data/uploads/`. Their metadata and SHA-256
checksums are stored in `data/metadata/document_registry.json`.

## Running the FastAPI service

Start the API with Uvicorn:

```bash
uv run uvicorn app.api.main:api --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive OpenAPI
documentation is available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service health |
| `POST` | `/ask` | Ask a question using the RAG graph |
| `POST` | `/upload` | Upload and index a PDF, DOCX, or TXT file |
| `GET` | `/documents` | List registered uploaded documents |
| `GET` | `/documents/{file_hash}` | Retrieve one registry entry |
| `DELETE` | `/documents/{file_hash}` | Remove an uploaded document and its indexed chunks |

Example question request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How many annual leave days are available each year?"}'
```

Example health request:

```bash
curl http://127.0.0.1:8000/health
```

Uploads are limited to 10 MB and duplicate files are rejected using their
SHA-256 checksum.

## Running evaluations

The evaluation dataset is stored at
[`data/evaluation/rag_test_dataset.json`](./data/evaluation/rag_test_dataset.json).
It contains answerable and intentionally unanswerable questions, expected
answers, and relevant document identifiers.

### Retrieval benchmark

Compare dense, BM25, and hybrid retrieval:

```bash
uv run python -m app.evaluation.run_retrieval_evaluation
```

The benchmark reports:

- Precision@3
- Precision over the actual retrieved results
- Recall@3
- Hit Rate@3
- Mean Reciprocal Rank

### RAGAS evaluation

Run answer-quality and answerability evaluation:

```bash
uv run python -m app.evaluation.run_ragas_evaluation
```

RAGAS evaluation requires the configured OpenRouter model and reports metrics
for answerable questions. The evaluation also measures answerability accuracy,
false abstention rate, and unsupported answer rate.

## Project layout

```text
.
├── app/
│   ├── api/                 # FastAPI application
│   ├── evaluation/          # Retrieval and RAGAS evaluation runners
│   ├── ingestion/           # Loaders, chunking, embeddings, registry
│   ├── orchestration/       # LangGraph workflow and generation logic
│   └── retrieval/           # Pinecone access and hybrid retrieval
├── data/
│   ├── evaluation/          # Evaluation questions and ground truth
│   ├── metadata/            # Uploaded-document registry
│   ├── raw/                 # Source policy documents
│   └── uploads/             # Runtime-uploaded files
├── streamlitui.py           # Streamlit application
├── pyproject.toml           # Project metadata and dependencies
└── uv.lock                  # Reproducible dependency lock file
```

## RAG pipeline behavior

The LangGraph workflow follows these stages:

1. Retrieve the top hybrid-search results.
2. Grade whether the context contains enough explicit evidence.
3. Abstain if the evidence is insufficient.
4. Generate a concise answer using only the retrieved context.
5. Verify the generated answer for faithfulness.
6. Return the answer only when it passes verification; otherwise abstain.

The answer status is one of the following:

- `answered`
- `abstained_insufficient_evidence`
- `abstained_unfaithful_answer`

Sources are returned with numbered references such as `[1]`, along with the
source path and Reciprocal Rank Fusion score.

## Development notes

- Run commands from the repository root so relative data paths resolve
  correctly.
- Do not commit credentials, uploaded documents, generated registries, or
  local runtime artifacts.
- Keep Pinecone index configuration consistent with the 384-dimensional
  `all-MiniLM-L6-v2` embeddings.
- Re-index documents after changing source files or chunking behavior.
- The application depends on external OpenRouter and Pinecone services for
  generation and retrieval.

## Troubleshooting

### Missing environment variables

If retrieval or generation fails with a configuration error, confirm that
`.env` exists in the project root and contains:

```text
OPENROUTER_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

### No search results

Confirm that the documents were indexed into the
`employee-policy-v2` namespace and that the configured Pinecone index contains
the expected records.

### Upload fails

Check that the file is a non-empty PDF, DOCX, or TXT file smaller than 10 MB.
An identical file cannot be uploaded twice because duplicate content is
detected by checksum.