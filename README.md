# Notebook LM

A local, document-grounded question-answering system inspired by [Google NotebookLM](https://notebooklm.google.com). Ingest PDFs into a hybrid vector index, then ask questions in an interactive CLI and receive answers backed by retrieved chunks with source citations.

All inference runs on your machine: embeddings and generation use [Ollama](https://ollama.com), and vectors are stored in an embedded [Qdrant](https://qdrant.tech) database under `storage/qdrant`. No cloud API keys are required for the default setup.

## Features

- **PDF ingestion** — Load one or more PDFs, split them into chunks, and index metadata (filename, page, document ID).
- **Hybrid retrieval** — Combine dense embeddings with sparse BM25 (`Qdrant/bm25`) for better recall on keyword-heavy queries.
- **Pluggable chunking** — `recursive` (size/overlap) or `semantic` (sentence similarity via spaCy + Sentence Transformers).
- **Pluggable embeddings** — Ollama (default) or Hugging Face sentence-transformers.
- **Cited answers** — Jinja2 prompts instruct the LLM to cite sources as `[S1]`, `[S2]`, etc.; only markers used in the answer are shown.
- **Configurable** — Environment variables and an optional `.env` file (see [Configuration](#configuration)).

## Architecture

```text
PDF files (data/)
       │
       ▼
  Chunking (recursive | semantic)
       │
       ▼
  Dense + sparse embeddings
       │
       ▼
  Qdrant (local, storage/qdrant)
       │
       ▼
  Hybrid similarity search → top-k chunks
       │
       ▼
  Ollama LLM (prompt + context) → answer + citations
```

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.12+** | See `.python-version` |
| **[uv](https://docs.astral.sh/uv/)** | Recommended for installing dependencies |
| **[Ollama](https://ollama.com)** | Running locally for embeddings and chat |
| **spaCy model** | Required only if you use semantic chunking: `en_core_web_sm` |

### Ollama models (defaults)

Pull the models referenced by the project defaults before first ingest or query:

```bash
ollama pull qwen3-embedding:8b
ollama pull qwen2.5:7b
```

You can use other models by setting `RAG_EMBEDDING_MODEL` and changing the model name in `src/llm.py` (`invoke_llm`).

## Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd notebook_lm
uv sync
```

### 2. Add documents

Place PDF files in the `data/` directory (or any path you pass to `/ingest`).

## Usage

Start the interactive CLI:

```bash
uv run python main.py
```

### Commands

| Input | Description |
|-------|-------------|
| `/help` | List available commands |
| `/ingest <files...> [options]` | Index PDF(s) into Qdrant |
| `/bye` | Exit |
| *any other text* | Ask a question (RAG answer over indexed chunks) |

### Ingest examples

```text
/ingest data/manual.pdf
```

```text
/ingest data/a.pdf data/b.pdf --chunker recursive --chunk-size 1000 --chunk-overlap 150
```

```text
/ingest data/report.pdf --chunker semantic
```

After ingestion completes, type a natural-language question at the `>>>` prompt:

```text
What are the main policies described in the HR document?
```

The CLI prints the answer and, when applicable, citation lines (`S1`, `S2`, …) with filename and page.

## Configuration

Settings are defined in `src/config.py` and loaded via `pydantic-settings`. Environment variables and `.env` entries use the `RAG_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DATA_DIR` | `data` | Default location for source PDFs |
| `RAG_STORAGE_DIR` | `storage/qdrant` | Local Qdrant data path |
| `RAG_QDRANT_COLLECTION` | `rag_chunks` | Collection name |
| `RAG_CHUNKING_STRATEGY` | `semantic` | Default strategy in settings (`recursive` or `semantic`) |
| `RAG_CHUNK_SIZE` | `1000` | Recursive chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `150` | Recursive chunk overlap |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per question |
| `RAG_EMBEDDING` | `ollama` | `ollama` or `huggingface` |
| `RAG_EMBEDDING_MODEL` | `qwen3-embedding:8b` | Model name for the selected embedding backend |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reserved for future reranking |

**Note:** `/ingest` defaults to the `recursive` chunker unless you pass `--chunker` or align settings via `.env`.

## Project structure

```text
notebook_lm/
├── main.py                 # Interactive CLI entry point
├── cli/
│   ├── command_handler.py  # Slash commands (/help, /ingest, /bye)
│   └── rag_answer.py       # RAG Q&A formatting
├── src/
│   ├── config.py           # Settings
│   ├── ingest.py           # PDF load, chunk, index
│   ├── rag.py              # Retrieve, prompt, answer
│   ├── store.py            # Qdrant client and hybrid store
│   ├── llm.py              # Ollama chat
│   ├── chunking/           # Recursive and semantic strategies
│   ├── embeddings/         # Ollama and Hugging Face backends
│   └── prompts/            # Jinja2 templates
├── data/                   # Source PDFs (gitignored content typical)
└── storage/qdrant/       # Local vector database (generated)
```

## Troubleshooting

- **Ollama connection errors** — Ensure `ollama serve` is running and models are pulled (`ollama list`).
- **Empty or weak answers** — Run `/ingest` on the relevant PDFs first; try increasing `RAG_TOP_K` or using a different chunker.
- **Re-indexing from scratch** — Remove `storage/qdrant` or use ingest with collection recreation in code if you extend the CLI; a fresh folder forces a new local index.
