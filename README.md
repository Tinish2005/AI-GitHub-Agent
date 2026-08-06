# AI GitHub Agent

An AI agent that understands GitHub repositories. Paste a repo URL and it indexes the code with Google Gemini, then answers questions with citations, plans multi-step work, and proposes code fixes grounded in the actual repository.

Built with a **Planner – Executor – Validator** architecture, retrieval-augmented generation (RAG), and the Model Context Protocol (MCP).

---

## Features

- **Index a Repo** – fetch any public GitHub repo via the GitHub API (no local Git needed) and index it.
- **Ask** – retrieval-augmented Q&A over the indexed code, with source citations.
- **Plan** – turn a plain-English goal into a structured multi-step plan.
- **Propose Fix** – generate a code fix as a unified diff, grounded in the real repo code, then validate it in a sandbox.
- **Execute** – run a plan end-to-end with per-step reporting.
- **MCP server** – core tools exposed over JSON-RPC 2.0 for agent interoperability.

---

## Architecture

**Two pipelines:**

- **Offline (indexing):** GitHub repo → fetch files via API → AST parsing into chunks → metadata extraction (hashes, imports, calls) → Gemini embeddings → vector store.
- **Online (runtime):** user request → planner → executor → semantic retrieval + Gemini reasoning → grounded answer / plan / fix → validation → draft pull request (human-reviewed).

Every proposed change lands as a **draft PR reviewed by a human** — autonomous but safe.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, pydantic |
| Frontend | React, Vite |
| LLM + Embeddings | Google Gemini |
| Retrieval | Pure-Python vector store (NumPy, cosine similarity) |
| Code analysis | Python AST |
| Interop | Model Context Protocol (MCP), JSON-RPC 2.0 |
| GitHub access | GitHub REST API |
| Testing | pytest (318 tests) |

---

## Quick Start

### Backend
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    cp .env.example .env
    uvicorn backend.main:app

### Frontend
    cd frontend
    npm install
    npm run dev

Then open http://localhost:5173 in your browser.

### Run the tests
    pytest -v

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | /health | Health check |
| POST | /qa | Ask a question about the indexed repo |
| POST | /plan | Generate a multi-step plan |
| POST | /execute | Run a plan end-to-end |
| POST | /fix/propose | Propose a code fix as a diff |
| POST | /fix/validate | Validate a proposed fix in a sandbox |
| POST | /fix/pr | Prepare a draft pull request |
| POST | /index/github | Index a GitHub repo via the API |
| POST | /mcp | MCP JSON-RPC endpoint |

Interactive API docs are auto-generated at http://localhost:8000/docs.

---

## Configuration

Create a `.env` file (see `.env.example`):

    GEMINI_API_KEY=your-gemini-key
    GITHUB_TOKEN=your-github-token

Keys are wrapped in SecretStr and never committed (.env is gitignored).

---

## Roadmap

- One-click end-to-end auto-fix flow (paste URL → fix → draft PR)
- Replanning on step failure
- Multi-file fixes and richer validation (lint, tests, security)

---

## License

Personal project — for learning and portfolio use.