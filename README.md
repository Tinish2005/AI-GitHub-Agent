# AI GitHub Agent

An AI agent that understands GitHub repositories. Paste a repository URL and it indexes the code with Google Gemini, then answers questions with citations, creates multi-step plans, executes repository analysis workflows, and proposes code fixes grounded in the actual codebase.

Built with a **Planner → Executor → Validator** architecture, Retrieval Augmented Generation (RAG), and the Model Context Protocol (MCP).

---

# 🌐 Live Demo

## Frontend

https://ai-git-hub-agent.vercel.app/

## Backend API

https://ai-github-agent-2q87.onrender.com

## API Documentation

https://ai-github-agent-2q87.onrender.com/docs

---

# Features

- Repository Indexing
- Retrieval Augmented Generation (RAG)
- Ask the Codebase
- Planning Engine
- Execution Engine
- Fix Generator
- Validation Pipeline
- Draft Pull Request Generation
- GitHub API Integration
- Gemini Integration
- React Frontend
- FastAPI Backend
- MCP Support

---

# Core Capabilities

## Index a Repository

Fetch any public GitHub repository through the GitHub API and build a repository knowledge base.

## Ask

Ask questions about the codebase and receive repository-grounded answers with citations.

## Plan

Convert a natural language goal into a structured multi-step execution plan.

## Execute

Run repository analysis workflows end-to-end and generate actionable findings.

## Propose Fix

Generate repository-aware code fixes in unified diff format.

## Validate Fix

Validate generated fixes before creating a pull request.

## Draft Pull Request

Prepare draft pull requests for human review and approval.

---

# Architecture

## Offline Pipeline

```text
GitHub Repository
        ↓
GitHub API
        ↓
AST Parsing
        ↓
Chunk Creation
        ↓
Metadata Extraction
        ↓
Gemini Embeddings
        ↓
Vector Store
```

## Runtime Pipeline

```text
User Request
        ↓
Planner
        ↓
Executor
        ↓
Repository Retrieval
        ↓
Gemini Reasoning
        ↓
Grounded Response
        ↓
Validation
        ↓
Draft Pull Request
```

---

# Tech Stack

| Layer | Technology |
|---------|---------|
| Backend | Python, FastAPI, Pydantic |
| Frontend | React, Vite |
| LLM | Google Gemini |
| Embeddings | Gemini Embeddings |
| Retrieval | Vector Store (NumPy + Cosine Similarity) |
| Parsing | Python AST |
| GitHub Access | GitHub REST API |
| Agent Architecture | Planner → Executor → Validator |
| Protocol | MCP (Model Context Protocol) |
| Testing | Pytest |

---

# Quick Start

## Backend

```bash
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

# Available API Endpoints

| Method | Endpoint |
|----------|----------|
| GET | /health |
| POST | /qa |
| POST | /plan |
| POST | /execute |
| POST | /fix/propose |
| POST | /fix/validate |
| POST | /fix/pr |
| POST | /index/github |
| POST | /mcp |

---

# Environment Variables

Backend:

```env
GEMINI_API_KEY=your_key

GITHUB_TOKEN=your_token
```

Frontend Local:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Frontend Production:

```env
VITE_API_URL=https://ai-github-agent-2q87.onrender.com
```

---

# Current Status

✅ Repository Indexing

✅ Ask

✅ Plan

✅ Execute

✅ Fix Generation

✅ Validation

✅ Draft Pull Request

✅ React Frontend

✅ FastAPI Backend

✅ Vercel Deployment

✅ Render Deployment

✅ Gemini Integration

✅ RAG Pipeline

---

# Live Links

Frontend:

https://ai-git-hub-agent.vercel.app/

Backend:

https://ai-github-agent-2q87.onrender.com

API Docs:

https://ai-github-agent-2q87.onrender.com/docs

---

# License

Personal portfolio and learning project.