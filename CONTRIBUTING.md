# Contributing to Agentic Customer Support RAG

Thank you for your interest in contributing! This project is an enterprise-grade agentic customer-support system combining LangGraph, FastAPI, Pinecone, Tavily, and React 19.

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 20+ and npm

### Backend Setup
1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/tilakchow/agentic-customer-support-rag.git
   cd agentic-customer-support-rag
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv sync --group dev --extra ingest
   ```

3. Configure environment variables in `.env`:
   ```bash
   cp .env.example .env  # or edit .env directly
   ```

4. Run the backend service:
   ```bash
   uv run agenticrag serve --reload
   ```

### Frontend Setup
1. Enter the frontend directory:
   ```bash
   cd frontend
   npm install
   ```

2. Run development server:
   ```bash
   npm run dev
   ```

## Quality Assurance & Verification

Before submitting a Pull Request, ensure all linters, type checkers, and test suites pass:

### Backend Checks
```bash
# Linting and formatting
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run mypy src/

# Test suite with code coverage (enforcing >= 85% coverage)
uv run pytest tests/ -v --cov=agenticrag --cov-fail-under=85

# Dependency vulnerability audit
uv run pip-audit
```

### Frontend Checks
```bash
cd frontend

# Code linting
npm run lint

# TypeScript compilation
npx tsc -b

# Vitest suite
npm test

# Production build verification
npm run build

# Dependency security audit
npm audit --audit-level=moderate
```

## Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat(...)`: New feature or user-facing capability
- `fix(...)`: Bug fix or security patch
- `test(...)`: Adding or updating test suites
- `refactor(...)`: Code structure enhancement without behavior change
- `docs(...)`: Documentation updates

## Pull Request Guidelines
1. Create a descriptive feature branch (`git checkout -b feat/my-improvement`).
2. Keep commits atomic, cohesive, and clearly documented.
3. Ensure test coverage does not regress.
4. Verify that CI workflows (`.github/workflows/ci.yml`) pass on your fork/branch.
