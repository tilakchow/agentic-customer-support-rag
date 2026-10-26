# Evaluations & Golden Dataset (Phase 0)

This directory contains evaluation datasets, baseline benchmarks, and evaluation scripts for the Agentic Customer Support RAG system.

---

## 1. Directory Structure

- `build_golden.py`: Script generating candidate golden test cases from Bitext dataset rows that are held out from the ingestion sample.
- `check_intents.py`: Validates that every non-null `expected_intent` in the golden dataset is present in the Pinecone vectorstore.
- `debug_retrieval.py`: Quick diagnostic script to inspect retriever similarity scores across sample queries.
- `retrieval_eval.py`: Quantitative retrieval evaluation script reporting mean, bootstrap 95% confidence intervals, and n for `recall@4`, `precision@4`, and `MRR`, as well as p10/p90 top-score percentiles.
- `golden_in_kb.jsonl`: 100 in-KB reference cases.
- `golden_other.jsonl`: Hand-written and edge cases (out-of-scope, injection, multi-turn, edge).
- `golden.jsonl`: The consolidated golden dataset (169 rows across `dev` and `test` splits).
- `baseline.json`: Ground-truth benchmark output recorded on the `dev` split.

---

## 2. Dev / Test Split Rule

> **Important**: Never tune prompts, thresholds, chunking, or algorithms on the **`test`** split.
> 
> The `test` split is strictly held out for final model verification and blind regression testing. All exploratory analysis, prompt engineering, threshold calibration, and continuous regression gates must be run against the **`dev`** split.

---

## 3. How Baseline was Produced

- **Date**: 2026-10-26
- **Git Commit**: `7f5cd7c47da207cea61ef55bf8f6df3f1641b9c7`
- **Command**:
  ```bash
  uv run python evals/retrieval_eval.py evals/golden.jsonl dev
  ```
- **Results (`evals/baseline.json`)**:
  ```json
  {
    "split": "dev",
    "recall@4": {
      "mean": 0.939,
      "ci95": [
        0.89,
        0.988
      ],
      "n": 82
    },
    "precision@4": {
      "mean": 0.811,
      "ci95": [
        0.744,
        0.875
      ],
      "n": 82
    },
    "mrr": {
      "mean": 0.886,
      "ci95": [
        0.824,
        0.945
      ],
      "n": 82
    },
    "top_score_in_kb_p10": 0.49,
    "top_score_oos_p90": 0.393,
    "n_oos": 24
  }
  ```

---

## 4. Running Evaluations Locally

### Intent Health Check
Verify that all 27 intents are present in the Pinecone namespace:
```bash
uv run python evals/check_intents.py
```

### Retrieval Evaluation (Dev Split)
Run quantitative metrics with bootstrap 95% confidence intervals:
```bash
uv run python evals/retrieval_eval.py evals/golden.jsonl dev
```

Optional threshold gating (e.g. require `recall@4 >= 0.889`):
```bash
uv run python evals/retrieval_eval.py evals/golden.jsonl dev 0.889
```

### Live Behavior & Quality Tests
Run live integration tests against the API and DeepEval:
```bash
# Run behavior checks (requires live server or TestClient with API keys)
uv run pytest tests/live -m live

# Run DeepEval quality checks (requires JUDGE_API_KEY)
uv run pytest tests/quality -m live
```
