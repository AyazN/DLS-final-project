# AISE — AI Search Engine

Educational project for the **Deep Learning for Search** course.

AISE helps find relevant AI models based on user queries.

Example queries:

* `AI for football`
* `model for medicine`
* `AI for code generation`
* `model for image analysis`

---

## MVP

A Python CLI application that searches for relevant AI models in a model cards dataset.

---

## Team

| Participant | Role                                | Folder                               |
| ----------- | ----------------------------------- | ------------------------------------ |
| Ayaz        | Data Ingestion + Dataset Processing | `participants/01_data_ingestion/`    |
| Damir       | Embeddings + Vector Analysis   | `participants/02_preprocessing/`     |
| Magomedgadzhi        | Vector Search + Indexing           | `participants/03_embeddings_index/`  |
| Malik       | Retrieval + Ranking                 | `participants/04_retrieval_ranking/` |
| Denis       | Evaluation + Metrics                | `participants/05_evaluation/`        |

---

## Project Structure

```text
AISE/
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   └── results/
├── notebooks/
├── participants/
│   ├── 01_data_ingestion/
│   ├── 02_preprocessing/
│   ├── 03_embeddings_index/
│   ├── 04_retrieval_ranking/
│   └── 05_evaluation/
├── src/
│   └── aise/
│       ├── __init__.py
│       ├── cli.py
│       ├── contracts.py
│       └── pipeline.py
└── tests/
    ├── conftest.py
    └── test_contracts.py
```

---

## Quick Start

### Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Run

```bash
$env:PYTHONPATH="src"
python -m aise.cli search "AI for football"
```

---

## Git Workflow

We use three types of branches:

```text
main
develop
feature/*
```

### main

Stable version of the project.

Direct pushes are forbidden.

### develop

Main working branch of the team.

### feature/*

Individual participant branches.

Example:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/retrieval-ranking
```

After work:

```bash
git add .
git commit -m "Add retrieval module"
git push origin feature/retrieval-ranking
```

Then create a Pull Request into `develop`.

---

## Rules of Work

1. Everyone works in their own folder under `participants/`.
2. All changes are made via `feature/*` branches.
3. No direct pushes to `main`.
4. Shared interfaces are defined in `src/aise/contracts.py`.
5. Changes to shared contracts must be discussed with the team.

---

## Pipeline

```text
raw dataset
→ ModelCard
→ SearchDocument
→ embeddings / index
→ SearchResult
→ EvaluationReport
```

The project is designed so that each participant can work independently, and then all components are assembled into a unified search service.
