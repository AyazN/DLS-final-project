# AI Search Engine

Учебный проект для курса **Deep Learning for Search**.

Пользователь вводит запрос вроде `"ИИ для футбола"` или `"модель для медицины"`, а система ищет наиболее подходящие AI-модели в датасете model cards.

## MVP

CLI-приложение на Python, которое постепенно объединит пять независимых модулей:

1. Data Ingestion / Dataset Processing
2. Preprocessing / Document Building
3. Embeddings + Vector Index
4. Retrieval + Ranking
5. Evaluation + Metrics

## Структура

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

## Ответственность папок

- `src/aise/` - общий код проекта: контракты, типы данных, CLI и будущая сборка pipeline.
- `src/aise/contracts.py` - единые интерфейсы между модулями. Участники должны возвращать и принимать эти типы.
- `src/aise/pipeline.py` - место, где позже будут соединены ingestion, preprocessing, indexing, retrieval и evaluation.
- `participants/` - независимые рабочие папки участников. Каждый может писать свою реализацию, эксперименты и тесты внутри своей зоны.
- `data/raw/` - исходные датасеты model cards.
- `data/processed/` - очищенные документы и промежуточные таблицы.
- `data/indexes/` - векторные индексы, BM25-индексы и другие поисковые артефакты.
- `data/results/` - результаты запусков и evaluation-отчеты.
- `notebooks/` - исследовательские ноутбуки, EDA и эксперименты.
- `tests/` - общие тесты контрактов и интеграции.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m aise.cli search "ИИ для футбола"
```

Если пакет запускается из корня без установки, добавьте `src` в `PYTHONPATH`:

```bash
export PYTHONPATH=src  # Windows PowerShell: $env:PYTHONPATH="src"
```

## Контракт разработки

Каждый модуль должен зависеть от общих типов из `src/aise/contracts.py`, а не от внутренних файлов других участников.

Минимальный поток данных:

```text
raw dataset -> ModelCard -> SearchDocument -> embeddings/index -> SearchResult -> EvaluationReport
```

Такой подход позволяет разрабатывать части независимо и позже заменить любую реализацию без переписывания всего pipeline.
