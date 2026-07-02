# AISE — AI Search Engine

Учебный проект для курса **Deep Learning for Search**.

AISE помогает находить подходящие AI-модели по запросу пользователя.

Примеры запросов:

- `ИИ для футбола`
- `модель для медицины`
- `AI для написания кода`
- `модель для анализа изображений`

---

## MVP

CLI-приложение на Python, которое ищет релевантные AI-модели в датасете model cards.

---

## Команда

| Участник | Роль | Папка |
|-----------|-------|--------|
| Аяз | Data Ingestion / Dataset Processing | `participants/01_data_ingestion/` |
| Дамир | Preprocessing / Document Building | `participants/02_preprocessing/` |
| Имя | Embeddings + Vector Index | `participants/03_embeddings_index/` |
| Малик | Retrieval + Ranking | `participants/04_retrieval_ranking/` |
| Имя | Evaluation + Metrics | `participants/05_evaluation/` |

---

## Структура проекта

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

## Быстрый старт

### Создание окружения

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

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск

```bash
$env:PYTHONPATH="src"
python -m aise.cli search "ИИ для футбола"
```

---

## Git Workflow

Используем три типа веток:

```text
main
develop
feature/*
```

### main
Стабильная версия проекта.

Прямые пуши запрещены.

### develop
Основная рабочая ветка команды.

### feature/*
Личные ветки участников.

Пример:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/retrieval-ranking
```

После работы:

```bash
git add .
git commit -m "Add retrieval module"
git push origin feature/retrieval-ranking
```

Далее создается Pull Request в `develop`.

---

## Правила работы

1. Каждый работает в своей папке в `participants/`.
2. Все изменения делаются через `feature/*`.
3. Не пушим напрямую в `main`.
4. Общие интерфейсы находятся в `src/aise/contracts.py`.
5. Изменения общих контрактов обсуждаем с командой.

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

Проект построен так, чтобы каждый участник мог работать независимо, а затем все части собирались в единый поисковый сервис.
