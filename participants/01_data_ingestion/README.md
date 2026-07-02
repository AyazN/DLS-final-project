# 01 Data Ingestion / Dataset Processing

Ответственность:

- загрузка исходного датасета model cards;
- чтение JSON/JSONL/CSV/Parquet или Hugging Face dataset;
- базовая нормализация полей;
- преобразование записей в `ModelCard` из `src/aise/contracts.py`;
- сохранение сырых и нормализованных данных в `data/raw/` или `data/processed/`.

Ожидаемый выход модуля: коллекция `ModelCard`.
