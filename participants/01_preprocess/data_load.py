from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "modelbiome/ai_ecosystem_withmodelcards"
USEFUL_COLUMNS = [
    "model_id",
    "likes",
    "downloads",
    "tags",
    "pipeline_tag",
    "library_name",
    "createdAt",
    "languages",
    "modelCard",
]


def clean_model_card(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.casefold() in {"nan", "none", "null", "<na>"}:
        return ""

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"#+\s*", " ", text)
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[-*•`]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def preprocess_dataframe(
    dataframe: Any,
    *,
    target_size: int = 600_000,
    popular_fraction: float = 0.25,
    random_state: int = 42,
):
    import pandas as pd

    missing = set(USEFUL_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    if not 0 <= popular_fraction <= 1:
        raise ValueError(
            "popular_fraction must be between 0 and 1."
        )

    df = dataframe[USEFUL_COLUMNS].copy()
    df["likes"] = pd.to_numeric(
        df["likes"],
        errors="coerce",
    ).fillna(0)
    df["downloads"] = pd.to_numeric(
        df["downloads"],
        errors="coerce",
    ).fillna(0)
    df = df.sort_values(
        by=["downloads", "likes"],
        ascending=False,
    )

    generated_pattern = (
        r"<!--\s*This model card has been generated automatically"
    )
    df = df[
        ~df["modelCard"].fillna("").str.contains(
            generated_pattern,
            regex=True,
        )
    ]
    df["modelCard"] = df["modelCard"].map(clean_model_card)
    df = df[df["modelCard"].str.len() > 20]
    df = df.drop_duplicates(subset=["model_id"])

    sample_size = min(target_size, len(df))
    popular_size = min(
        int(sample_size * popular_fraction),
        sample_size,
    )
    popular = df.head(popular_size)
    remaining = df.drop(popular.index)
    random_size = min(
        sample_size - popular_size,
        len(remaining),
    )
    random_sample = remaining.sample(
        n=random_size,
        random_state=random_state,
    )
    df = pd.concat(
        [popular, random_sample],
        ignore_index=True,
    )

    for column in ("tags", "languages"):
        df[column] = df[column].map(_list_value)
    df["createdAt"] = pd.to_datetime(
        df["createdAt"],
        errors="coerce",
    ).dt.date
    return df


def run(
    dataset_name: str,
    output: Path,
    *,
    split: str = "train",
    target_size: int = 600_000,
    popular_fraction: float = 0.25,
    random_state: int = 42,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Data ingestion requires the datasets package."
        ) from exc

    dataframe = load_dataset(
        dataset_name,
        split=split,
    ).to_pandas()
    cleaned = preprocess_dataframe(
        dataframe,
        target_size=target_size,
        popular_fraction=popular_fraction,
        random_state=random_state,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and clean model cards."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "clean_dataset.parquet"
        ),
    )
    parser.add_argument("--target-size", type=int, default=600_000)
    parser.add_argument(
        "--popular-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run(
        args.dataset,
        args.output,
        split=args.split,
        target_size=args.target_size,
        popular_fraction=args.popular_fraction,
        random_state=args.random_state,
    )
    print(f"Saved cleaned dataset to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
