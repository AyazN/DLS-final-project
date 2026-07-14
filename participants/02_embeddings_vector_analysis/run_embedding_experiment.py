from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embedding_utils import get_embedding_dir  # noqa: E402


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full embedding experiment: generate two encoders, analyze them, and optionally shrink MiniLM."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/clean_dataset.parquet"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-card-chars", type=int, default=2500)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analysis-sample-size", type=int, default=5000)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--shrink-baseline", action="store_true")
    parser.add_argument("--target-dim", type=int, default=128)
    parser.add_argument("--fit-sample-size", type=int, default=5000)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    gen_cmd = [
        sys.executable,
        "participants/02_embeddings_vector_analysis/generate_embeddings.py",
        "--input",
        str(args.input),
        "--models",
        *args.models,
        "--batch-size",
        str(args.batch_size),
        "--max-model-card-chars",
        str(args.max_model_card_chars),
    ]
    if args.limit and args.limit > 0:
        gen_cmd += ["--limit", str(args.limit)]
    if args.device:
        gen_cmd += ["--device", args.device]
    if args.overwrite:
        gen_cmd.append("--overwrite")
    if args.resume:
        gen_cmd.append("--resume")
    run(gen_cmd)

    if args.analyze:
        for model_name in args.models:
            emb_dir = get_embedding_dir(model_name)
            run(
                [
                    sys.executable,
                    "participants/02_embeddings_vector_analysis/analyze_embeddings.py",
                    "--embedding-dir",
                    str(emb_dir),
                    "--sample-size",
                    str(args.analysis_sample_size),
                    "--n-clusters",
                    str(args.n_clusters),
                ]
            )

    if args.shrink_baseline:
        baseline_dir = get_embedding_dir(args.models[0])
        run(
            [
                sys.executable,
                "participants/02_embeddings_vector_analysis/shrink_embeddings.py",
                "--embedding-dir",
                str(baseline_dir),
                "--target-dim",
                str(args.target_dim),
                "--fit-sample-size",
                str(args.fit_sample_size),
                "--batch-size",
                str(args.fit_sample_size),
                "--overwrite",
            ]
        )

    print("\nEmbedding experiment finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
