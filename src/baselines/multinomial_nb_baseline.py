"""Count Vectorizer + Multinomial Naive Bayes baseline."""
from __future__ import annotations

import argparse
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from baselines.train_text_baselines import main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Multinomial Naive Bayes baseline.")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="liar", choices=("liar", "merged"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    main(
        L=args.L,
        dataset=args.dataset,
        seed=args.seed,
        save_models=not args.no_save_models,
        models=("multinomial_nb",),
    )