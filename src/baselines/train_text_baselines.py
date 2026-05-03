"""
Train three classical text-classification baselines for paper comparison.

The script uses the same LIAR / merged train-val-test splits as the neural
pipeline, selects the best hyperparameters on the validation split, and then
reports test metrics for:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Count Vectorizer + Multinomial Naive Bayes

Usage:
    python src/baselines/train_text_baselines.py --dataset merged --L 128

Outputs:
    results/{model}_baseline_{dataset}_L{L}.json
    results/{model}_baseline_{dataset}_L{L}.csv
    models/baselines/{dataset}_L{L}_{model}.pkl
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED  # noqa: E402


@dataclass
class BaselineResult:
    model: str
    vectorizer: str
    best_params: dict
    val_acc: float
    val_macro_f1: float
    test_acc: float
    test_macro_f1: float
    train_seconds: float
    test_domain_breakdown: dict | None


def _split_csv(L: int, dataset: str, split: str) -> str:
    if dataset == "liar":
        return os.path.join(DATA_DIR, f"{split}_split_L{L}.csv")
    return os.path.join(DATA_DIR, f"{dataset}_{split}_split_L{L}.csv")


def _text_column(df: pd.DataFrame) -> pd.Series:
    if "clean_statement" in df.columns:
        return df["clean_statement"].fillna("").astype(str)
    return df["statement"].fillna("").astype(str)


def _label_array(df: pd.DataFrame) -> np.ndarray:
    labels = df["binary_label"] if "binary_label" in df.columns else df["label"]
    if len(labels) == 0:
        return np.zeros(0, dtype=np.int64)
    first = labels.iloc[0]
    if isinstance(first, str):
        return np.array([0 if str(v).strip().lower() == "fake" else 1 for v in labels], dtype=np.int64)
    return labels.astype(int).to_numpy(dtype=np.int64)


def _load_split(L: int, dataset: str, split: str) -> pd.DataFrame:
    path = _split_csv(L, dataset, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the Phase 1 split generation first for L={L} ({dataset})."
        )
    return pd.read_csv(path)


def _domain_breakdown(preds: np.ndarray, labels: np.ndarray, split_df: pd.DataFrame) -> dict | None:
    if "has_meta" not in split_df.columns:
        return None
    has_meta = split_df["has_meta"].to_numpy(dtype=np.float32)
    out: dict[str, dict[str, float | int]] = {}
    for name, mask_val in (("liar", 1.0), ("coaid", 0.0)):
        mask = has_meta == mask_val
        if int(mask.sum()) == 0:
            continue
        out[name] = {
            "n": int(mask.sum()),
            "acc": float(accuracy_score(labels[mask], preds[mask])),
            "macro_f1": float(f1_score(labels[mask], preds[mask], average="macro")),
        }
    return out or None


def _fit_one(
    *,
    model_name: str,
    vectorizer_factory,
    estimator_factory,
    param_grid: dict,
    train_texts: pd.Series,
    train_labels: np.ndarray,
    val_texts: pd.Series,
    val_labels: np.ndarray,
):
    best = None
    best_score = -1.0
    best_val_acc = -1.0
    best_pipeline = None

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        pipe = Pipeline([
            ("vectorizer", vectorizer_factory()),
            ("classifier", estimator_factory(**params)),
        ])
        pipe.fit(train_texts, train_labels)
        val_preds = pipe.predict(val_texts)
        val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))
        val_acc = float(accuracy_score(val_labels, val_preds))
        if val_macro_f1 > best_score or (val_macro_f1 == best_score and val_acc > best_val_acc):
            best_score = val_macro_f1
            best_val_acc = val_acc
            best = params
            best_pipeline = pipe

    assert best is not None and best_pipeline is not None
    return best_pipeline, best, best_score, best_val_acc


def _save_pipeline(pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def main(
    L: int = 128,
    dataset: str = "liar",
    seed: int = RANDOM_SEED,
    save_models: bool = True,
    models: tuple[str, ...] | list[str] | None = None,
):
    np.random.seed(seed)
    print(f"\nClassical baseline comparison (dataset={dataset}, L={L}, seed={seed})")

    train_df = _load_split(L, dataset, "train")
    val_df = _load_split(L, dataset, "val")
    test_df = _load_split(L, dataset, "test")

    train_texts = _text_column(train_df)
    val_texts = _text_column(val_df)
    test_texts = _text_column(test_df)
    train_labels = _label_array(train_df)
    val_labels = _label_array(val_df)
    test_labels = _label_array(test_df)

    model_specs = {
        "logreg": {
            "vectorizer_name": "tfidf_word_1_2",
            "vectorizer_factory": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            "estimator_factory": lambda C: LogisticRegression(
                C=C,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=seed,
            ),
            "param_grid": {"C": [0.5, 1.0, 2.0]},
        },
        "linear_svm": {
            "vectorizer_name": "tfidf_word_1_2",
            "vectorizer_factory": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            "estimator_factory": lambda C: LinearSVC(
                C=C,
                class_weight="balanced",
                random_state=seed,
            ),
            "param_grid": {"C": [0.5, 1.0, 2.0]},
        },
        "multinomial_nb": {
            "vectorizer_name": "count_word_1_2",
            "vectorizer_factory": lambda: CountVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                strip_accents="unicode",
            ),
            "estimator_factory": lambda alpha: MultinomialNB(alpha=alpha),
            "param_grid": {"alpha": [0.25, 0.5, 1.0]},
        },
    }

    selected = list(models) if models else list(model_specs.keys())
    summary_rows = []
    result_rows = []
    models_dir = os.path.join(MODELS_DIR, "baselines")
    os.makedirs(models_dir, exist_ok=True)

    for model_name in selected:
        spec = model_specs[model_name]
        print(f"\n[{model_name}] searching validation grid ...")
        started = time.time()
        pipeline, best_params, val_macro_f1, val_acc = _fit_one(
            model_name=model_name,
            vectorizer_factory=spec["vectorizer_factory"],
            estimator_factory=spec["estimator_factory"],
            param_grid=spec["param_grid"],
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
        )
        train_seconds = time.time() - started

        test_preds = pipeline.predict(test_texts)
        test_acc = float(accuracy_score(test_labels, test_preds))
        test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))
        breakdown = _domain_breakdown(test_preds, test_labels, test_df)

        if save_models:
            model_path = os.path.join(models_dir, f"{dataset}_L{L}_{model_name}.pkl")
            _save_pipeline(pipeline, model_path)

        result = BaselineResult(
            model=model_name,
            vectorizer=spec["vectorizer_name"],
            best_params=best_params,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
            test_acc=test_acc,
            test_macro_f1=test_macro_f1,
            train_seconds=train_seconds,
            test_domain_breakdown=breakdown,
        )
        result_rows.append(result)
        summary_rows.append({
            "model": model_name,
            "vectorizer": spec["vectorizer_name"],
            "best_params": json.dumps(best_params, sort_keys=True),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "test_acc": round(test_acc, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "train_seconds": round(train_seconds, 1),
        })
        print(
            f"  best={best_params}  val_f1={val_macro_f1:.4f}  "
            f"test_f1={test_macro_f1:.4f}  test_acc={test_acc:.4f}"
        )
        if breakdown:
            for domain, scores in breakdown.items():
                print(f"    {domain}: n={scores['n']} acc={scores['acc']:.4f} macro_f1={scores['macro_f1']:.4f}")

    if len(result_rows) == 1:
        model_name = result_rows[0].model
        out_json = os.path.join(RESULTS_DIR, f"{model_name}_baseline_{dataset}_L{L}.json")
        out_csv = os.path.join(RESULTS_DIR, f"{model_name}_baseline_{dataset}_L{L}.csv")
        payload = {
            "dataset": dataset,
            "L": L,
            "seed": seed,
            **asdict(result_rows[0]),
        }
    else:
        out_json = os.path.join(RESULTS_DIR, f"baselines_text_{dataset}_L{L}.json")
        out_csv = os.path.join(RESULTS_DIR, f"baselines_text_{dataset}_L{L}.csv")
        payload = {
            "dataset": dataset,
            "L": L,
            "seed": seed,
            "results": [asdict(r) for r in result_rows],
        }
    with open(out_json, "w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nSaved results to {out_json}")
    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical text baselines.")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="liar", choices=("liar", "merged"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=("all", "logreg", "linear_svm", "multinomial_nb"),
        help="Run one baseline or all three.",
    )
    args = parser.parse_args()
    selected_models = None if args.model == "all" else (args.model,)
    main(
        L=args.L,
        dataset=args.dataset,
        seed=args.seed,
        save_models=not args.no_save_models,
        models=selected_models,
    )