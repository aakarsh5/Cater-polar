# Cater-polar / ReasonVeritas

**Confidence-Calibrated Chain-of-Thought Reasoning for Interpretable and Reliable Deep Learning Models**
_(LIAR dataset for fake-news / claim prediction)_

---

## Overview

ReasonVeritas is a four-phase pipeline for interpretable, confidence-aware fake-news classification on the LIAR dataset. It cleans and tokenizes the text, caches DistilBERT features, trains a strong BiLSTM-Attention baseline, and then adds a Chain-of-Thought (CoT) reasoning head with three concept-gated attention channels: Emotion, Modality, and Negation.

If you only need the headline numbers, check [RESULTS_LOG.md](RESULTS_LOG.md) for the saved result locations and the table below for the current best scores.

## Start Here

If you want the shortest path to a usable run, start with the best model and the three baselines already tracked in this repo.

- Best overall model: Phase 4 fine-tune with `--unfreeze 2`
- Best classical baseline: Multinomial Naive Bayes
- All saved outputs: [RESULTS_LOG.md](RESULTS_LOG.md)

## Current Results Snapshot

| Model                             | Test Macro-F1 | Test Accuracy | Saved Result                                                                                               |
| --------------------------------- | ------------: | ------------: | ---------------------------------------------------------------------------------------------------------- |
| Phase 4 fine-tune, `--unfreeze 2` |    **0.7145** |    **0.7214** | [results/phase4_finetune_merged_L128_uf2_seed42.json](results/phase4_finetune_merged_L128_uf2_seed42.json) |
| Phase 4 fine-tune, `--unfreeze 4` |        0.7135 |        0.7178 | [results/phase4_finetune_merged_L128_uf4_seed42.json](results/phase4_finetune_merged_L128_uf4_seed42.json) |
| Logistic Regression baseline      |        0.6711 |        0.6823 | [results/logreg_baseline_merged_L128.json](results/logreg_baseline_merged_L128.json)                       |
| Linear SVM baseline               |        0.6618 |        0.6797 | [results/linear_svm_baseline_merged_L128.json](results/linear_svm_baseline_merged_L128.json)               |
| Multinomial NB baseline           |        0.6754 |        0.6902 | [results/multinomial_nb_baseline_merged_L128.json](results/multinomial_nb_baseline_merged_L128.json)       |

Best checkpoint and rationale sample for the strongest model:

- [models/phase4_finetune_merged_L128_uf2_best.pt](models/phase4_finetune_merged_L128_uf2_best.pt)
- [logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt](logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt)

For paper comparison, the strongest classical baseline is Multinomial NB, while the best overall model is the Phase 4 fine-tuned `--unfreeze 2` configuration.

| Phase       | Purpose                                                                                                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Clean → normalize → tokenize → truncate (L=128) → class balance → concept mapping → stratified splits → eval variants.                                                                                          |
| **Phase 2** | Reasoning-unit retokenization (legacy) → vocabulary encoding (legacy) → **DistilBERT contextual feature caching + subword↔word alignment** (current).                                                           |
| **Phase 3** | BiLSTM + multi-head self-attention classifier with optional metadata (party + credit-history + speaker/subject/context embeddings). Two variants: **frozen** features and **end-to-end fine-tuned** DistilBERT. |
| **Phase 4** | Chain-of-Thought head: 3 concept-gated attention heads + auxiliary multi-task losses + attention-grounded rationales. Same two variants (frozen / fine-tuned).                                                  |

All pipeline outputs are written to `data/`. Paths and constants are centralized in `src/config.py`.

---

## Key Highlights

This version expands the original README into a full project guide. The main additions are:

- **DistilBERT replaces GloVe.** Phase 2 step 3 caches `last_hidden_state` from `distilbert-base-uncased` plus the `subword_to_word` alignment used by Phase 4.
- **L=128 is the main comparison setting.** L=256 and L=512 still exist, but the reported paper numbers use L=128.
- **Metadata encoder.** `src/meta_encoder.py` adds party, credit-history counts, and learned speaker / subject / context embeddings.
- **Phase 3 and Phase 4 are included.** You now have both the frozen-feature and fine-tuned neural variants.
- **Reasoning explanations are saved.** Phase 4 writes attention-grounded rationales so the model output is easier to inspect.
- **Classical baselines are included.** Logistic Regression, Linear SVM, and Multinomial Naive Bayes are available for paper comparison.
- **Best results are indexed.** See [RESULTS_LOG.md](RESULTS_LOG.md) for one place to find every saved output.

## Baseline Models

The repository includes three simple, paper-friendly text baselines in `src/baselines/`:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Count Vectorizer + Multinomial Naive Bayes

Each baseline trains on the same `merged` LIAR + CoAID splits used by the neural models and saves a JSON/CSV pair in `results/`.

Run them from the project root:

```powershell
python src/baselines/logistic_regression_baseline.py --dataset merged --L 128
python src/baselines/linear_svm_baseline.py --dataset merged --L 128
python src/baselines/multinomial_nb_baseline.py --dataset merged --L 128
```

If you want a single place to find every saved path, use [RESULTS_LOG.md](RESULTS_LOG.md).

---

## Directory Structure

```
Reasonveritas/
├── README.md
├── FINETUNE_README.md         # Quick start for the fine-tuned pipeline (≥16 GB GPU/MPS)
├── run_phase1.py              # Run all Phase 1 steps and show summary
├── show_phase1_results.py     # View Phase 1 outputs without re-running
├── requirements.txt
├── data/                      # All pipeline outputs (created on first run)
│   ├── liar_cleaned_step1.csv
│   ├── liar_normalized_step2.csv
│   ├── liar_tokenized_step3.csv
│   ├── liar_truncated_L128.csv (also _L256, _L512)
│   ├── class_weights.txt
│   ├── liar_concepts_step6_L128.csv (also _L256, _L512)
│   ├── train_split_L128.csv, val_split_L128.csv, test_split_L128.csv
│   ├── train_eval_variants_step8_L128.csv
│   ├── bert_features_{train,val,test}_L128.npz   # Phase 2 step 3 (DistilBERT)
│   ├── subword_to_word_{train,val,test}_L128.npz # Phase 2 step 3 (alignment)
│   ├── concept_masks_{train,val,test}_L128.npz   # built lazily by Phase 4
│   ├── coaid/                                    # CoAID Claim CSVs (download_datasets.py)
│   ├── coaid_concepts_step6_L128.csv             # CoAID Phase 1 output (prepare_coaid.py)
│   ├── merged_{train,val,test}_split_L128.csv    # LIAR+CoAID joint splits (merge_datasets.py)
│   └── bert_features_merged_{train,val,test}_L128.npz  # joint BERT cache
├── models/                    # Saved checkpoints (best val macro-F1)
│   ├── phase3_L128_best.pt, phase3_finetune_L128_uf2_best.pt
│   ├── phase4_L128_best.pt,  phase4_finetune_L128_uf2_best.pt
│   └── baselines/             # serialized classical baseline models (pkl)
├── results/                   # Final test JSONs / CSV summaries (one per run)
├── logs/                      # CSV training logs + rationale dumps
├── docs/                      # auxiliary docs and diagrams (architecture SVG)
├── PHASE4_CHANGE_HISTORY.md    # change log for Phase 4 development
├── RESULTS_LOG.md              # index of saved results and models
└── src/
    ├── config.py              # Paths, BERT_MODEL_NAME, sequence lengths
    ├── meta_encoder.py        # Party + credit + speaker/subject/context embeddings
    ├── utils_logger.py
  ├── baselines/             # classical baselines (logreg, svm, multinomial_nb)
  │   ├── __init__.py
  │   ├── train_text_baselines.py
  │   ├── logistic_regression_baseline.py
  │   ├── linear_svm_baseline.py
  │   └── multinomial_nb_baseline.py
    ├── phase1/                # Dataset + preprocessing (steps 1–8)
    │   ├── prepare_coaid.py        # Self-contained CoAID Phase 1 pipeline
    │   └── merge_datasets.py       # Build LIAR + CoAID joint splits
    ├── phase2/                # Reasoning tokens + vocab (legacy) + BERT features (current)
    │   ├── step1_retokenize.py
    │   ├── step2_vocabulary_encoding.py
    │   └── step3_embeddings.py        # DistilBERT contextual features + subword alignment
    ├── phase3/                # BiLSTM-Attention classifier
    │   ├── bilstm_attention.py        # backbone (frozen-feature variant)
    │   ├── train_model.py             # frozen-feature trainer
    │   ├── bert_finetune.py           # FineTunedBERTClassifier (end-to-end DistilBERT)
    │   └── train_finetune.py          # fine-tune trainer
    └── phase4/                # Chain-of-Thought reasoning head
        ├── cot_model.py               # 3 concept-gated heads + fusion (frozen variant)
        ├── train_phase4.py            # frozen-feature trainer (gated aux loss)
        ├── cot_finetune.py            # CoTModelFineTune (end-to-end DistilBERT)
        ├── train_phase4_finetune.py   # fine-tune trainer + attention rationales
        └── rationale.py               # template + attention-grounded rationale generators
      ```

      ## Architecture diagram

      ![Model architecture](docs/architecture_diagram.svg)

      ---

## Prerequisites

- **Python** 3.8+
- **LIAR dataset:** place `train.tsv`, `valid.tsv`, `test.tsv` in a folder and set that path in `src/config.py` (`LIAR_DATASET_DIR`).
- **PyTorch** with MPS (Apple Silicon) or CUDA. Frozen pipeline runs comfortably on 8 GB RAM; fine-tuned pipeline needs ≥16 GB.

---

## Installation

From the project root:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Main dependencies:** `pandas`, `numpy`, `scikit-learn`, `spacy`, `emoji`, `beautifulsoup4`, `contractions`, `torch`, `transformers`.

---

## Configuration

Edit **`src/config.py`** before the first run:

| Variable           | Meaning                                                           | Default                   |
| ------------------ | ----------------------------------------------------------------- | ------------------------- |
| `LIAR_DATASET_DIR` | Folder containing raw `train.tsv`, `valid.tsv`, `test.tsv`        | (set me)                  |
| `DATA_DIR`         | Where pipeline outputs are saved                                  | `<repo>/data`             |
| `BERT_MODEL_NAME`  | HuggingFace model id used by Phase 2 step 3 + Phase 3/4 fine-tune | `distilbert-base-uncased` |
| `SEQUENCE_LENGTHS` | Truncation lengths produced by Phase 1                            | `[128, 256, 512]`         |

> The legacy `EMBEDDING_DIM` and `GLOVE_PATH` fields are still in `config.py` but are no longer consumed by Phase 3 or Phase 4. They only affect the legacy Phase 2 step 2 vocabulary encoding, which downstream code does not import.

---

## Running the full pipeline

All commands run from the project root. The recommended order on a fresh checkout is:

```bash
# Phase 1 — preprocess + splits
python src/phase1/step1_clean_liar.py
python src/phase1/step2_normalize_text.py
python src/phase1/step3_tokenize.py
python src/phase1/step4_sequence_length.py
python src/phase1/step5_class_balance.py
python src/phase1/step6_concept_mapping.py
python src/phase1/step7_dataset_split.py
python src/phase1/step8_evaluation_variants.py        # optional, for sufficiency/comprehensiveness eval

# Phase 2 — DistilBERT contextual features + subword alignment (only step 3 is needed by Phase 3/4)
python src/phase2/step3_embeddings.py

# Phase 3 — BiLSTM-Attention classifier (frozen features)
python src/phase3/train_model.py --L 128

# Phase 4 — Chain-of-Thought reasoning head (frozen features)
python src/phase4/train_phase4.py  --L 128
```

---

## Cross-domain training (LIAR + CoAID)

The `coaid-dataset-implementation` branch adds a second corpus — Cui & Lee's CoAID — and lets you train a single joint model on the union of LIAR (political claims) + CoAID (health claims). Same architecture, same trainers, just `--dataset merged`.

**Why joint training:** out-of-domain robustness. A LIAR-only model is fluent in political evasion patterns but has never seen a single COVID claim. A CoAID-only model has the inverse blind spot. Training on both forces the concept-gated heads to find domain-agnostic emotion / modality / negation signals instead of memorizing speakers.

**What's pulled from CoAID:** both the **Claim** subset (short factual claims) and the **News** subset (article headlines — only the `title` column, not the full article body). Title-level use keeps inputs short enough for L=128 and stylistically close to LIAR statements. We strip leading verdict-like prefixes (`FAKE:`, `FALSE:`, `DEBUNKED:`, `FACT-CHECK:`, `HOAX:`, `MYTH:`, etc.) from CoAID news headlines before tokenization so the classifier can't shortcut on them.

**One-time setup**

```bash
# 1. Download CoAID CSVs — 16 files (4 snapshots × {ClaimReal, ClaimFake, NewsReal, NewsFake}, ~5 MB total)
python data/download_datasets.py

# 2. CoAID Phase 1: clean → normalize → tokenize → truncate (L=128) → concept-tag
#    Auto-detects claim vs news from filename, strips label-leak prefixes, dedupes across snapshots.
python src/phase1/prepare_coaid.py
# Output: data/coaid_concepts_step6_L128.csv  (with coaid_subset column = "claim" | "news")

# 3. Merge LIAR + CoAID into joint train/val/test splits with dataset_source + has_meta columns,
#    stratified on (binary_label, dataset_source) so every split has proportional CoAID coverage.
python src/phase1/merge_datasets.py
# Outputs: data/merged_{train,val,test}_split_L128.csv

# 4. Re-cache DistilBERT contextual features over the merged splits
python src/phase2/step3_embeddings.py --dataset merged --L 128
# Outputs: data/bert_features_merged_{train,val,test}_L128.npz
```

### Joint training

Each trainer accepts `--dataset merged` and `--balance`. The recommended joint-training command for paper numbers is:

```bash
# Phase 3 frozen baseline (joint corpus, balanced sampler) — ~5 min on M2 Air
python src/phase3/train_model.py        --L 128 --dataset merged --balance

# Phase 4 CoT head (joint corpus, balanced sampler) — ~5 min
python src/phase4/train_phase4.py       --L 128 --dataset merged --balance

# End-to-end fine-tuned variants (~30–60 min on M3 Pro 36 GB each)
python src/phase3/train_finetune.py        --L 128 --unfreeze 2 --dataset merged --balance
python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2 --dataset merged --balance
```

Outputs are tagged with the dataset:

- `results/phase{3,4}_merged_L128_seed42.json`
- `models/phase{3,4}_merged_L128_best.pt`
- `logs/phase{3,4}_merged_L128_train.csv`
- Fine-tune runs use `phase{3,4}_finetune_merged_L128_uf{N}_*`.

The test report ends with a per-domain breakdown so you see LIAR-only and CoAID-only acc / macro-F1, not just the aggregate:

```
PER-DOMAIN TEST METRICS
  liar   n=1916  acc=0.6XX  macro_f1=0.6XX
  coaid  n=  XX  acc=0.X    macro_f1=0.X
```

### `--balance` (domain-balanced sampler)

Without it: ~96% of every batch is LIAR, ~4% CoAID. With CoAID's class-imbalance (real ≫ fake), the fake CoAID rows are essentially invisible during training.

With `--balance`: `WeightedRandomSampler` weights each row by `1 / count_of_(dataset_source, binary_label)_bucket` so each batch contains a balanced mix of `liar/fake`, `liar/real`, `coaid/fake`, `coaid/real`. Strongly recommended whenever `--dataset merged`.

### `has_meta` per-row mask (no fake metadata for CoAID)

LIAR rows carry speaker / party / credit-history / subject / context; CoAID rows don't. Rather than fabricate values:

- `prepare_coaid.py` writes empty strings / 0 for the missing columns and tags rows `has_meta = 0`.
- `merge_datasets.py` propagates `has_meta` per row (1 = LIAR, 0 = CoAID).
- `MetaEncoder.fit_metadata()` only fits the speaker/subject/context vocabs and the credit scaler on `has_meta == 1` rows so CoAID's empties can't poison vocab top-K.
- `MetaEncoder.forward(meta, has_meta=mask)` zeroes the metadata branch for `has_meta == 0` rows. The classifier sees an all-zero metadata slot for CoAID, never a noisy fake one.
- All four trainers thread `has_meta` end-to-end. With `--dataset liar` (default) `has_meta` is all-ones and behavior is unchanged.

### Multi-seed runs (paper numbers)

Run three seeds and report mean ± std. Add this loop after the joint corpus is built:

```bash
for s in 41 42 43; do
  python src/phase3/train_model.py        --L 128 --dataset merged --balance --seed $s
  python src/phase4/train_phase4.py       --L 128 --dataset merged --balance --seed $s
done

# Fine-tune (M3 Pro)
for s in 41 42 43; do
  python src/phase3/train_finetune.py        --L 128 --unfreeze 2 --dataset merged --balance --seed $s
  python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2 --dataset merged --balance --seed $s
done
```

Each run writes `results/<tag>_seed<S>.json` with `test_acc`, `test_macro_f1`, and `test_per_domain.{liar,coaid}.{n,acc,macro_f1}`. Aggregate them with this one-liner:

```bash
python3 -c "
import json, glob, statistics as s
files = sorted(glob.glob('results/phase4_finetune_merged_L128_uf2_seed*.json'))
fs = [json.load(open(f))['test_macro_f1'] for f in files]
print(f'macro_f1 = {s.mean(fs):.4f} ± {s.stdev(fs):.4f}  (n={len(fs)})')
"
```

### Recommended ablation table for the paper

| Run     | Frozen LIAR-only | Frozen merged + balance              | Fine-tuned LIAR-only   | Fine-tuned merged + balance                       |
| ------- | ---------------- | ------------------------------------ | ---------------------- | ------------------------------------------------- |
| Phase 3 | `--L 128`        | `--L 128 --dataset merged --balance` | `--L 128 --unfreeze 2` | `--L 128 --unfreeze 2 --dataset merged --balance` |
| Phase 4 | `--L 128`        | `--L 128 --dataset merged --balance` | `--L 128 --unfreeze 2` | `--L 128 --unfreeze 2 --dataset merged --balance` |

Each row × 3 seeds = 12 runs per phase. Report mean ± std and per-domain breakdown.

Or alternatively, run all of Phase 1 in one shot:

```bash
python run_phase1.py
```

For end-to-end DistilBERT fine-tuning (≥16 GB), see **Phase 3 — Fine-tuned variant** and **Phase 4 — Fine-tuned variant** below, and the `FINETUNE_README.md` quick start.

---

## Phase 1 — Step Details

| Step | Script                                | Output                                   | What it does                                                                                                                                    |
| ---- | ------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `phase1/step1_clean_liar.py`          | `liar_cleaned_step1.csv`                 | Load LIAR TSVs, select statement + label, drop missing/duplicates, clean text (HTML, URLs→`<URL>`, numbers→`<NUM>`, ALLCAPS marker, lowercase). |
| 2    | `phase1/step2_normalize_text.py`      | `liar_normalized_step2.csv`              | Unicode normalization, contractions, whitespace and punctuation normalization.                                                                  |
| 3    | `phase1/step3_tokenize.py`            | `liar_tokenized_step3.csv`               | Word-level tokenization (SpaCy), keep punctuation and negation.                                                                                 |
| 4    | `phase1/step4_sequence_length.py`     | `liar_truncated_L128.csv` (+ L256, L512) | Head truncation for ablation. All current results use L=128.                                                                                    |
| 5    | `phase1/step5_class_balance.py`       | `class_weights.txt`                      | Binary (fake/real) distribution + class weights for weighted loss.                                                                              |
| 6    | `phase1/step6_concept_mapping.py`     | `liar_concepts_step6_L{L}.csv`           | Lexicon-based concept tags: Emotion, Modality, Negation. Ground truth for Phase 4's concept heads — no labels are used here.                    |
| 7    | `phase1/step7_dataset_split.py`       | `train/val/test_split_L{L}.csv`          | Stratified 70/15/15 with `random_state=42`. Same row assignment across L.                                                                       |
| 8    | `phase1/step8_evaluation_variants.py` | `train_eval_variants_step8_L{L}.csv`     | Sufficiency (concept tokens only) and comprehensiveness (non-concept only) variants for faithfulness eval.                                      |

---

## Phase 2 — Step Details

Only **step 3** is consumed by Phase 3 and Phase 4. Steps 1 and 2 remain for the legacy GloVe-style pipeline and ablation purposes.

| Step | Script                                | Output                                                                                 | What it does                                                                                                                                                                                               |
| ---- | ------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `phase2/step1_retokenize.py`          | `liar_phase2_reasoning_tokens.csv`                                                     | (Legacy) Retokenize for reasoning units. Not required by Phase 3/4.                                                                                                                                        |
| 2    | `phase2/step2_vocabulary_encoding.py` | `vocab_word2idx.json`, `encoded_L{L}.pkl`                                              | (Legacy) Build vocabulary + integer-encode. Not required by Phase 3/4.                                                                                                                                     |
| 3    | `phase2/step3_embeddings.py`          | `bert_features_{train,val,test}_L128.npz`, `subword_to_word_{train,val,test}_L128.npz` | **Cache `distilbert-base-uncased` `last_hidden_state` (B, T=128, 768) + a per-token mapping from WordPiece subwords back to whitespace word indices**, used by Phase 4 to align concept masks to subwords. |

---

## Phase 3 — BiLSTM-Attention Classifier

Phase 3 trains the strong baseline: DistilBERT features → BiLSTM → multi-head self-attention → mean+max pooling → MLP head, with optional metadata (party + credit history + speaker/subject/context embeddings) concatenated into the fused vector before the classifier.

**Backbone (`src/phase3/bilstm_attention.py`)**

- BiLSTM over BERT `last_hidden_state`, `hidden_size=192`, single layer.
- 4-head self-attention over the BiLSTM outputs (with the attention mask from the tokenizer).
- Mean + max pool of attended outputs → fused 768-dim representation.
- Optional metadata branch:
  - `meta_dim` linear path (legacy, `meta_dim=15` = party-7 + credit-5 + presence flags-3), or
  - `meta_vocabs` path → `MetaEncoder` (party-7 + credit-5 + speaker-32 + subject-16 + context-16 = 96-dim out).
- Final classifier: `[fused | meta] → Linear → GELU → Dropout → Linear → 2 logits`.
- Dropout: `0.2` on attention outputs and inside the classifier MLP.

**Training (`src/phase3/train_model.py`, frozen features)**

- Reads `bert_features_*_L128.npz` from disk; BERT is _not_ loaded at train time.
- Smoothed cross-entropy (`smoothing=0.05`) with class weights from `class_weights.txt`.
- AdamW, single LR group `5e-4`, cosine schedule.
- Batch 32, ~10 epochs with early stop on val macro-F1.
- Best checkpoint → `models/phase3_L128_best.pt`. Test JSON → `results/phase3_L128_seed42.json`.

**Run (Run this when ram is less than 32gb)**

```bash
python src/phase3/train_model.py --L 128
```

Reference number: **macro-F1 ≈ 0.633** on the LIAR test split (seed 42).

### Phase 3 — Fine-tuned variant (end-to-end DistilBERT)

`src/phase3/bert_finetune.py` adds `FineTunedBERTClassifier`, which loads `distilbert-base-uncased` live, freezes its embeddings, unfreezes the top-N transformer blocks, and feeds `last_hidden_state` into the same BiLSTM-Attention head. `src/phase3/train_finetune.py` is the matching trainer:

- On-the-fly tokenization with `DistilBertTokenizer` (no NPZ cache reads).
- Two-group AdamW (ULMFiT-style): `bert_lr=2e-5`, `head_lr=5e-4`.
- Linear warmup (10%) + linear decay floored at 0.1.
- Gradient accumulation: defaults `--batch 16 --accum 2` → effective batch 32. Drop to `--batch 8 --accum 4` if you OOM.

**Run this when ram is above 32gb**

```bash
python src/phase3/train_finetune.py --L 128 --unfreeze 2
```

Knobs: `--unfreeze {2,4,6}` (top-N transformer blocks; 6 = full transformer), `--bert_lr`, `--head_lr`, `--batch`, `--accum`, `--epochs`, `--dropout`, `--seed`. Best checkpoint → `models/phase3_finetune_L128_uf2_best.pt`. Expected: **macro-F1 ≈ 0.66–0.69** with `--unfreeze 2`.

### Classical baselines for paper comparison

For a paper-ready comparison against the neural models, the repo now includes three classical baseline entry points trained on the same train/val/test splits:

1. `src/baselines/logistic_regression_baseline.py` - TF-IDF + Logistic Regression
2. `src/baselines/linear_svm_baseline.py` - TF-IDF + Linear SVM
3. `src/baselines/multinomial_nb_baseline.py` - Count Vectorizer + Multinomial Naive Bayes

Each script selects hyperparameters on validation macro-F1 and writes a result file in the same JSON/CSV style as the main models.

```bash
python src/baselines/logistic_regression_baseline.py --dataset merged --L 128
python src/baselines/linear_svm_baseline.py --dataset merged --L 128
python src/baselines/multinomial_nb_baseline.py --dataset merged --L 128
```

Outputs:

- `results/{model}_baseline_{dataset}_L{L}.json`
- `results/{model}_baseline_{dataset}_L{L}.csv`
- `models/baselines/{dataset}_L{L}_{model}.pkl`

Example console output:

```text
Classical baseline comparison (dataset=merged, L=128, seed=42)
[logreg] searching validation grid ...
  best={'C': 1.0}  val_f1=0.67xx  test_f1=0.66xx  test_acc=0.67xx
Saved results to results/logreg_baseline_merged_L128.json
Saved summary to results/logreg_baseline_merged_L128.csv
```

---

## Phase 4 — Chain-of-Thought Reasoning Head

Phase 4 adds three **concept-gated attention heads** on top of the Phase 3 backbone, each looking at a single reasoning channel of the input:

1. **Emotion** — affective lexicon hits (e.g. `terrible`, `outrageous`).
2. **Modality + scope** — modal verbs and their scope (e.g. `might`, `should`, `claimed that ...`).
3. **Negation + scope** — negation cues and their scope (e.g. `not`, `never`).

The masks come from Phase 1 step 6's lexicon-based concept mapping and are aligned to DistilBERT subwords via the cached `subword_to_word` map (Phase 2 step 3).

**Architecture (`src/phase4/cot_model.py`)**

- Shared BiLSTM + self-attention base (same as Phase 3).
- Three `ConceptHead` modules: each is a small attention pooler restricted to its concept mask, producing a per-concept score plus an auxiliary 2-class logit.
- Fusion: `[base_pool | em_pool | mo_pool | ne_pool | meta] → Linear → GELU → Dropout → Linear → 2 logits`.
- Returns an 11-tuple: main logits, three concept scores, three aux logits, three attention distributions, plus the base attention.

**Training (`src/phase4/train_phase4.py`, frozen features)**

- Main loss: smoothed CE with class weights.
- **Gated auxiliary loss (the dead-head fix):** each concept's aux classifier is only trained on examples where that concept actually appears in the input. Without this, the emotion head was learning nothing (`E_mean ≈ 0.04`); with it we get `E_mean ≥ 0.20`.
- Coverage penalty (target ≈ 0.3) on each concept attention to prevent the head from collapsing onto a single token.
- Per-step rationale generation via `rationale.generate_rationale` (template-based: maps scores to a fixed reasoning sentence).
- Best checkpoint → `models/phase4_L128_best.pt`. Test JSON → `results/phase4_L128_seed42.json`.

**Run (Run this when ram is less than 32gb)**

```bash
python src/phase4/train_phase4.py --L 128
```

Reference number: **macro-F1 ≈ 0.631** on the LIAR test split (seed 42).

### Phase 4 — Fine-tuned variant (end-to-end DistilBERT + attention-grounded rationales)

`src/phase4/cot_finetune.py` wraps a live DistilBERT around the existing `CoTModel` so the head architecture is identical to the frozen variant — clean ablation. `src/phase4/train_phase4_finetune.py` is the trainer. **What's new compared to the frozen Phase 4 trainer:**

1. End-to-end DistilBERT fine-tuning (selectively unfreeze top-N blocks).
2. Concept-presence-gated aux loss (the dead-head fix, same as frozen).
3. **Attention-grounded rationales.** `rationale.generate_rationale_with_tokens` decodes the top-K subword tokens each concept head attended to (WordPiece `##` continuations stripped, special tokens filtered), with their attention weights. Sample dump → `logs/rationales_phase4_finetune/sample_phase4_finetune_L128_uf2.txt`.

The attention-grounded rationale is what makes the paper's "faithful reasoning" claim substantiable: each step cites the actual tokens the head attended to with their attention weights, not just a templated score-to-text mapping.

**use this only when ram is more than 32gb**

```bash
python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2
```

Knobs: same as Phase 3 fine-tune. Best checkpoint → `models/phase4_finetune_L128_uf2_best.pt`. Test JSON → `results/phase4_finetune_L128_uf2_seed42.json`.

### Multi-seed (paper numbers)

After confirming end-to-end runs work, average over three seeds:

```bash
for s in 41 42 43; do
  python src/phase3/train_finetune.py        --L 128 --unfreeze 2 --seed $s
  python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2 --seed $s
done
```

Take mean ± std of `test_macro_f1` across the three `results/*_finetune_*_seed{41,42,43}.json` files.

---

## Metadata Encoder (`src/meta_encoder.py`)

Both Phase 3 and Phase 4 backbones accept an optional `meta_vocabs` dict that switches the metadata branch from the legacy 15-dim linear projection to a `MetaEncoder` module, motivated by Karimi & Tang (2019) (speaker history is the single most predictive feature on LIAR):

- **Party** — 7-way one-hot (dem, rep, ind, libertarian, green, none, other).
- **Credit history** — 5 numeric counts (barely_true, false, half_true, mostly_true, pants_on_fire), MinMax-scaled on train.
- **Speaker / subject / context** — top-K (default 500/100/80) vocabularies built on train, with `<UNK>` at index 0; learnable `nn.Embedding`s of size 32 / 16 / 16.
- Combined output: 96-dim, concatenated into the fused vector before the classifier.

`fit_metadata(train_df)` returns the vocab + scaler; `encode_metadata(df, vocabs, scaler)` produces the per-row tensors used by the trainers.

---

## Output Files Reference

- **Phase 1:** `liar_cleaned_step1.csv`, `liar_normalized_step2.csv`, `liar_tokenized_step3.csv`, `liar_truncated_L{L}.csv`, `class_weights.txt`, `liar_concepts_step6_L{L}.csv`, `train/val/test_split_L{L}.csv`, `train_eval_variants_step8_L{L}.csv`.
- **Phase 2 (current):** `bert_features_{train,val,test}_L128.npz`, `subword_to_word_{train,val,test}_L128.npz`.
- **Phase 4 mask cache:** `concept_masks_{train,val,test}_L128.npz` (built lazily on first run).
- **Models:** `models/phase{3,4}_L128_best.pt`, `models/phase{3,4}_finetune_L128_uf{N}_best.pt`.
- **Results:** `results/phase{3,4}_L128_seed{N}.json`, `results/phase{3,4}_finetune_L128_uf{N}_seed{N}.json`.
- **Logs:** `logs/phase{3,4}_*.csv` for training metrics; `logs/rationales_phase4*/` for rationale dumps.

All paths are under `data/` / `models/` / `results/` / `logs/` (or whatever you set in `config.py`).

---

## Sequence Length Variants

The pipeline produces L=128, 256, and 512 truncated splits, but **all reported numbers and recommended commands use L=128.** The longer variants are kept for ablations on accuracy vs. L and reasoning faithfulness vs. L.

---

## License and Citation

Use and cite according to your institution's and the LIAR dataset's terms.

Key references:

- Wang, W. Y. (2017). _"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection._ ACL.
- Karimi, H. & Tang, J. (2019). _Learning Hierarchical Discourse-level Structure for Fake News Detection._ NAACL.
- Howard, J. & Ruder, S. (2018). _Universal Language Model Fine-tuning for Text Classification._ ACL. (ULMFiT-style two-group LR.)
- Sun, C. et al. (2019). _How to Fine-Tune BERT for Text Classification?_ CCL.
- Sanh, V. et al. (2019). _DistilBERT, a distilled version of BERT._ NeurIPS EMC^2 Workshop.

## Chain-of-Thought (CoT) details and data flow

This section explains how data flows through the four phases and how the Phase 4 Chain-of-Thought (CoT) reasoning head is computed and rendered into human-readable rationales.

### High-level data flow

- Raw inputs: LIAR TSVs + (optional) CoAID CSVs → `src/phase1/*` preprocessing.
- Tokenization & truncation: `phase1/step3_tokenize.py` → `phase1/step4_sequence_length.py` produces `liar_truncated_L{L}.csv` and CoAID equivalents.
- Concept mapping: `phase1/step6_concept_mapping.py` tags every whitespace token with lexicon-derived concept indicators (Emotion, Modality, Negation) and writes `liar_concepts_step6_L{L}.csv` and `coaid_concepts_step6_L{L}.csv`.
- Merge (optional): `phase1/merge_datasets.py` builds `merged_{train,val,test}_split_L{L}.csv` combining LIAR + CoAID with `dataset_source` and `has_meta` columns.
- Feature caching: `phase2/step3_embeddings.py` runs `distilbert-base-uncased` over tokenized inputs and writes `bert_features_{*}_L{L}.npz` plus `subword_to_word` alignment arrays; these caches decouple expensive transformer passes from training loops.
- Model training: Phase 3 consumes BERT caches (or runs live for fine-tune) and trains a BiLSTM-Attention classifier. Phase 4 attaches 3 concept-gated attention heads and auxiliary losses; fine-tune variants optionally unfreeze the top-N transformer blocks and train end-to-end.

### Where to find artifacts (quick)

- Preprocessed CSVs: `data/liar_*`, `data/merged_*`
- BERT features and alignment: `data/bert_features*`, `data/subword_to_word*`
- Concept masks (word-level): `data/liar_concepts_step6_L{L}.csv`, `data/concept_masks_*.npz`
- Models & results: `models/` and `results/`
- Rationale text dumps (human-readable CoT): `logs/rationales_phase4*/*.txt`

### Phase 4 — CoT computation (conceptual)

Phase 4 implements an interpretable Chain-of-Thought head made of three concept channels: Emotion, Modality, and Negation. The implementation files are in `src/phase4/` (`cot_model.py`, `cot_finetune.py`, `rationale.py`, `train_phase4*.py`). The computation proceeds as follows for a single example:

1. Backbone encoding
  - Input tokens are mapped (via `subword_to_word`) to DistilBERT subword tokens and encoded into `last_hidden_state` vectors (shape B×T×H). For frozen runs we load these from `bert_features_*.npz`; for fine-tune runs we compute them on the fly.

2. Sequential encoder & attention
  - `BiLSTM` (Phase 3 backbone) runs over the BERT token vectors producing contextualized token outputs. A multi-head self-attention layer refines per-token importance signals.

3. Concept heads (Emotion / Modality / Negation)
  - Each concept head is an attention module over the token sequence that produces:
    - a per-token attention distribution α_concept (summing to 1 across tokens), and
    - a scalar concept score s_concept ∈ [0,1] obtained by pooling the attended token features followed by a small MLP + sigmoid.
  - Concept heads are trained with an auxiliary binary loss (when lexicon-derived concept labels exist) to encourage agreement between the head score and the lexicon presence of that concept in the ground-truth sentence. The auxiliary weight is `CONCEPT_AUX_LAMBDA` in `src/config.py`.

4. Coverage penalty and sparsity
  - During training we optionally apply a coverage penalty (hyperparameter `COVERAGE_PENALTY_LAMBDA`) that discourages diffuse attention across concepts and encourages concise token-level explanations. See `train_phase4*.py` for the exact term added to the loss.

5. Gating & fusion
  - The three concept head outputs are fused into the final classification in two ways:
    - Concatenation: pooled backbone representation + three concept scores → classifier MLP.
    - Gated modulation: learned scalar gates allow concept scores to modulate the classifier logits (helps the model rely more or less on CoT signals per-example).
  - The fusion weights and gates are learned end-to-end with the main classification objective.

6. Final label & confidence
  - The classifier outputs softmax logits; the `Verdict` is the argmax label and `Confidence` is the top softmax probability. These are saved to the JSON test report alongside per-concept scores and top contributing tokens.

### Rationale formatting (human readable)

Rationale files (examples in `logs/rationales_phase4_finetune/`) are generated by `src/phase4/rationale.py`. For each example the file contains:

- The original `Input` and normalized `Statement` text.
- `Actual label` (gold) and `Verdict` (model). `Confidence` is the model's top probability.
- For each concept head, a line showing `score` → a categorical label (low/medium/high) computed by thresholds (configurable) plus a short English hint (e.g. “heavy hedging”, “strong negation”).
- `Top tokens:` — token snippets with their α weights (how much that token contributed to the concept score). These are produced by mapping the per-subword attention back to whitespace words using the `subword_to_word` alignment and summing α across subwords.

Example snippet (already in `logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt`):

```
Step 1 [Emotion]:   score 0.00 → low     — minimal emotional language — claim is relatively neutral
        Top tokens: (no concept tokens in this example)
Step 2 [Modality]:  score 0.79 → HIGH    — heavy use of hedging words (may/might/could)
        Top tokens: "what" (α=0.72), "call" (α=0.18)
Step 3 [Negation]:  score 0.00 → low     — little to no negation
```

### Key knobs (config names)

- `CONCEPT_AUX_LAMBDA` — weight for auxiliary concept classification losses.
- `COVERAGE_PENALTY_LAMBDA` — weight for the attention coverage penalty (reduces redundant attention spread).
- `UNFREEZE_N` / CLI `--unfreeze N` — number of DistilBERT transformer blocks to unfreeze for fine-tuning.
- `--dataset {liar,coaid,merged}` — choose which dataset split to train on.
- `--balance` — domain-balanced sampler when using `merged` dataset.

### Quick inference / rationale export

To run inference and dump rationales for a saved checkpoint:

```bash
python src/phase4/train_phase4_finetune.py \
  --L 128 --dataset merged --unfreeze 2 --seed 42 \
  --mode evaluate --checkpoint models/phase4_finetune_merged_L128_uf2_best.pt \
  --rationale_out logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt
```

This will write per-example CoT explanations alongside the JSON test report in `results/`.

---

If you'd like, I can also:

- Commit this README change and push it to `origin/main` now.
- Add a small diagram image (SVG) showing the phase-by-phase data flow and call it from the README.

