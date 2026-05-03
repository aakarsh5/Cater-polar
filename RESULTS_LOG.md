# Results Log

This file records the saved locations for model outputs in this repository.

## Phase 4 Fine-Tune Results

### Best current model

- Result JSON: [results/phase4_finetune_merged_L128_uf2_seed42.json](results/phase4_finetune_merged_L128_uf2_seed42.json)
- Checkpoint: [models/phase4_finetune_merged_L128_uf2_best.pt](models/phase4_finetune_merged_L128_uf2_best.pt)
- Rationale sample: [logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt](logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf2.txt)

### Alternative run

- Result JSON: [results/phase4_finetune_merged_L128_uf4_seed42.json](results/phase4_finetune_merged_L128_uf4_seed42.json)
- Checkpoint: [models/phase4_finetune_merged_L128_uf4_best.pt](models/phase4_finetune_merged_L128_uf4_best.pt)
- Rationale sample: [logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf4.txt](logs/rationales_phase4_finetune/sample_phase4_finetune_merged_L128_uf4.txt)

### Earlier checkpoint

- Result JSON: [results/phase4_L128_seed42.json](results/phase4_L128_seed42.json)
- Checkpoint: [models/phase4_L128_best.pt](models/phase4_L128_best.pt)

## Baseline Results

### Logistic Regression baseline

- Result JSON: [results/logreg_baseline_merged_L128.json](results/logreg_baseline_merged_L128.json)
- Result CSV: [results/logreg_baseline_merged_L128.csv](results/logreg_baseline_merged_L128.csv)
- Checkpoint: [models/baselines/merged_L128_logreg.pkl](models/baselines/merged_L128_logreg.pkl)

### Multinomial Naive Bayes baseline

- Result JSON: [results/multinomial_nb_baseline_merged_L128.json](results/multinomial_nb_baseline_merged_L128.json)
- Result CSV: [results/multinomial_nb_baseline_merged_L128.csv](results/multinomial_nb_baseline_merged_L128.csv)

### Pending baseline runs

- Linear SVM baseline: `src/baselines/linear_svm_baseline.py`
- Multinomial Naive Bayes baseline: `src/baselines/multinomial_nb_baseline.py`

## Notes

- The paths above are relative to the repository root.
- Add new entries here whenever another model is trained or a new result file is created.
