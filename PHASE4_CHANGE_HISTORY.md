# Phase 4 Change History

Date: 2026-05-03

This file documents the changes made during the recent Phase 4 maintenance pass and how to revert them if needed.

## 1. Windows rationale export fix

Problem observed:

- `src/phase4/train_phase4.py` failed at the rationale export step on Windows with `UnicodeEncodeError` because the generated rationale text contains Unicode symbols such as `→`, while the default Windows text encoding is often `cp1252`.

Change made:

- Updated the rationale sample write call to open the output file with `encoding="utf-8"`.

Effect:

- Rationale samples now save correctly on Windows without changing the actual rationale content.

Rollback:

- Remove the explicit UTF-8 encoding from the write call in `src/phase4/train_phase4.py` if you want to return to the previous behavior.

## 2. Early-stop behavior clarification

Current behavior:

- Early stopping is controlled by validation macro F1.
- The best validation macro F1 is tracked during training.
- If a new epoch improves validation macro F1, `patience` resets to `0`.
- If it does not improve, `patience` increments by `1`.
- Training stops when `patience >= EARLY_STOP_PATIENCE`.

Files referenced:

- `src/phase4/train_phase4.py`
- `src/config.py`

## 3. Result-improvement change: threshold calibration

Motivation:

- The task is binary classification, so a fixed `argmax` decision rule is not always optimal for macro F1.

Change made:

- Added validation-based positive-class threshold calibration.
- Before final test evaluation, the script now searches a small threshold grid on the validation set and uses the best threshold for test predictions.

New behavior:

- Default run: calibrated threshold is used.
- Rollback option: run with `--no-calibrate-threshold` to restore the original argmax-based decision rule.

Files changed:

- `src/phase4/train_phase4.py`

Outputs added to the saved JSON results:

- `calibrated_threshold`
- `calibrated_val_macro_f1`

## 4. Validation performed

Checks run after edits:

- Python syntax compile check on `src/phase4/train_phase4.py`

Result:

- File compiled successfully.

## 5. CUDA / GPU support update

Problem observed:

- The training scripts were not actually using CUDA in the current Python environment, even though the user has a GPU and CUDA installed on the system.
- Runtime probing showed the active PyTorch build was CPU-only (`torch 2.10.0+cpu`), so CUDA was unavailable to the code.

Change made:

- Expanded `src/config.py:get_device()` so it can:
  - auto-detect CUDA, then MPS, then CPU
  - require CUDA explicitly when requested
  - respect the `CATER_POLAR_DEVICE` environment variable
- Added a `--device` CLI option to `src/phase4/train_phase4.py` so Phase 4 can be launched with `--device cuda` / `--device gpu`.

Effect:

- The code is now CUDA-aware and will use GPU when PyTorch exposes it.
- If CUDA is requested but the installed PyTorch build is CPU-only, the script raises a clear error instead of silently falling back.

Rollback:

- Remove the `--device` flag wiring from `src/phase4/train_phase4.py` and restore the old `get_device()` implementation in `src/config.py` if you want the original auto-detect-only behavior.

## 6. Ongoing logging policy

Requested behavior:

- Keep logging this and future code changes in this file for documentation purposes.

Practical rule for future edits:

- Append a new dated section here whenever a meaningful code change is made.
- Include what changed, why it changed, and how to roll it back.

## 7. Quick rollback summary

If you need to restore the original Phase 4 behavior:

1. Remove the UTF-8 encoding change in the rationale export.
2. Remove the threshold calibration code from `src/phase4/train_phase4.py`, or run the script with `--no-calibrate-threshold`.
3. Re-run the syntax compile check if you edit the file again.

## 8. Explicit device console logging

Problem observed:

- The startup logs showed the resolved device, but the console did not clearly spell out the requested device mode and the CUDA/PyTorch runtime state.

Change made:

- Added an explicit startup banner in `src/phase4/train_phase4.py` that prints:
  - requested device mode
  - resolved device
  - `torch` version
  - `torch.cuda.is_available()`
  - `torch.version.cuda`
  - `torch.backends.mps.is_available()`
- Expanded the `src/bert_features.py` load message with the same runtime details.

Effect:

- The console now makes it obvious whether the code is using CUDA, MPS, or CPU, and whether the active PyTorch build actually exposes CUDA.

Rollback:

- Remove the added startup banner in `src/phase4/train_phase4.py` and the extra runtime details in `src/bert_features.py` if you want the shorter original logs back.

## 9. GPU training speedup optimization

Problem observed:

- Training was slower than necessary because batch sizes and data loading were tuned for CPU/MPS, not CUDA GPUs.

Change made:

- Increased `TRAIN_BATCH_SIZE` from 32 → 128 in `src/config.py` (main training loop).
- Increased `BERT_BATCH_SIZE` from 16 → 64 in `src/config.py` (BERT feature extraction).
- Added `num_workers=4` to all DataLoaders in `src/phase4/train_phase4.py` (parallel data loading):
  - train_loader
  - val_loader
  - test_loader

Effect:

- Expected 2–4× training speedup depending on GPU VRAM.
- Parallel data loading reduces GPU idle time.
- Larger batches improve GPU utilization.

Rollback:

- Revert `TRAIN_BATCH_SIZE` to 32 and `BERT_BATCH_SIZE` to 16 in `src/config.py`.
- Remove `num_workers=4` from all DataLoader calls in `src/phase4/train_phase4.py`.

Notes:

- If GPU OOM occurs, reduce `TRAIN_BATCH_SIZE` to 64 and `BERT_BATCH_SIZE` to 32.
- These settings assume CUDA GPU with sufficient VRAM (8GB+). Adjust for your hardware.

## 10. Accuracy improvement: extended training and lower patience

Current baseline (before optimization):

- test_acc: 0.6360
- test_macro_f1: 0.6284
- calibrated_threshold: 0.4449
- Device: CUDA

Problem observed:

- The early stopping patience of 5 was cutting off training before the learning rate scheduler could fully refine the model in its later phases.

Change made:

- Increased `TRAIN_NUM_EPOCHS` from 30 → 50 in `src/config.py` (more time for cosine annealing decay).
- Decreased `EARLY_STOP_PATIENCE` from 5 → 3 in `src/config.py` (stricter stopping, forces validation improvement).

Rationale:

- With cosine annealing LR schedule, the learning rate decays slowly and then more steeply. Epochs 20–30 often show the best refinement.
- Lowering patience from 5 → 3 means the model must improve validation macro-F1 more frequently, pushing training to explore better local minima.

Expected effect:

- 0.5–1.5% improvement in macro-F1 (your baseline 0.6284 → target ~0.635–0.64).

Rollback:

- Revert `TRAIN_NUM_EPOCHS` to 30 and `EARLY_STOP_PATIENCE` to 5 in `src/config.py` if you want the original behavior back.

## 11. Fix Windows multiprocessing error in DataLoaders

Problem observed:

- Training crashed on Windows with `RuntimeError: Couldn't open shared file mapping` when using `num_workers=4`.
- Root cause: Windows uses `spawn` context for multiprocessing (not `fork`), and PyTorch's tensor serialization for multiple workers on Windows is unreliable.

Change made:

- Reverted `num_workers` from 4 → 0 in all DataLoader calls in `src/phase4/train_phase4.py`.
- This disables parallel data loading but ensures training runs on Windows without crashes.

Effect:

- Training now works on Windows. Single-threaded data loading with CUDA GPU is still fast enough given the cached BERT features.

Rollback:

- Set `num_workers=4` in DataLoader calls if you're on Unix/Mac and want parallel loading back.

Notes:

- On Unix/Mac with CUDA, you can restore `num_workers=4` for faster data pipeline.
- On Windows, `num_workers=0` is recommended due to multiprocessing limitations.

## 12. Rationale samples now include input and actual label

Problem observed:

- The saved rationale samples showed only the generated explanation and predicted verdict, so the input claim and true label were not visible.

Change made:

- Updated the sample export in `src/phase4/train_phase4.py` to include:
  - the original input statement
  - the actual label (`fake` / `real`)
  - the predicted label with confidence

Effect:

- Rationale files are now self-contained for inspection and error analysis.
- You can directly compare the input claim, ground truth, and model verdict in one place.

Rollback:

- Remove the extra sample export lines in `src/phase4/train_phase4.py` if you want the shorter rationale-only output back.

## 13. Fine-tune rationale export fix and stronger concept-loss tuning

Problem observed:

- `src/phase4/train_phase4_finetune.py` crashed on Windows while writing rationale samples because the output file used the default cp1252 encoding.

Change made:

- Opened the finetune rationale sample file with `encoding="utf-8"`.

## 14. Classical baseline comparison suite

Motivation:

- The paper needs simple, reproducible baselines alongside the stronger Phase 3 / Phase 4 neural models.

Change made:

- Added a new `src/baselines/` folder with three model-specific entry points:
  - `logistic_regression_baseline.py`
  - `linear_svm_baseline.py`
  - `multinomial_nb_baseline.py`
- Kept a shared baseline runner in `src/baselines/train_text_baselines.py` so all three scripts emit the same JSON/CSV result structure and optional pickled model files under `models/baselines/`.
- Updated `README.md` with the exact paper-comparison commands and output file locations.

Effect:

- You now have a clean baseline suite for tables/ablations without changing the neural pipeline.

Rollback:

- Remove `src/baselines/train_text_baselines.py` and the README/history entries if you want to return to the pre-baseline state.
- Expanded finetune rationale samples to include:
  - the original input statement
  - the actual label text
  - the gold label index
- Increased `CONCEPT_AUX_LAMBDA` in `src/config.py` from 0.3 → 0.5.
- Increased the finetune coverage penalty in `src/phase4/train_phase4_finetune.py` from 0.05 → 0.1.

Effect:

- Finetune rationale export now works on Windows.
- The samples are now self-contained for inspection.
- Stronger concept supervision should help the auxiliary heads stay active and improve downstream accuracy.

Rollback:

- Restore the default file encoding in `train_phase4_finetune.py` if you want the old behavior.
- Revert `CONCEPT_AUX_LAMBDA` to 0.3 and the finetune coverage penalty back to 0.05 if you want the previous loss weighting.

## 14. Point CoAID to local download path

Problem observed:

- The repo expected CoAID data under `data/coaid/`, but the dataset already exists at `C:\Users\lohan\Downloads\CoAID-0.4`.

Change made:

- Updated `src/config.py` so `COAID_DIR` now defaults to `C:\Users\lohan\Downloads\CoAID-0.4`.
- Updated `src/config.py` so `COAID_DIR` now defaults to `C:\Users\lohan\Downloads\CoAID-0.4\CoAID-0.4`.
- The path can still be overridden with the `COAID_DIR` environment variable.

Effect:

- `prepare_coaid.py` and the merged LIAR+CoAID pipeline can use your local download directly.
- No manual copy into `data/coaid/` is required as long as the snapshot folders live under that root.

Rollback:

- Set `COAID_DIR` back to `os.path.join(DATA_DIR, "coaid")` in `src/config.py` if you want the repo-local folder behavior again.

## 15. Make CoAID loader scan snapshot subfolders

Problem observed:

- The downloaded CoAID release stores CSVs inside dated snapshot folders (`05-01-2020/`, `07-01-2020/`, etc.), but `prepare_coaid.py` originally only searched the top-level directory.

Change made:

- Updated `src/phase1/prepare_coaid.py` to search recursively for `*COVID-19.csv`.
- It now collects both top-level matches and nested snapshot-folder matches.

Effect:

- `prepare_coaid.py` can now read the actual CoAID release layout without manual file copying.

Rollback:

- Restore the original non-recursive glob in `load_coaid()` if you want the old behavior.

## 16. Add `--device` support to finetune trainer

Problem observed:

- `src/phase4/train_phase4_finetune.py` rejected `--device cuda`, even though the frozen Phase 4 trainer already supported it.

Change made:

- Added a `--device` CLI option to `train_phase4_finetune.py`.
- Wired it into `get_device(device_mode)` so the finetune run can explicitly use CUDA, MPS, CPU, or auto-detect.

Effect:

- The finetune command can now be run exactly as requested with `--device cuda`.

Rollback:

- Remove the `--device` CLI argument and the `device_mode` plumbing from `train_phase4_finetune.py` if you want the old auto-only behavior back.
