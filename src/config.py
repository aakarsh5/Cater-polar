"""
Central config for paths and pipeline constants.
Adjust LIAR_DATASET_DIR if the raw data lives elsewhere.
"""
import os

# -----------------------------
# DATASET PATHS
# -----------------------------
# Raw LIAR dataset directory (train.tsv, valid.tsv, test.tsv)
LIAR_DATASET_DIR = "/Users/lohan/Downloads/liar_dataset"

# Project data directory for pipeline outputs (default: project/data)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Cross-domain datasets.
# COAID_DIR should point at the CoAID repo root that contains the snapshot
# folders (05-01-2020, 07-01-2020, 09-01-2020, 11-01-2020, etc.).
COAID_DIR     = os.getenv("COAID_DIR", r"C:\Users\lohan\Downloads\CoAID-0.4\CoAID-0.4")
LIAR_PLUS_DIR = os.path.join(DATA_DIR, "liar_plus")   # LIAR-PLUS with justifications

# -----------------------------
# SEQUENCE LENGTHS
# -----------------------------
SEQUENCE_LENGTHS = [128, 256, 512]
MAX_SEQUENCE_LEN = SEQUENCE_LENGTHS[0]

# -----------------------------
# VOCABULARY (legacy custom-vocab path; still used by some scripts)
# -----------------------------
VOCAB_MIN_FREQ = 2

# -----------------------------
# EMBEDDINGS  (Phase 2 Step 3)
# -----------------------------
# We now use a frozen DistilBERT encoder as a feature extractor.
# GloVe path is kept only for backward-compatibility; ignored if BERT_MODEL_NAME is set.
EMBEDDING_DIM = 768                     # DistilBERT hidden size
GLOVE_PATH    = None
BERT_MODEL_NAME = "distilbert-base-uncased"   # CPU/MPS-friendly, 66M params, 768-dim
BERT_MAX_LEN    = 128                          # cap subword sequence length for memory
BERT_BATCH_SIZE = 64                           # increased for GPU; reduce to 32 if OOM

# -----------------------------
# DEVICE
# -----------------------------
def get_device(preferred: str | None = None):
    """Return the requested or best available torch device.

    preferred values:
      - "auto" / None: cuda -> mps -> cpu
      - "cuda" / "gpu": require CUDA-enabled PyTorch
      - "mps": require Apple Metal
      - "cpu": force CPU

    The CATER_POLAR_DEVICE environment variable can be used as a global
    override when preferred is not passed explicitly.
    """
    import torch

    requested = (preferred or os.getenv("CATER_POLAR_DEVICE", "auto")).lower()
    if requested in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build (not the CPU-only build) and "
            "confirm that the NVIDIA driver/runtime is visible to Python."
        )
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available in this PyTorch build.")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# TRAINING DEFAULTS  (Phase 3 / 4)
# -----------------------------
TRAIN_BATCH_SIZE   = 128
TRAIN_LR           = 1e-3
TRAIN_WEIGHT_DECAY = 1e-5
TRAIN_DROPOUT      = 0.4
TRAIN_NUM_EPOCHS   = 50
EARLY_STOP_PATIENCE = 3
FOCAL_LOSS_GAMMA   = 2.0
GRAD_CLIP_NORM     = 1.0
RANDOM_SEED        = 42

# -----------------------------
# PHASE 4 CONCEPT REASONING
# -----------------------------
CONCEPT_AUX_LAMBDA = 0.5   # weight on auxiliary concept-prediction losses
CONCEPT_GATE_ALPHA = 1.0   # boost factor for concept-tagged tokens in attention

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, COAID_DIR, LIAR_PLUS_DIR):
    os.makedirs(_d, exist_ok=True)
