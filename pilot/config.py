"""
Central configuration for the pilot study:
Multitask x Architecture (BiGRU / small Transformer / small Mamba-style SSM)
on a Project CodeNet (AOJ) subset.

Everything here is sized to run on a single 6GB-VRAM GPU (e.g. RTX 2060) with
16GB system RAM. Adjust the sizes in "SUBSET" and "MODEL" if you have more or
less memory available.
"""

import os
import torch

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
# Raw Project CodeNet root (set the CODENET_PATH env var). Expected layout (standard CodeNet release):
#   DATASET_PATH/
#       data/<problem_id>/<language>/<submission_id>.<ext>
#       metadata/<problem_id>.csv
DATASET_PATH = os.environ.get("CODENET_PATH", r"path/to/Project_CodeNet")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")           # tokenized/cached dataset
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "efficiency_frontier.png")

for d in (OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Data subset (keeps this a pilot, not a full-corpus run)
# ---------------------------------------------------------------------------
LANGUAGE = "C++"          # folder name under data/<problem_id>/  (e.g. "C++", "Python")
FILE_EXT_FALLBACK = "cpp" # used only if metadata is missing a filename_ext column
NUM_PROBLEMS = 30         # number of distinct problems (classes) to sample
MAX_PER_PROBLEM = 200     # max accepted+non-accepted submissions sampled per problem
MIN_PER_PROBLEM = 20      # skip problems with fewer usable submissions than this
RANDOM_SEED = 42

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# remainder (0.1) is TEST_FRAC

# ---------------------------------------------------------------------------
# 3. Tokenization
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE = 8000
MAX_SEQ_LEN = 256         # tokens per code sample (truncate/pad) -- keeps memory bounded
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"

# ---------------------------------------------------------------------------
# 4. Model sizing (kept small deliberately -- see proposal "Reduced Scope")
# ---------------------------------------------------------------------------
EMBED_DIM = 64
HIDDEN_DIM = 128          # encoder hidden size for all three architectures
NUM_LAYERS = 2            # recurrent layers / transformer layers / mamba blocks
NUM_HEADS = 4             # transformer only
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
USE_AMP = True            # mixed precision -- important for 6GB VRAM
NUM_WORKERS = 2           # DataLoader workers; set to 0 on Windows if you hit issues

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multitask loss weighting (classification : verdict)
MTL_WEIGHTS = {"classification": 1.0, "verdict": 1.0}

# ---------------------------------------------------------------------------
# 6. Architectures / regimes to sweep
# ---------------------------------------------------------------------------
ARCHITECTURES = ["bigru", "transformer", "mamba"]
TASKS = ["classification", "verdict"]

# Run each (architecture, regime) cell once per seed and report mean +/- std.
# With a test set this small (~380 examples), single-seed deltas of a few
# percentage points are not reliable evidence of an effect -- see the
# aggregated table in outputs/results_aggregated.csv.
SEEDS = [0, 1, 2]
