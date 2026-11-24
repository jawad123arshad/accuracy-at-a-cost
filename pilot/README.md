# Pilot study (C++, 30 problems)

An earlier, smaller version of the study, written as plain Python scripts. It compares **BiGRU**, a small
**Transformer** and a minimal pure-PyTorch **Mamba-style selective SSM** on 30 C++ problems from Project
CodeNet (3,828 submissions), with two tasks: problem classification and binary verdict (Accepted vs. not).
The main study in [`../notebooks`](../notebooks) replaced it; the code is kept here for reference.

> The Mamba block in `models.py` reimplements the selective-scan recurrence of Gu & Dao (2023) with a
> Python loop over time steps. It is not the official `mamba_ssm` CUDA package, and it is much slower.

## Run

```bash
cd pilot
export CODENET_PATH=/path/to/Project_CodeNet     # PowerShell: $env:CODENET_PATH = "..."
python dataset.py           # optional: build + cache the subset (run_experiments does this too)
python run_experiments.py   # 3 architectures × 3 regimes × 3 seeds
```

If you hit out-of-memory errors, lower `BATCH_SIZE`, `MAX_SEQ_LEN` or `HIDDEN_DIM` in `config.py`. On
Windows, set `NUM_WORKERS = 0` if the DataLoader hangs.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths and hyper-parameters |
| `dataset.py` | Scans CodeNet, tokenizes, builds and caches the subset |
| `models.py` | BiGRU / Transformer / Mamba-style encoders + multitask heads |
| `efficiency.py` | Parameter count, latency, peak memory, optional FLOPs (`thop`) |
| `train.py` | Training loop with best-validation checkpoint selection |
| `run_experiments.py` | Runs the full sweep and writes the tables and the plot |

## Results of the reported pilot run (`results/`)

Mean test accuracy over 3 seeds, multitask (single-task in brackets):

| Encoder | Classification | Verdict | Train time |
|---|---:|---:|---:|
| BiGRU | 0.444 (0.440) | 0.608 (0.606) | ~285 s |
| Transformer | 0.642 (0.649) | 0.655 (0.635) | ~200 s |
| Mamba-style SSM | 0.741 (0.745) | 0.643 (0.668) | ~1,425 s |

Majority-class baselines: 0.068 (classification), 0.544 (verdict). No single-task vs. multitask difference
was significant (Welch's t-test, p > 0.15).
