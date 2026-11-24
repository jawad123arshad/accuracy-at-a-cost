"""
Builds a small, cached, tokenized subset of Project CodeNet for the pilot.

Two labels are produced per code sample (so single-task and multitask regimes
can reuse the exact same underlying data):
  - "classification": which of the NUM_PROBLEMS sampled problems the code solves
  - "verdict": 1 if status == "Accepted", else 0

Run this file directly once to build and cache the subset:
    python dataset.py
Subsequent runs (train.py / run_experiments.py) will reuse the cache.
"""

import os
import re
import json
import random
import pickle
import logging
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.\d+|\d+|[^\sA-Za-z0-9_]")


def tokenize(code: str):
    """Cheap, language-agnostic tokenizer: identifiers/keywords, numbers,
    and single-character punctuation/operators. Good enough for a pilot;
    swap in a real language tokenizer later if you extend this."""
    return TOKEN_RE.findall(code)


def _read_source(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def _discover_problem_ids():
    meta_dir = os.path.join(C.DATASET_PATH, "metadata")
    data_dir = os.path.join(C.DATASET_PATH, "data")
    if not os.path.isdir(meta_dir) or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Could not find 'metadata/' and 'data/' under DATASET_PATH={C.DATASET_PATH}. "
            "Check config.DATASET_PATH points at the extracted Project_CodeNet folder."
        )
    problem_ids = [
        fn[:-4] for fn in os.listdir(meta_dir)
        if fn.endswith(".csv") and os.path.isdir(os.path.join(data_dir, fn[:-4]))
    ]
    problem_ids.sort()
    return problem_ids, meta_dir, data_dir


def _load_problem_submissions(problem_id, meta_dir, data_dir, rng):
    """Returns a list of (code_text, status_str) for one problem, capped at
    MAX_PER_PROBLEM, restricted to config.LANGUAGE."""
    meta_path = os.path.join(meta_dir, f"{problem_id}.csv")
    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        log.warning(f"skip {problem_id}: could not read metadata ({e})")
        return []

    if "language" in df.columns:
        df = df[df["language"] == C.LANGUAGE]
    if df.empty:
        return []

    # Keep a mix of Accepted / non-Accepted so the verdict task isn't degenerate.
    if "status" not in df.columns:
        return []
    df = df.dropna(subset=["submission_id", "status"])

    if len(df) > C.MAX_PER_PROBLEM:
        df = df.sample(n=C.MAX_PER_PROBLEM, random_state=rng.randint(0, 2**31 - 1))

    ext_col = df["filename_ext"] if "filename_ext" in df.columns else None
    lang_dir = os.path.join(data_dir, problem_id, C.LANGUAGE)

    samples = []
    for i, row in df.iterrows():
        sub_id = str(row["submission_id"])
        ext = str(row["filename_ext"]) if ext_col is not None and pd.notna(row["filename_ext"]) else C.FILE_EXT_FALLBACK
        src_path = os.path.join(lang_dir, f"{sub_id}.{ext}")
        code = _read_source(src_path)
        if not code:
            continue
        samples.append((code, str(row["status"])))
    return samples


def build_raw_subset():
    """Scans DATASET_PATH and returns:
        problems: list[str]                 -- the sampled problem_ids (label space)
        records: list[dict]                 -- {"code":.., "problem_id":.., "status":..}
    """
    rng = random.Random(C.RANDOM_SEED)
    all_problem_ids, meta_dir, data_dir = _discover_problem_ids()
    rng.shuffle(all_problem_ids)

    chosen, records = [], []
    for pid in all_problem_ids:
        if len(chosen) >= C.NUM_PROBLEMS:
            break
        samples = _load_problem_submissions(pid, meta_dir, data_dir, rng)
        if len(samples) < C.MIN_PER_PROBLEM:
            continue
        chosen.append(pid)
        for code, status in samples:
            records.append({"code": code, "problem_id": pid, "status": status})

    if len(chosen) < C.NUM_PROBLEMS:
        log.warning(
            f"Only found {len(chosen)} problems with >= {C.MIN_PER_PROBLEM} "
            f"'{C.LANGUAGE}' submissions (wanted {C.NUM_PROBLEMS}). "
            "Consider lowering MIN_PER_PROBLEM or changing LANGUAGE in config.py."
        )
    log.info(f"Selected {len(chosen)} problems, {len(records)} total code samples.")
    return chosen, records


def build_vocab(records):
    counter = Counter()
    for r in records:
        counter.update(tokenize(r["code"]))
    specials = [C.PAD_TOKEN, C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN]
    most_common = [w for w, _ in counter.most_common(C.MAX_VOCAB_SIZE - len(specials))]
    itos = specials + most_common
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi):
    unk = stoi[C.UNK_TOKEN]
    ids = [stoi[C.BOS_TOKEN]] + [stoi.get(t, unk) for t in tokens[: C.MAX_SEQ_LEN - 2]] + [stoi[C.EOS_TOKEN]]
    return ids


def prepare_and_cache():
    """Builds the subset + vocab once and writes a pickle cache. Idempotent."""
    cache_file = os.path.join(C.CACHE_DIR, "subset.pkl")
    if os.path.exists(cache_file):
        log.info(f"Using cached subset: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    problems, records = build_raw_subset()
    problem_to_idx = {pid: i for i, pid in enumerate(problems)}
    stoi, itos = build_vocab(records)

    data = []
    for r in records:
        toks = tokenize(r["code"])
        ids = encode(toks, stoi)
        data.append({
            "ids": ids,
            "problem_label": problem_to_idx[r["problem_id"]],
            "verdict_label": 1 if r["status"].strip().lower() == "accepted" else 0,
        })

    rng = random.Random(C.RANDOM_SEED)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * C.TRAIN_FRAC)
    n_val = int(n * C.VAL_FRAC)
    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }

    payload = {
        "splits": splits,
        "stoi": stoi,
        "itos": itos,
        "problems": problems,
        "vocab_size": len(itos),
        "num_classes": len(problems),
    }
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(C.CACHE_DIR, "subset_stats.json"), "w") as f:
        json.dump({
            "num_problems": len(problems),
            "num_samples": n,
            "vocab_size": len(itos),
            "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]),
        }, f, indent=2)
    log.info(f"Cached subset -> {cache_file} "
             f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, "
             f"classes={len(problems)}, vocab={len(itos)})")
    return payload


class CodeNetPilotDataset(Dataset):
    def __init__(self, records, pad_id):
        self.records = records
        self.pad_id = pad_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "ids": r["ids"],
            "problem_label": r["problem_label"],
            "verdict_label": r["verdict_label"],
        }


def collate_fn(batch, pad_id):
    max_len = max(len(b["ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    problem_labels = torch.zeros(len(batch), dtype=torch.long)
    verdict_labels = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        input_ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        lengths[i] = L
        problem_labels[i] = b["problem_label"]
        verdict_labels[i] = b["verdict_label"]
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "problem_label": problem_labels,
        "verdict_label": verdict_labels,
    }


if __name__ == "__main__":
    payload = prepare_and_cache()
    print(json.dumps(
        {k: v for k, v in payload.items() if k in ("num_classes", "vocab_size")},
        indent=2,
    ))
