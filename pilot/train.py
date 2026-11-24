"""
Training/evaluation loop used by run_experiments.py for every
(architecture, regime) cell in the sweep, where regime is one of:
    "single_classification", "single_verdict", "multitask"
"""

import copy
import time
import random
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from dataset import CodeNetPilotDataset, collate_fn
from models import MultiTaskModel
from efficiency import (
    count_params, estimate_flops, measure_inference_latency,
    reset_peak_memory, peak_memory_mb,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(records, pad_id, shuffle):
    ds = CodeNetPilotDataset(records, pad_id)
    return DataLoader(
        ds, batch_size=C.BATCH_SIZE, shuffle=shuffle,
        collate_fn=functools.partial(collate_fn, pad_id=pad_id),
        num_workers=C.NUM_WORKERS, pin_memory=(C.DEVICE == "cuda"),
    )


@torch.no_grad()
def evaluate(model, loader, tasks, device):
    model.eval()
    correct = {t: 0 for t in tasks}
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids, lengths)
        bs = input_ids.size(0)
        total += bs
        for t in tasks:
            label_key = "problem_label" if t == "classification" else "verdict_label"
            labels = batch[label_key].to(device)
            preds = logits[t].argmax(dim=-1)
            correct[t] += (preds == labels).sum().item()
    return {t: correct[t] / max(total, 1) for t in tasks}


def majority_baseline(records, tasks):
    """Accuracy of always predicting the most frequent class in `records`,
    per task -- report alongside model accuracy so a task isn't mistaken for
    'learned' when it's really just close to the base rate."""
    out = {}
    for t in tasks:
        label_key = "problem_label" if t == "classification" else "verdict_label"
        labels = [r[label_key] for r in records]
        if not labels:
            out[t] = float("nan")
            continue
        most_common = max(set(labels), key=labels.count)
        out[t] = labels.count(most_common) / len(labels)
    return out


def train_one(arch, tasks, vocab_size, num_classes, pad_id, splits, device=C.DEVICE,
              epochs=C.EPOCHS, seed=0, verbose=True):
    """Trains one model (single-task if len(tasks)==1, else multitask) and
    returns a dict of accuracy + efficiency metrics.

    Uses best-validation-checkpoint selection: the model state with the
    highest mean validation accuracy (averaged over `tasks`) across epochs is
    what gets evaluated on test, not simply whatever the last epoch produced.
    This matters here because the verdict task tends to overfit well before
    the classification task does (see per-epoch val_acc logs), so "last
    epoch" and "best epoch" are often different checkpoints.
    """
    set_seed(seed)
    model = MultiTaskModel(arch, vocab_size, num_classes, active_tasks=tasks).to(device)

    train_loader = make_loader(splits["train"], pad_id, shuffle=True)
    val_loader = make_loader(splits["val"], pad_id, shuffle=False)
    test_loader = make_loader(splits["test"], pad_id, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(C.USE_AMP and device == "cuda"))
    ce = nn.CrossEntropyLoss()

    reset_peak_memory(device)
    start = time.perf_counter()

    best_val_avg = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(C.USE_AMP and device == "cuda")):
                logits = model(input_ids, lengths)
                loss = 0.0
                for t in tasks:
                    labels = (batch["problem_label"] if t == "classification" else batch["verdict_label"]).to(device)
                    loss = loss + C.MTL_WEIGHTS.get(t, 1.0) * ce(logits[t], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, tasks, device)
        val_avg = sum(val_acc.values()) / len(val_acc)
        if verbose:
            print(f"  [{arch} | {'+'.join(tasks)}] epoch {epoch+1}/{epochs} "
                  f"loss={running_loss/len(train_loader):.4f} val_acc={val_acc} (avg={val_avg:.4f})")
        if val_avg > best_val_avg:
            best_val_avg = val_avg
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    train_time_sec = time.perf_counter() - start
    peak_mem = peak_memory_mb(device)

    # Restore the best-validation checkpoint before touching the test set.
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  [{arch} | {'+'.join(tasks)}] using checkpoint from epoch {best_epoch} "
              f"(best val avg={best_val_avg:.4f}) for test evaluation")

    test_acc = evaluate(model, test_loader, tasks, device)
    baseline = majority_baseline(splits["test"], tasks)

    sample_batch = next(iter(test_loader))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    latency_ms = measure_inference_latency(model, sample_batch, device)
    flops = estimate_flops(model, sample_batch)

    result = {
        "arch": arch,
        "tasks": "+".join(tasks),
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": count_params(model),
        "train_time_sec": round(train_time_sec, 2),
        "peak_mem_mb": round(peak_mem, 1) if peak_mem == peak_mem else float("nan"),  # NaN-safe
        "inference_latency_ms_per_batch": round(latency_ms, 3),
        "flops": flops,
    }
    for t in tasks:
        result[f"test_acc_{t}"] = round(test_acc[t], 4)
        result[f"majority_baseline_{t}"] = round(baseline[t], 4)

    return result, model
