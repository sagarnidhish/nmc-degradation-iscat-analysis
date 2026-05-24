#!/usr/bin/env python3
"""Simple next-frame prediction baseline on ROI NPZ tensors.

Goal:
- Provide a fast, reproducible baseline for particle-ROI-only next-frame
  prediction using the existing NPZ exports (frames_norm).
- This is deliberately small and intended for rapid iteration and ablations.

Design:
- Input: last K frames (default 4) from a 96x96 ROI crop.
- Model: lightweight 2D CNN that predicts a residual over the last frame.
- Loss: L1 on predicted next frame.
- Split: grouped by `source_stem` (leave-source-out style) to avoid leakage.

Outputs:
- metrics.json with per-split and overall metrics
- per-sequence CSV with errors

Guardrail:
- This is a pixel-space baseline. It does not claim physical correctness.
  The next step is to add physics-facing auxiliary losses/heads (front proxies,
  intensity drift, interface density) and/or echem conditioning.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(str(path), newline="") as f:
        return list(csv.DictReader(f))


def split_by_source(rows: List[Dict[str, str]], holdout_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(int(seed))
    sources = sorted({str(r.get("source_stem")) for r in rows if r.get("source_stem") is not None})
    rng.shuffle(sources)
    n_holdout = max(1, int(round(len(sources) * float(holdout_frac))))
    holdout = set(sources[:n_holdout])
    train_idx, test_idx = [], []
    for i, r in enumerate(rows):
        if str(r.get("source_stem")) in holdout:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--context", type=int, default=4, help="Number of past frames as input.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(Path(args.manifest_csv))
    train_idx, test_idx = split_by_source(rows, holdout_frac=float(args.holdout_frac), seed=int(args.seed))

    # Lazy import torch only once arguments are parsed (so metadata-only reads stay cheap).
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SmallResNet(nn.Module):
        def __init__(self, in_ch: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
            self.conv4 = nn.Conv2d(16, 1, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (b, k, h, w)
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            delta = self.conv4(h)  # (b,1,h,w)
            last = x[:, -1:, :, :]
            return last + delta

    model = SmallResNet(in_ch=int(args.context)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    def iter_batches(indices: List[int], shuffle: bool) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        idxs = indices[:]
        if shuffle:
            np.random.default_rng(int(args.seed)).shuffle(idxs)
        bs = int(args.batch_size)
        for start in range(0, len(idxs), bs):
            batch_ids = idxs[start : start + bs]
            xs = []
            ys = []
            kept = []
            for bi in batch_ids:
                npz_path = Path(str(rows[bi].get("npz_path")))
                if not npz_path.exists():
                    continue
                with np.load(str(npz_path)) as data:
                    frames = np.asarray(data["frames_norm"], dtype=np.float32)
                if frames.ndim != 3 or frames.shape[0] < (int(args.context) + 1):
                    continue
                # Random time index per epoch for variety; fixed seed so it's reproducible.
                rng = np.random.default_rng(int(args.seed) + bi + start)
                t = int(rng.integers(int(args.context), frames.shape[0] - 1))
                x = frames[t - int(args.context) : t]  # (k,h,w)
                y = frames[t + 1]  # (h,w)
                xs.append(x)
                ys.append(y)
                kept.append(bi)
            if not xs:
                continue
            x_t = torch.from_numpy(np.stack(xs, axis=0)).to(device)
            y_t = torch.from_numpy(np.stack(ys, axis=0)[:, None, :, :]).to(device)
            yield x_t, y_t, kept

    def eval_split(indices: List[int]) -> Dict[str, Any]:
        model.eval()
        abs_err = []
        sq_err = []
        per_seq = []
        with torch.no_grad():
            for x_t, y_t, kept in iter_batches(indices, shuffle=False):
                pred = model(x_t)
                err = pred - y_t
                ae = err.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
                se = (err * err).mean(dim=(1, 2, 3)).detach().cpu().numpy()
                for bi, a, s in zip(kept, ae, se):
                    per_seq.append(
                        {
                            "roi_id": rows[bi].get("roi_id"),
                            "cycleNo": rows[bi].get("cycleNo"),
                            "source_stem": rows[bi].get("source_stem"),
                            "next_frame_mae": float(a),
                            "next_frame_mse": float(s),
                        }
                    )
                abs_err.extend(ae.tolist())
                sq_err.extend(se.tolist())
        def q(vals, qq):
            if not vals:
                return None
            vals = sorted(vals)
            idx = qq * (len(vals) - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                return float(vals[lo])
            return float(vals[lo] * (hi - idx) + vals[hi] * (idx - lo))
        return {
            "n_examples": int(len(abs_err)),
            "mae_median": q(abs_err, 0.5),
            "mae_p90": q(abs_err, 0.9),
            "mse_median": q(sq_err, 0.5),
            "mse_p90": q(sq_err, 0.9),
            "per_sequence": per_seq,
        }

    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        losses = []
        for x_t, y_t, _kept in iter_batches(train_idx, shuffle=True):
            pred = model(x_t)
            loss = (pred - y_t).abs().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "train_loss_l1": float(np.mean(losses)) if losses else None})

    train_metrics = eval_split(train_idx)
    test_metrics = eval_split(test_idx)

    per_seq = test_metrics["per_sequence"]
    csv_path = out_dir / "next_frame_baseline_per_sequence.csv"
    if per_seq:
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_seq[0].keys()))
            w.writeheader()
            for r in per_seq:
                w.writerow(r)

    metrics = {
        "device": str(device),
        "context": int(args.context),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "holdout_frac": float(args.holdout_frac),
        "n_sequences": int(len(rows)),
        "n_train_sequences": int(len(train_idx)),
        "n_test_sequences": int(len(test_idx)),
        "train": {k: v for k, v in train_metrics.items() if k != "per_sequence"},
        "test": {k: v for k, v in test_metrics.items() if k != "per_sequence"},
        "history": history,
        "outputs": {"per_sequence_csv": str(csv_path)},
        "guardrail": "Pixel-space next-frame baseline; not a physics-calibrated model.",
    }
    (out_dir / "next_frame_baseline_metrics.json").write_text(json.dumps(clean_json(metrics), indent=2, sort_keys=True))
    torch.save(model.state_dict(), out_dir / "next_frame_baseline_model.pt")

    print(json.dumps(clean_json(metrics["test"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

