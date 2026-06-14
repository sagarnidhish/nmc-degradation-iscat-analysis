#!/usr/bin/env python3
# File: scripts/tier1_h5_inventory.py
# Tier 1: metadata-only scan of all HDF5 files — never loads movie dataset
# Writes: derived/h5_inventory.csv

import argparse
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def ensure_output_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def discover_h5_files(base_dir: str) -> List[str]:
    results = []
    skip = {"derived", ".git", "__pycache__"}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if fn.lower().endswith((".h5", ".hdf5")):
                results.append(os.path.join(root, fn))
    return sorted(results)


def row_stats_2d(ds: h5py.Dataset, row: int, chunk: int = 2_000_000) -> Tuple[float, float, float]:
    if ds.ndim < 2 or row >= ds.shape[0] or ds.shape[1] == 0:
        return np.nan, np.nan, np.nan
    n = int(ds.shape[1])
    vmin, vmax, total, count = np.inf, -np.inf, 0.0, 0
    for s in range(0, n, chunk):
        arr = np.asarray(ds[row, s:min(s+chunk, n)], dtype=float)
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            continue
        vmin = min(vmin, float(fin.min()))
        vmax = max(vmax, float(fin.max()))
        total += float(fin.sum())
        count += fin.size
    if count == 0:
        return np.nan, np.nan, np.nan
    return vmin, vmax, total / count


def process_h5(base_dir: str, path: str) -> Dict:
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    session = parts[0] if len(parts) > 1 else "root"
    rec = dict(filename=rel, session=session,
               n_frames=np.nan, duration_s=np.nan, fps=np.nan,
               echem_rows=np.nan, V_min=np.nan, V_max=np.nan, I_mean_mA=np.nan)
    try:
        with h5py.File(path, "r") as f:
            # n_frames from movie shape (do NOT load data)
            if "movie" in f and isinstance(f["movie"], h5py.Dataset):
                try:
                    rec["n_frames"] = int(f["movie"].shape[0])
                except Exception:
                    pass

            # timing
            if "camera_timing" in f and isinstance(f["camera_timing"], h5py.Dataset):
                try:
                    ct = f["camera_timing"]
                    row = 1 if ct.ndim == 2 and ct.shape[0] >= 2 else 0
                    t = np.asarray(ct[row, :] if ct.ndim == 2 else ct[:], dtype=float)
                    t = t[np.isfinite(t)]
                    if t.size >= 2:
                        dur = float(t[-1] - t[0])
                        if dur > 0:
                            rec["duration_s"] = dur
                        dt = np.diff(t)
                        dt = dt[(dt > 0) & np.isfinite(dt)]
                        if dt.size > 0:
                            rec["fps"] = float(1.0 / np.median(dt))
                except Exception:
                    pass

            # echem from potentiostat_value preferred (rows: V, I, charge, time)
            if "potentiostat_value" in f and isinstance(f["potentiostat_value"], h5py.Dataset):
                try:
                    pv = f["potentiostat_value"]
                    if pv.ndim == 2 and pv.shape[0] >= 2:
                        rec["echem_rows"] = int(pv.shape[1])
                        v_min, v_max, _ = row_stats_2d(pv, 0)
                        _, _, i_mean = row_stats_2d(pv, 1)
                        rec["V_min"] = v_min
                        rec["V_max"] = v_max
                        rec["I_mean_mA"] = i_mean
                except Exception:
                    pass

            # fallback to echem_dataframe group
            if np.isnan(rec["echem_rows"]) and "echem_dataframe" in f:
                try:
                    grp = f["echem_dataframe"]
                    for k in grp.keys():
                        d = grp[k]
                        if isinstance(d, h5py.Dataset) and d.ndim >= 1:
                            rec["echem_rows"] = max(
                                float(rec["echem_rows"]) if np.isfinite(rec["echem_rows"]) else 0,
                                int(d.shape[-1])
                            )
                except Exception:
                    pass

    except Exception as exc:
        print(f"Warning: {path}: {exc}")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    args = parser.parse_args()

    ensure_output_dir(args.out_dir)
    files = discover_h5_files(args.base_dir)
    if not files:
        print(f"Warning: No HDF5 files found under {args.base_dir}")
        return
    print(f"Found {len(files)} HDF5 files. Scanning metadata...")

    records = []
    for i, fp in enumerate(files, 1):
        records.append(process_h5(args.base_dir, fp))
        if i % 5 == 0:
            print(f"  {i}/{len(files)} done")

    cols = ["filename", "session", "n_frames", "duration_s", "fps", "echem_rows", "V_min", "V_max", "I_mean_mA"]
    df = pd.DataFrame(records)[cols]
    out_csv = os.path.join(args.out_dir, "h5_inventory.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df.to_string())


if __name__ == "__main__":
    main()
