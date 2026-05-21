#!/usr/bin/env python3
"""Audit spatial/time calibration evidence for NMC front/diffusion proxies."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd


CALIBRATION_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*nm\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:um|µm|micron|microns)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:fps|Hz)\b", re.IGNORECASE),
    re.compile(r"\bpixel(?:\s+size)?\b", re.IGNORECASE),
    re.compile(r"\bfield\s*of\s*view\b|\bFoV\b", re.IGNORECASE),
]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def discover_files(base: Path, suffixes: Iterable[str], skip_dirs: Iterable[str] = ("derived", ".git", "__pycache__")) -> List[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    skip = set(skip_dirs)
    out: List[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if name.lower().endswith(suffixes):
                out.append(Path(root) / name)
    return sorted(out)


def stringify_attr(value: Any, max_len: int = 500) -> str:
    try:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            text = np.array2string(value, threshold=12)
        else:
            text = str(value)
    except Exception as exc:
        text = f"<unprintable:{type(value).__name__}:{exc}>"
    return text[:max_len]


def timing_scale_to_seconds(t: np.ndarray) -> tuple[np.ndarray, str]:
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return t, "unknown"
    span = float(np.nanmax(finite) - np.nanmin(finite))
    med_abs = float(np.nanmedian(np.abs(finite)))
    if med_abs > 1e12 or span > 1e9:
        return t / 1e9, "ns_to_s"
    if med_abs > 1e9 or span > 1e6:
        return t / 1e6, "us_to_s"
    if med_abs > 1e6 or span > 1e3:
        return t / 1e3, "ms_to_s"
    return t, "seconds"


def read_camera_timing(ds: h5py.Dataset, max_points: int = 20000) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"camera_timing_shape": tuple(int(x) for x in ds.shape)}
    try:
        candidates = []
        if ds.ndim == 2:
            n = min(int(ds.shape[1]), max_points)
            for row in range(int(ds.shape[0])):
                candidates.append((row, np.asarray(ds[row, :n], dtype=float)))
        elif ds.ndim == 1:
            n = min(int(ds.shape[0]), max_points)
            candidates.append((0, np.asarray(ds[:n], dtype=float)))
        else:
            return rec
        best = None
        for row, raw in candidates:
            t_scaled, unit = timing_scale_to_seconds(raw)
            t = t_scaled[np.isfinite(t_scaled)]
            if t.size < 2:
                continue
            dt = np.diff(t)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if dt.size == 0:
                continue
            score = (dt.size, float(np.nanmax(t) - np.nanmin(t)))
            if best is None or score > best[0]:
                best = (score, row, unit, t, dt)
        if best is not None:
            _, row, unit, t, dt = best
            rec["camera_timing_row_used"] = int(row)
            rec["camera_timing_unit_inference"] = unit
            rec["sampled_timing_duration_s"] = float(t[-1] - t[0])
            rec["sampled_timing_dt_median_s"] = float(np.median(dt))
            rec["sampled_timing_fps_median"] = float(1.0 / np.median(dt)) if np.median(dt) > 0 else None
            rec["sampled_timing_dt_min_s"] = float(np.min(dt))
            rec["sampled_timing_dt_max_s"] = float(np.max(dt))
    except Exception as exc:
        rec["camera_timing_error"] = str(exc)
    return rec


def audit_h5(base: Path, path: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(base)) if path.is_relative_to(base) else str(path),
        "file_size_mb": path.stat().st_size / 1e6,
    }
    attrs: List[Dict[str, str]] = []
    datasets: List[Dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as f:
            for key, value in f.attrs.items():
                attrs.append({"object": "/", "key": str(key), "value": stringify_attr(value)})
            for name, obj in f.items():
                if isinstance(obj, h5py.Dataset):
                    drec = {"name": name, "shape": tuple(int(x) for x in obj.shape), "dtype": str(obj.dtype)}
                    datasets.append(drec)
                    if name == "movie":
                        rec["movie_shape"] = drec["shape"]
                    if name == "camera_timing":
                        rec.update(read_camera_timing(obj))
                    for key, value in obj.attrs.items():
                        attrs.append({"object": name, "key": str(key), "value": stringify_attr(value)})
                elif isinstance(obj, h5py.Group):
                    datasets.append({"name": name, "shape": "<group>", "dtype": "group"})
                    for key, value in obj.attrs.items():
                        attrs.append({"object": name, "key": str(key), "value": stringify_attr(value)})
    except Exception as exc:
        rec["error"] = str(exc)
    rec["n_attrs"] = len(attrs)
    rec["n_datasets"] = len(datasets)
    rec["dataset_names"] = ";".join(str(d["name"]) for d in datasets[:40])
    hit_terms = []
    for item in attrs:
        text = f"{item['object']} {item['key']} {item['value']}"
        if any(p.search(text) for p in CALIBRATION_PATTERNS):
            hit_terms.append(text)
    rec["calibration_attr_hits"] = " || ".join(hit_terms[:20])
    rec["has_calibration_attr_hit"] = bool(hit_terms)
    return rec

def pptx_text_hits(path: Path) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
            for name in names:
                raw = zf.read(name).decode("utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"\s+", " ", text)
                snippets = []
                for pat in CALIBRATION_PATTERNS:
                    for m in pat.finditer(text):
                        start = max(0, m.start() - 100)
                        end = min(len(text), m.end() + 120)
                        snippets.append(text[start:end].strip())
                if snippets:
                    slide_num = re.findall(r"slide(\d+)\.xml", name)
                    hits.append({
                        "pptx_path": str(path),
                        "slide": int(slide_num[0]) if slide_num else None,
                        "snippets": " || ".join(dict.fromkeys(snippets[:10])),
                    })
    except Exception as exc:
        hits.append({"pptx_path": str(path), "slide": None, "snippets": f"ERROR: {exc}"})
    return hits


def audit_csv(path: Path, max_rows: int = 2000) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"path": str(path), "file_size_mb": path.stat().st_size / 1e6}
    try:
        df = pd.read_csv(path, nrows=max_rows)
        rec["n_rows_sampled"] = int(len(df))
        rec["columns"] = ";".join(map(str, df.columns[:80]))
        time_cols = [c for c in df.columns if re.search(r"time|elapsed|second|frame", str(c), re.IGNORECASE)]
        rec["time_frame_columns"] = ";".join(map(str, time_cols))
        for col in time_cols[:8]:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(x) >= 2:
                rec[f"{col}_min"] = float(x.min())
                rec[f"{col}_max"] = float(x.max())
                dx = x.sort_values().diff().dropna()
                dx = dx[(dx > 0) & np.isfinite(dx)]
                if len(dx):
                    rec[f"{col}_positive_step_median"] = float(dx.median())
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--repo-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_metadata_audit")
    parser.add_argument("--max-h5-files", type=int, default=0, help="Optional cap for metadata-only HDF5 scan; 0 scans all discovered raw HDF5 files.")
    args = parser.parse_args()

    base = Path(args.base_dir)
    repo = Path(args.repo_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    h5_files = discover_files(base, [".h5", ".hdf5"])
    n_h5_discovered = len(h5_files)
    if args.max_h5_files and args.max_h5_files > 0:
        h5_files = h5_files[: args.max_h5_files]
    h5_records = [audit_h5(base, p) for p in h5_files]
    h5_df = pd.DataFrame(h5_records)
    h5_df.to_csv(out / "h5_calibration_metadata.csv", index=False)

    csv_files = [p for p in discover_files(repo, [".csv"]) if "derived" not in p.parts]
    # Keep this audit metadata-only in spirit: small samples are enough to identify time/frame columns.
    csv_records = [audit_csv(p, max_rows=2000) for p in csv_files]
    csv_df = pd.DataFrame(csv_records)
    csv_df.to_csv(out / "csv_timebase_metadata.csv", index=False)

    pptx_files = discover_files(repo, [".pptx"])
    ppt_hits = []
    for p in pptx_files:
        ppt_hits.extend(pptx_text_hits(p))
    ppt_df = pd.DataFrame(ppt_hits)
    ppt_df.to_csv(out / "pptx_calibration_text_hits.csv", index=False)

    fps = pd.to_numeric(h5_df.get("sampled_timing_fps_median", pd.Series(dtype=float)), errors="coerce").dropna()
    movie_shapes = h5_df.get("movie_shape", pd.Series(dtype=str)).astype(str).value_counts().head(12).to_dict() if not h5_df.empty else {}
    attr_hits = h5_df[h5_df.get("has_calibration_attr_hit", False).astype(bool)] if "has_calibration_attr_hit" in h5_df else pd.DataFrame()
    summary = {
        "n_h5_discovered_before_cap": int(n_h5_discovered),
        "max_h5_files": int(args.max_h5_files),
        "n_h5_files": int(len(h5_df)),
        "n_h5_with_movie": int(h5_df.get("movie_shape", pd.Series(dtype=object)).notna().sum()) if not h5_df.empty else 0,
        "n_h5_with_camera_timing": int(h5_df.get("camera_timing_shape", pd.Series(dtype=object)).notna().sum()) if not h5_df.empty else 0,
        "n_h5_with_calibration_attr_hits": int(len(attr_hits)),
        "fps_median_across_h5": float(fps.median()) if len(fps) else None,
        "fps_min_across_h5": float(fps.min()) if len(fps) else None,
        "fps_max_across_h5": float(fps.max()) if len(fps) else None,
        "movie_shape_counts": movie_shapes,
        "n_csv_files_scanned": int(len(csv_df)),
        "n_pptx_files_scanned": int(len(pptx_files)),
        "n_pptx_calibration_hits": int(len(ppt_df)),
        "top_h5_calibration_attr_hits": attr_hits[["relative_path", "calibration_attr_hits"]].head(10).to_dict("records") if not attr_hits.empty else [],
        "top_pptx_hits": ppt_df.head(20).to_dict("records") if not ppt_df.empty else [],
        "guardrail": "This audit verifies metadata evidence for timebase/spatial calibration. Camera timing is present in sampled HDF5 files, but the sampled timing rows can represent sparse segment/cycle timing rather than true camera frame cadence. Physical pixel-size evidence should be treated as slide-derived unless raw HDF5 attributes or microscope metadata explicitly confirm it.",
        "outputs": {
            "h5_calibration_metadata": str(out / "h5_calibration_metadata.csv"),
            "csv_timebase_metadata": str(out / "csv_timebase_metadata.csv"),
            "pptx_calibration_text_hits": str(out / "pptx_calibration_text_hits.csv"),
            "summary": str(out / "calibration_metadata_audit_summary.json"),
        },
    }
    with (out / "calibration_metadata_audit_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Calibration Metadata Audit",
        "",
        "Metadata-only audit for spatial and time calibration evidence behind apparent front/diffusion proxies.",
        "",
        f"- HDF5 files discovered before optional cap: {summary['n_h5_discovered_before_cap']}",
        f"- Max HDF5 files requested: {summary['max_h5_files']}",
        f"- HDF5 files scanned: {summary['n_h5_files']}",
        f"- HDF5 files with movie datasets: {summary['n_h5_with_movie']}",
        f"- HDF5 files with camera timing: {summary['n_h5_with_camera_timing']}",
        f"- HDF5 files with calibration-like attributes: {summary['n_h5_with_calibration_attr_hits']}",
        f"- Median sampled HDF5 timing FPS proxy: {summary['fps_median_across_h5']}",
        f"- PPTX files scanned/calibration hits: {summary['n_pptx_files_scanned']} / {summary['n_pptx_calibration_hits']}",
        "",
        "## Interpretation",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
