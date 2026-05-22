#!/usr/bin/env python3
"""Audit calibration provenance for the assumed NMC optical scale."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np
import pandas as pd


TARGET_NM_PER_PX = 96.0
NEAR_FRACTION = 0.10

CALIBRATION_TERMS = re.compile(
    r"\bpixels?\b|\bpx\b|\bnm\b|\bum\b|µm|\bmicrons?\b|field\s*of\s*view|\bfov\b|"
    r"\bmagnification\b|\bobjective\b|\bcamera\b|\bexposure\b|\bcalibration\b|\bscale\b|"
    r"\bresolution\b|\bbinning\b",
    re.IGNORECASE,
)
NM_PER_PIXEL = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|nanometer|nanometers)\s*(?:/|per|\s+)?\s*(?:px|pixel|pixels|pixel\s+size)",
    re.IGNORECASE,
)
PIXEL_SIZE_NM = re.compile(
    r"(?:pixel\s+size|pixelsize|scale)\D{0,40}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|nanometer|nanometers)",
    re.IGNORECASE,
)
UM_LENGTH = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>um|µm|micron|microns)\b", re.IGNORECASE)
FOV_PAIR = re.compile(
    r"(?P<x>\d+(?:\.\d+)?)\s*(?:x|by|×)\s*(?P<y>\d+(?:\.\d+)?)\s*(?P<unit>um|µm|micron|microns)",
    re.IGNORECASE,
)
FPS_OR_EXPOSURE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fps|hz|ms|s)\b",
    re.IGNORECASE,
)


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
        return float(value) if math.isfinite(float(value)) else None
    return value


def discover_files(base: Path, suffixes: Iterable[str], skip_dirs: Iterable[str]) -> List[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    skip = set(skip_dirs)
    out: List[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in files:
            if name.lower().endswith(suffixes):
                out.append(Path(root) / name)
    return sorted(out)


def discover_files_from_roots(roots: Iterable[Path], suffixes: Iterable[str], skip_dirs: Iterable[str]) -> List[Path]:
    suffix_tuple = tuple(s.lower() for s in suffixes)
    seen = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root] if root.name.lower().endswith(suffix_tuple) else []
        else:
            candidates = discover_files(root, suffix_tuple, skip_dirs)
        for path in candidates:
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return sorted(out)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def xml_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def snippets_for_terms(text: str, width: int = 140) -> List[str]:
    snippets: List[str] = []
    for match in CALIBRATION_TERMS.finditer(text):
        start = max(0, match.start() - width)
        end = min(len(text), match.end() + width)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= 12:
            break
    return snippets


def classify_path(path: Path, base: Path) -> str:
    parts = {p.lower() for p in path.relative_to(base).parts} if path.is_relative_to(base) else {p.lower() for p in path.parts}
    name = path.name.lower()
    if name == "objectives_observations.md":
        return "project_observation_note"
    if name.endswith((".h5", ".hdf5")):
        return "raw_hdf5"
    if name.endswith((".ppt", ".pptx")):
        return "presentation"
    if name.endswith((".xlsx", ".xls")):
        return "spreadsheet"
    if name.endswith(".csv"):
        if any("echem" in p for p in parts) or "electrochemistry" in name:
            return "electrochemistry_csv"
        return "csv"
    if name.endswith((".txt", ".md")):
        return "text_note"
    return "other"


def evidence_strength(source_class: str, channel: str) -> str:
    if source_class == "raw_hdf5" and channel in {"attribute", "dataset_name"}:
        return "primary_raw_metadata"
    if source_class in {"spreadsheet", "csv", "text_note"}:
        return "project_file_text"
    if source_class == "project_observation_note":
        return "derived_project_note"
    if source_class == "presentation":
        return "slide_text"
    return "weak_context"


def scale_status(nm_per_px: Optional[float]) -> str:
    if nm_per_px is None:
        return ""
    delta = abs(nm_per_px - TARGET_NM_PER_PX) / TARGET_NM_PER_PX
    if delta <= NEAR_FRACTION:
        return "near_96nm_px"
    return "contradicts_96nm_px"


def add_statement(
    rows: List[Dict[str, Any]],
    *,
    path: Path,
    base: Path,
    source_class: str,
    channel: str,
    location: str,
    statement_type: str,
    value: Optional[float],
    unit: str,
    snippet: str,
    inferred_nm_per_px: Optional[float] = None,
    px_dimension: Optional[int] = None,
) -> None:
    rows.append(
        {
            "relative_path": rel(path, base),
            "source_class": source_class,
            "channel": channel,
            "location": location,
            "statement_type": statement_type,
            "value": value,
            "unit": unit,
            "px_dimension": px_dimension,
            "inferred_nm_per_px": inferred_nm_per_px,
            "scale_status": scale_status(inferred_nm_per_px),
            "evidence_strength": evidence_strength(source_class, channel),
            "snippet": re.sub(r"\s+", " ", snippet).strip()[:800],
        }
    )


def extract_statements_from_text(
    rows: List[Dict[str, Any]],
    *,
    path: Path,
    base: Path,
    source_class: str,
    channel: str,
    location: str,
    text: str,
    movie_shapes: Iterable[tuple[int, ...]] = (),
) -> None:
    if not CALIBRATION_TERMS.search(text):
        return
    seen = set()
    for regex in (NM_PER_PIXEL, PIXEL_SIZE_NM):
        for match in regex.finditer(text):
            value = float(match.group("value"))
            key = ("nm_per_pixel", value, match.start())
            if key in seen:
                continue
            seen.add(key)
            add_statement(
                rows,
                path=path,
                base=base,
                source_class=source_class,
                channel=channel,
                location=location,
                statement_type="explicit_nm_per_pixel",
                value=value,
                unit="nm/px",
                inferred_nm_per_px=value,
                snippet=text[max(0, match.start() - 160) : min(len(text), match.end() + 160)],
            )

    for match in FOV_PAIR.finditer(text):
        fov_x = float(match.group("x"))
        fov_y = float(match.group("y"))
        snippet = text[max(0, match.start() - 160) : min(len(text), match.end() + 160)]
        add_statement(
            rows,
            path=path,
            base=base,
            source_class=source_class,
            channel=channel,
            location=location,
            statement_type="fov_pair",
            value=fov_x,
            unit=f"{match.group('unit')} x {match.group('unit')}",
            snippet=snippet,
        )
        for shape in movie_shapes:
            if len(shape) >= 2:
                height = int(shape[-2])
                width = int(shape[-1])
                if width > 0:
                    add_statement(
                        rows,
                        path=path,
                        base=base,
                        source_class=source_class,
                        channel=channel,
                        location=location,
                        statement_type="fov_width_div_movie_width",
                        value=fov_x,
                        unit="um / px",
                        inferred_nm_per_px=1000.0 * fov_x / width,
                        px_dimension=width,
                        snippet=snippet,
                    )
                if height > 0:
                    add_statement(
                        rows,
                        path=path,
                        base=base,
                        source_class=source_class,
                        channel=channel,
                        location=location,
                        statement_type="fov_height_div_movie_height",
                        value=fov_y,
                        unit="um / px",
                        inferred_nm_per_px=1000.0 * fov_y / height,
                        px_dimension=height,
                        snippet=snippet,
                    )

    for match in FPS_OR_EXPOSURE.finditer(text):
        add_statement(
            rows,
            path=path,
            base=base,
            source_class=source_class,
            channel=channel,
            location=location,
            statement_type="time_or_exposure_statement",
            value=float(match.group("value")),
            unit=match.group("unit"),
            snippet=text[max(0, match.start() - 120) : min(len(text), match.end() + 120)],
        )


def read_zip_text(path: Path, members: Iterable[str]) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if any(n.startswith(m) for m in members) and n.endswith(".xml")):
                try:
                    out.append((name, xml_text(zf.read(name).decode("utf-8", errors="ignore"))))
                except Exception as exc:
                    out.append((name, f"ERROR reading XML member: {exc}"))
    except Exception:
        return out
    return out


def audit_text_like(path: Path, base: Path, rows: List[Dict[str, Any]], inventory: List[Dict[str, Any]], movie_shapes: List[tuple[int, ...]]) -> None:
    source_class = classify_path(path, base)
    snippets: List[str] = []
    records_scanned = 0
    error = ""
    try:
        if path.suffix.lower() == ".pptx":
            texts = read_zip_text(path, ["ppt/slides/slide", "ppt/notesSlides/notesSlide"])
            for location, text in texts:
                records_scanned += 1
                if CALIBRATION_TERMS.search(text):
                    snippets.extend(snippets_for_terms(text))
                extract_statements_from_text(
                    rows,
                    path=path,
                    base=base,
                    source_class=source_class,
                    channel="slide_xml",
                    location=location,
                    text=text,
                    movie_shapes=movie_shapes,
                )
        elif path.suffix.lower() == ".xlsx":
            texts = read_zip_text(path, ["xl/worksheets/sheet", "xl/sharedStrings"])
            for location, text in texts:
                records_scanned += 1
                if CALIBRATION_TERMS.search(text):
                    snippets.extend(snippets_for_terms(text))
                extract_statements_from_text(
                    rows,
                    path=path,
                    base=base,
                    source_class=source_class,
                    channel="xlsx_xml",
                    location=location,
                    text=text,
                    movie_shapes=movie_shapes,
                )
        else:
            raw = path.read_bytes()[:262144]
            text = raw.decode("utf-8", errors="ignore")
            records_scanned = 1
            if path.suffix.lower() == ".csv":
                try:
                    sample = raw.decode("utf-8", errors="ignore").splitlines()[:80]
                    text = "\n".join(sample)
                except Exception:
                    pass
            if CALIBRATION_TERMS.search(text):
                snippets.extend(snippets_for_terms(text))
            extract_statements_from_text(
                rows,
                path=path,
                base=base,
                source_class=source_class,
                channel="file_text_sample",
                location="first_256kb",
                text=text,
                movie_shapes=movie_shapes,
            )
    except Exception as exc:
        error = str(exc)

    inventory.append(
        {
            "relative_path": rel(path, base),
            "source_class": source_class,
            "suffix": path.suffix.lower(),
            "file_size_mb": path.stat().st_size / 1e6 if path.exists() else None,
            "records_scanned": records_scanned,
            "has_calibration_terms": bool(snippets),
            "snippet_count": len(snippets),
            "snippets": " || ".join(dict.fromkeys(snippets[:8])),
            "error": error,
        }
    )


def stringify_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20)
    return str(value)


def audit_h5(path: Path, base: Path, rows: List[Dict[str, Any]], inventory: List[Dict[str, Any]]) -> Optional[tuple[int, ...]]:
    n_attrs = 0
    n_attr_hits = 0
    dataset_names: List[str] = []
    movie_shape: Optional[tuple[int, ...]] = None
    error = ""
    source_class = classify_path(path, base)
    try:
        with h5py.File(path, "r") as h5:
            def inspect_obj(name: str, obj: Any) -> None:
                nonlocal n_attrs, n_attr_hits, movie_shape
                if isinstance(obj, h5py.Dataset):
                    dataset_names.append(name)
                    if name.endswith("movie") or name == "movie":
                        movie_shape = tuple(int(x) for x in obj.shape)
                    if CALIBRATION_TERMS.search(name):
                        add_statement(
                            rows,
                            path=path,
                            base=base,
                            source_class=source_class,
                            channel="dataset_name",
                            location=name,
                            statement_type="calibration_like_dataset_name",
                            value=None,
                            unit="",
                            snippet=name,
                        )
                for key, value in obj.attrs.items():
                    n_attrs += 1
                    text = f"{name or '/'} {key} {stringify_attr(value)}"
                    if CALIBRATION_TERMS.search(text):
                        n_attr_hits += 1
                        add_statement(
                            rows,
                            path=path,
                            base=base,
                            source_class=source_class,
                            channel="attribute",
                            location=f"{name or '/'}:{key}",
                            statement_type="calibration_like_attribute",
                            value=None,
                            unit="",
                            snippet=text,
                        )
                        extract_statements_from_text(
                            rows,
                            path=path,
                            base=base,
                            source_class=source_class,
                            channel="attribute",
                            location=f"{name or '/'}:{key}",
                            text=text,
                        )

            inspect_obj("/", h5)
            # Keep this provenance audit shallow and metadata-only. The raw files
            # are large and the calibration blocker is about file-level microscope
            # metadata, top-level datasets, and their attributes.
            for name, obj in h5.items():
                inspect_obj(name, obj)
                if isinstance(obj, h5py.Group):
                    for child_name, child in obj.items():
                        inspect_obj(f"{name}/{child_name}", child)
    except Exception as exc:
        error = str(exc)

    inventory.append(
        {
            "relative_path": rel(path, base),
            "source_class": source_class,
            "suffix": path.suffix.lower(),
            "file_size_mb": path.stat().st_size / 1e6 if path.exists() else None,
            "records_scanned": len(dataset_names),
            "has_calibration_terms": n_attr_hits > 0 or any(CALIBRATION_TERMS.search(n) for n in dataset_names),
            "snippet_count": n_attr_hits,
            "snippets": ";".join(dataset_names[:40]),
            "h5_n_attrs": n_attrs,
            "h5_n_calibration_attr_hits": n_attr_hits,
            "movie_shape": str(movie_shape) if movie_shape else "",
            "error": error,
        }
    )
    return movie_shape


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Calibration Provenance Evidence Audit",
        "",
        "Metadata/text provenance audit for the assumed 96 nm/px optical scale and timing context.",
        "",
        f"- Files inventoried: {summary['n_files_inventoried']}",
        f"- HDF5 files scanned: {summary['n_h5_files_scanned']}",
        f"- Raw HDF5 explicit spatial-scale statements: {summary['n_raw_h5_spatial_scale_statements']}",
        f"- Explicit or inferred near-96 nm/px statements: {summary['n_near_96nm_px_statements']}",
        f"- Contradictory scale statements: {summary['n_contradictory_scale_statements']}",
        f"- Highest evidence class: {summary['highest_scale_evidence_strength']}",
        f"- Provenance status: {summary['provenance_status']}",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Outputs",
        "",
        "- `calibration_provenance_file_inventory.csv`",
        "- `calibration_provenance_evidence_statements.csv`",
        "- `calibration_provenance_summary.json`",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--repo-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_provenance_evidence_audit")
    parser.add_argument(
        "--h5-root",
        action="append",
        default=[],
        help="Raw HDF5 root to scan. May be repeated. Defaults to the full base tree if omitted.",
    )
    parser.add_argument(
        "--text-root",
        action="append",
        default=[],
        help="Project text/spreadsheet/presentation root to scan. May be repeated. Defaults to the full base tree if omitted.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    repo = Path(args.repo_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    skip_dirs = {"derived", "derived_local", ".git", "__pycache__", ".pytest_cache"}
    h5_roots = [Path(x) for x in args.h5_root] or [base]
    text_roots = [Path(x) for x in args.text_root] or [base]
    h5_files = discover_files_from_roots(h5_roots, [".h5", ".hdf5"], skip_dirs)
    text_files = discover_files_from_roots(text_roots, [".pptx", ".xlsx", ".csv", ".txt", ".md"], skip_dirs)
    # Restrict notes/markdown to project-level files rather than generated reports.
    text_files = [
        p
        for p in text_files
        if (
            (repo in p.parents or p == base / "NMC_degradation_3_160623_Halfthedata" / "details.txt")
            or p.parent.name.startswith("NMC_degradation_")
        )
        and (
            p.suffix.lower() not in {".txt", ".md"}
            or p == base / "NMC_degradation_3_160623_Halfthedata" / "details.txt"
            or repo in p.parents
        )
    ]

    rows: List[Dict[str, Any]] = []
    inventory: List[Dict[str, Any]] = []
    movie_shapes: List[tuple[int, ...]] = []

    for path in h5_files:
        shape = audit_h5(path, base, rows, inventory)
        if shape and shape not in movie_shapes:
            movie_shapes.append(shape)

    # If every movie has the same spatial dimensions, this allows slide FoV text
    # to be compared against the raw image dimensions without opening frame data.
    unique_spatial_shapes = sorted({(s[-2], s[-1]) for s in movie_shapes if len(s) >= 2})
    representative_shapes = [(1, h, w) for h, w in unique_spatial_shapes]

    for path in text_files:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            continue
        audit_text_like(path, base, rows, inventory, representative_shapes)

    evidence = pd.DataFrame(rows)
    inv = pd.DataFrame(inventory)
    if not evidence.empty:
        evidence = evidence.sort_values(["evidence_strength", "scale_status", "relative_path", "location"], na_position="last")
    if not inv.empty:
        inv = inv.sort_values(["source_class", "relative_path"])
    evidence.to_csv(out / "calibration_provenance_evidence_statements.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    inv.to_csv(out / "calibration_provenance_file_inventory.csv", index=False)

    if evidence.empty:
        near = pd.DataFrame()
        contradict = pd.DataFrame()
        raw_calibration_like = pd.DataFrame()
        raw_spatial_scale = pd.DataFrame()
    else:
        near = evidence[evidence["scale_status"] == "near_96nm_px"]
        contradict = evidence[evidence["scale_status"] == "contradicts_96nm_px"]
        raw_calibration_like = evidence[
            (evidence["source_class"] == "raw_hdf5")
            & (
                evidence["statement_type"].astype(str).str.contains("calibration_like", case=False, na=False)
                | evidence["statement_type"].astype(str).str.contains("time_or_exposure", case=False, na=False)
            )
        ]
        raw_spatial_scale = evidence[
            (evidence["source_class"] == "raw_hdf5")
            & (pd.to_numeric(evidence["inferred_nm_per_px"], errors="coerce").notna())
        ]

    strength_rank = {
        "primary_raw_metadata": 5,
        "project_file_text": 4,
        "slide_text": 3,
        "derived_project_note": 2,
        "weak_context": 1,
    }
    independent_near = near[near.get("evidence_strength", pd.Series(dtype=str)) != "derived_project_note"] if len(near) else near
    if len(independent_near):
        highest = max(independent_near["evidence_strength"], key=lambda x: strength_rank.get(str(x), 0))
    elif len(near):
        highest = max(near["evidence_strength"], key=lambda x: strength_rank.get(str(x), 0))
    else:
        highest = "none"

    if len(raw_spatial_scale) and len(near[near["evidence_strength"] == "primary_raw_metadata"]):
        status = "raw_metadata_supported_96nm_px"
    elif len(near) and not len(contradict):
        status = "slide_or_project_text_supported_but_raw_metadata_blocked"
    elif len(near) and len(contradict):
        status = "mixed_or_contradictory_scale_evidence"
    else:
        status = "no_scale_provenance_found"

    if status == "slide_or_project_text_supported_but_raw_metadata_blocked":
        interpretation = (
            "The 96 nm/px assumption is supported only by slide/project-text evidence and by FoV divided by raw movie "
            "dimensions being approximately compatible with that scale. No raw HDF5 attribute or microscope metadata "
            "statement was found, so calibrated diffusion constants remain blocked; use apparent optical-front units."
        )
    elif status == "raw_metadata_supported_96nm_px":
        interpretation = (
            "Raw metadata contains near-96 nm/px evidence. This clears the spatial provenance component, but diffusion "
            "claims still need timing stability, manual front QC, and control-balanced sanity gates."
        )
    elif status == "mixed_or_contradictory_scale_evidence":
        interpretation = (
            "The audit found both near-96 nm/px and contradictory scale statements. Treat calibrated spatial units as "
            "unresolved until the source-specific microscope calibration is manually reconciled."
        )
    else:
        interpretation = (
            "No usable 96 nm/px provenance statement was found in scanned files. Keep all front/diffusion-like values "
            "as pixel or apparent optical proxies."
        )

    source_counts = inv["source_class"].value_counts().to_dict() if not inv.empty else {}
    statement_counts = evidence["statement_type"].value_counts().to_dict() if not evidence.empty else {}
    summary = {
        "target_nm_per_px": TARGET_NM_PER_PX,
        "near_fraction_tolerance": NEAR_FRACTION,
        "n_files_inventoried": int(len(inv)),
        "n_h5_files_scanned": int((inv.get("source_class", pd.Series(dtype=str)) == "raw_hdf5").sum()) if not inv.empty else 0,
        "file_source_class_counts": {str(k): int(v) for k, v in source_counts.items()},
        "unique_movie_spatial_shapes": [list(x) for x in unique_spatial_shapes],
        "n_evidence_statements": int(len(evidence)),
        "statement_type_counts": {str(k): int(v) for k, v in statement_counts.items()},
        "n_raw_h5_calibration_like_statements": int(len(raw_calibration_like)),
        "n_raw_h5_spatial_scale_statements": int(len(raw_spatial_scale)),
        "n_near_96nm_px_statements": int(len(near)),
        "n_contradictory_scale_statements": int(len(contradict)),
        "highest_scale_evidence_strength": highest,
        "provenance_status": status,
        "near_96nm_px_examples": (independent_near if len(independent_near) else near).head(12).to_dict("records") if len(near) else [],
        "contradictory_examples": contradict.head(12).to_dict("records") if len(contradict) else [],
        "interpretation": interpretation,
        "outputs": {
            "file_inventory": str(out / "calibration_provenance_file_inventory.csv"),
            "evidence_statements": str(out / "calibration_provenance_evidence_statements.csv"),
            "summary": str(out / "calibration_provenance_summary.json"),
            "readme": str(out / "README.md"),
        },
    }
    with (out / "calibration_provenance_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)
    write_readme(out, summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
