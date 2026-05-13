#!/usr/bin/env python3
"""Summarize validation logs and enforce the PR-worthiness gates for this candidate."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ARTIFACT_CAP_BYTES = 16_000_000
TRAIN_BUDGET_MS = 600_000
EVAL_BUDGET_SECONDS = 600.0


SEED_RE = re.compile(r"train_seed(\d+)(?:_[^.]+)?\.log$")
TTT_RE = re.compile(
    r"quantized_ttt_phased\s+val_loss:(?P<loss>[0-9.]+)\s+"
    r"val_bpb:(?P<bpb>[0-9.]+)\s+eval_time:(?P<eval_ms>[0-9.]+)ms"
)
SIZE_RE = re.compile(r"Total submission size quantized\+pergroup:\s+(?P<size>[0-9]+)\s+bytes")
TOTAL_EVAL_RE = re.compile(r"total_eval_time:(?P<eval_s>[0-9.]+)s")
TRAIN_TIME_RE = re.compile(r"stopping_early:.*train_time:\s+(?P<train_ms>[0-9]+)ms")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    seed_match = SEED_RE.search(path.name)
    seed = int(seed_match.group(1)) if seed_match else None
    row = {
        "path": str(path),
        "seed": seed,
        "val_loss": None,
        "val_bpb": None,
        "artifact_bytes": None,
        "eval_seconds": None,
        "train_ms": None,
        "ngram_precompute_inside_eval_timer": None,
        "ngram_precompute_done": False,
        "failures": [],
    }

    if "ngram_hint_precompute_outside: False" in text:
        row["ngram_precompute_inside_eval_timer"] = True
    elif "ngram_hint_precompute_outside: True" in text:
        row["ngram_precompute_inside_eval_timer"] = False

    row["ngram_precompute_done"] = "ngram_tilt:precompute_done" in text

    ttt_matches = list(TTT_RE.finditer(text))
    if ttt_matches:
        m = ttt_matches[-1]
        row["val_loss"] = float(m.group("loss"))
        row["val_bpb"] = float(m.group("bpb"))
        row["eval_seconds"] = float(m.group("eval_ms")) / 1000.0

    total_eval_matches = list(TOTAL_EVAL_RE.finditer(text))
    if total_eval_matches:
        row["eval_seconds"] = float(total_eval_matches[-1].group("eval_s"))

    size_matches = list(SIZE_RE.finditer(text))
    if size_matches:
        row["artifact_bytes"] = int(size_matches[-1].group("size"))

    train_matches = list(TRAIN_TIME_RE.finditer(text))
    if train_matches:
        row["train_ms"] = int(train_matches[-1].group("train_ms"))

    for key in ["seed", "val_loss", "val_bpb", "artifact_bytes", "eval_seconds", "train_ms"]:
        if row[key] is None:
            row["failures"].append(f"missing_{key}")
    if row["artifact_bytes"] is not None and row["artifact_bytes"] > ARTIFACT_CAP_BYTES:
        row["failures"].append("artifact_over_cap")
    if row["eval_seconds"] is not None and row["eval_seconds"] > EVAL_BUDGET_SECONDS:
        row["failures"].append("eval_over_budget")
    if row["train_ms"] is not None and row["train_ms"] > TRAIN_BUDGET_MS:
        row["failures"].append("train_over_budget")
    if row["ngram_precompute_inside_eval_timer"] is not True:
        row["failures"].append("ngram_precompute_not_inside_eval_timer")
    if not row["ngram_precompute_done"]:
        row["failures"].append("missing_ngram_precompute_done")

    return row


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def summarize(rows: list[dict], required_seeds: list[int]) -> dict:
    failures: list[str] = []
    seen = {}
    for row in rows:
        seed = row.get("seed")
        if seed is not None and seed in seen:
            failures.append(f"duplicate_seed_{seed}")
        if seed is not None:
            seen[seed] = row
        failures.extend(f"seed_{seed}:{f}" for f in row.get("failures", []))

    for seed in required_seeds:
        if seed not in seen:
            failures.append(f"missing_seed_{seed}")

    complete_rows = [seen[s] for s in required_seeds if s in seen]
    bpbs = [r["val_bpb"] for r in complete_rows if r.get("val_bpb") is not None]
    losses = [r["val_loss"] for r in complete_rows if r.get("val_loss") is not None]
    artifacts = [r["artifact_bytes"] for r in complete_rows if r.get("artifact_bytes") is not None]
    evals = [r["eval_seconds"] for r in complete_rows if r.get("eval_seconds") is not None]
    trains = [r["train_ms"] for r in complete_rows if r.get("train_ms") is not None]

    summary = {
        "ok": len(failures) == 0,
        "required_seeds": required_seeds,
        "seed_count": len(complete_rows),
        "mean_bpb": sum(bpbs) / len(bpbs) if bpbs else None,
        "std_bpb": _sample_std(bpbs),
        "mean_val_loss": sum(losses) / len(losses) if losses else None,
        "max_artifact_bytes": max(artifacts) if artifacts else None,
        "max_eval_seconds": max(evals) if evals else None,
        "max_train_ms": max(trains) if trains else None,
        "failures": failures,
        "rows": complete_rows,
    }
    return summary


def print_summary(summary: dict) -> None:
    print("seed val_bpb val_loss artifact_bytes eval_s train_ms")
    for row in summary["rows"]:
        print(
            f"{row['seed']} {row['val_bpb']:.8f} {row['val_loss']:.8f} "
            f"{row['artifact_bytes']} {row['eval_seconds']:.1f} {row['train_ms']}"
        )
    print("")
    print(f"ok: {summary['ok']}")
    print(f"seed_count: {summary['seed_count']}")
    if summary["mean_bpb"] is not None:
        print(f"mean_bpb: {summary['mean_bpb']:.8f}")
        print(f"std_bpb: {summary['std_bpb']:.8f}")
    print(f"max_artifact_bytes: {summary['max_artifact_bytes']}")
    print(f"max_eval_seconds: {summary['max_eval_seconds']}")
    print(f"max_train_ms: {summary['max_train_ms']}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"  - {failure}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", type=Path, default=sorted(Path(".").glob("train_seed*.log")))
    ap.add_argument("--required-seeds", nargs="+", type=int, default=[42, 0, 314])
    ap.add_argument("--write-json", type=Path, default=Path("validation_summary.json"))
    args = ap.parse_args()

    rows = [parse_log(path) for path in args.logs]
    summary = summarize(rows, args.required_seeds)
    print_summary(summary)
    if args.write_json:
        args.write_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
