"""
Compute extended behavior metrics from Agent4Rec simulation logs.

This script reads one or more simulation run directories under:
  storage/<dataset>/<model>/<simulation_name>/
and aggregates behavior/*.pkl (+ optional interview/*.pkl + metrics.txt).

It supports two input modes:
1) --run_dirs <dir1> <dir2> ...
2) --metrics_csv <csv with a metrics_path column>
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_first_float(text: Any, default: float = 0.0) -> float:
    if text is None:
        return float(default)
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else float(default)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _to_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist() if _safe_int(v) is not None]
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            iv = _safe_int(v)
            if iv is not None:
                out.append(iv)
        return out
    iv = _safe_int(x)
    return [iv] if iv is not None else []


def _to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.quantile(np.asarray(xs, dtype=np.float32), q))


def parse_metrics_txt(path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not path.exists():
        return metrics
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]+)\s*:\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip()
        metrics[key] = val
    return metrics


def parse_run_components(run_dir: Path) -> Tuple[str, str, str]:
    # Expected: storage/<dataset>/<model>/<simulation_name>
    parts = run_dir.resolve().parts
    if "storage" in parts:
        i = parts.index("storage")
        if len(parts) >= i + 4:
            return parts[i + 1], parts[i + 2], parts[i + 3]
    # Fallback to directory names.
    name = run_dir.name
    model = run_dir.parent.name if run_dir.parent else ""
    dataset = run_dir.parent.parent.name if run_dir.parent and run_dir.parent.parent else ""
    return dataset, model, name


@dataclass
class UserAgg:
    user_id: int
    pages: int = 0
    exposures: int = 0
    watches: int = 0
    aligns: int = 0
    likes: int = 0
    dislike_proxy: int = 0
    gt_exposures: int = 0
    gt_hit_align: int = 0
    gt_hit_like: int = 0
    rating_sum: float = 0.0
    rating_cnt: int = 0
    align_reason_chars: int = 0
    align_reason_cnt: int = 0
    watch_reason_chars: int = 0
    watch_reason_cnt: int = 0
    firstn_recommended: List[int] = None  # type: ignore[assignment]
    interview_rating: float = 0.0
    interview_has_rating: int = 0

    def __post_init__(self):
        if self.firstn_recommended is None:
            self.firstn_recommended = []


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_interview_rating(interview_obj: Any) -> Optional[float]:
    if not isinstance(interview_obj, dict):
        return None
    arr = interview_obj.get("interview")
    if not isinstance(arr, list) or not arr:
        return None
    # Stored as regex-captured tail text, e.g. " 6; REASON: ..."
    score = _extract_first_float(arr[0], default=-1.0)
    if 1.0 <= score <= 10.0:
        return float(score)
    return None


def compute_extended_metrics_for_run(
    run_dir: Path,
    topn: int,
    method_name: str = "",
) -> Dict[str, Any]:
    behavior_dir = run_dir / "behavior"
    interview_dir = run_dir / "interview"
    metrics_txt = run_dir / "metrics.txt"
    if not behavior_dir.exists():
        raise FileNotFoundError(f"behavior directory not found: {behavior_dir}")

    dataset, model, simulation_name = parse_run_components(run_dir)
    raw_metrics = parse_metrics_txt(metrics_txt)

    behavior_files = sorted(behavior_dir.glob("*.pkl"), key=lambda p: _safe_int(p.stem) or 10**9)
    if not behavior_files:
        raise RuntimeError(f"no behavior pickle files found in: {behavior_dir}")

    per_user: List[UserAgg] = []
    retention_hist: Dict[int, int] = {}
    exit_pages: List[int] = []

    for bf in behavior_files:
        uid = _safe_int(bf.stem)
        if uid is None:
            continue
        user = UserAgg(user_id=uid)
        behavior_obj = _load_pickle(bf)
        if not isinstance(behavior_obj, dict):
            continue

        page_keys = sorted(behavior_obj.keys(), key=lambda x: int(x))
        user.pages = len(page_keys)
        exit_pages.append(float(user.pages))
        retention_hist[user.pages] = retention_hist.get(user.pages, 0) + 1

        for pk in page_keys:
            info = behavior_obj.get(pk, {}) or {}
            rec_ids = _to_int_list(info.get("recommended_id"))
            align_ids = _to_int_list(info.get("align_id"))
            like_ids = _to_int_list(info.get("like_id"))
            watch_ids = _to_int_list(info.get("watch_id"))
            watched_titles = _to_str_list(info.get("watched"))
            gt_ids = _to_int_list(info.get("ground_truth"))

            rating_ids = _to_int_list(info.get("rating_id"))
            ratings = []
            for r in info.get("rating", []) if isinstance(info.get("rating", []), list) else []:
                ir = _safe_int(r)
                if ir is not None:
                    ratings.append(int(ir))
            rating_map: Dict[int, int] = {}
            for iid, r in zip(rating_ids, ratings):
                rating_map[int(iid)] = int(r)

            align_set = set(align_ids)
            like_set = set(like_ids)
            watch_set = set(watch_ids)
            gt_set = set(gt_ids)

            user.exposures += len(rec_ids)
            user.watches += len(watched_titles) if watched_titles else len(watch_ids)
            user.aligns += len(align_ids)
            user.likes += len(like_ids)

            if gt_set:
                user.gt_exposures += 1
                if align_set & gt_set:
                    user.gt_hit_align += 1
                if like_set & gt_set:
                    user.gt_hit_like += 1

            for rid in rec_ids:
                aligned = rid in align_set
                rating = rating_map.get(rid)
                disliked = (not aligned) or (rating is not None and rating <= 2)
                user.dislike_proxy += int(disliked)

                if len(user.firstn_recommended) < topn:
                    user.firstn_recommended.append(int(rid))

            for r in ratings:
                user.rating_sum += float(r)
                user.rating_cnt += 1

            align_reasons = _to_str_list(info.get("align_reason"))
            watch_reason = str(info.get("watch_reason", "") or "")
            for reason in align_reasons:
                rc = len(reason.strip())
                if rc > 0:
                    user.align_reason_chars += rc
                    user.align_reason_cnt += 1
            rc_watch = len(watch_reason.strip())
            if rc_watch > 0:
                user.watch_reason_chars += rc_watch
                user.watch_reason_cnt += 1

        iv_path = interview_dir / f"{uid}.pkl"
        if iv_path.exists():
            iv_obj = _load_pickle(iv_path)
            iv_score = _parse_interview_rating(iv_obj)
            if iv_score is not None:
                user.interview_rating = float(iv_score)
                user.interview_has_rating = 1

        per_user.append(user)

    n_users = len(per_user)
    total_pages = int(sum(u.pages for u in per_user))
    total_exposures = int(sum(u.exposures for u in per_user))
    total_watch = int(sum(u.watches for u in per_user))
    total_align = int(sum(u.aligns for u in per_user))
    total_like = int(sum(u.likes for u in per_user))
    total_dislike_proxy = int(sum(u.dislike_proxy for u in per_user))
    total_ratings = int(sum(u.rating_cnt for u in per_user))
    sum_ratings = float(sum(u.rating_sum for u in per_user))
    sum_align_reason_chars = int(sum(u.align_reason_chars for u in per_user))
    sum_align_reason_cnt = int(sum(u.align_reason_cnt for u in per_user))
    sum_watch_reason_chars = int(sum(u.watch_reason_chars for u in per_user))
    sum_watch_reason_cnt = int(sum(u.watch_reason_cnt for u in per_user))
    gt_positive_pages = int(sum(u.gt_exposures for u in per_user))
    gt_hit_align_pages = int(sum(u.gt_hit_align for u in per_user))
    gt_hit_like_pages = int(sum(u.gt_hit_like for u in per_user))

    # If metrics.txt has max pages, re-use it for capacity-normalized CTR.
    max_pages_cfg = int(_extract_first_float(raw_metrics.get("Maximum exit page"), default=0.0))
    mean_items_per_page = float(total_exposures / max(total_pages, 1))
    ctr_by_capacity = 0.0
    if n_users > 0 and max_pages_cfg > 0:
        denom = float(n_users * max_pages_cfg * max(mean_items_per_page, 1e-6))
        ctr_by_capacity = float(total_watch / denom)

    # Retention curve by reached page.
    max_exit = max((u.pages for u in per_user), default=0)
    retention_curve: Dict[str, float] = {}
    for p in range(1, max_exit + 1):
        reached = sum(1 for u in per_user if u.pages >= p)
        retention_curve[str(p)] = float(reached / max(n_users, 1))

    # Top-N exposure overlap statistics.
    firstn_sets = [set(u.firstn_recommended[:topn]) for u in per_user]
    pair_scores: List[float] = []
    for a, b in itertools.combinations(firstn_sets, 2):
        if topn <= 0:
            continue
        pair_scores.append(float(len(a & b) / float(topn)))
    topn_unique = len(set(itertools.chain.from_iterable([u.firstn_recommended[:topn] for u in per_user])))
    topn_unique_ratio = float(topn_unique / max(topn * max(n_users, 1), 1))

    interview_ratings = [u.interview_rating for u in per_user if u.interview_has_rating > 0]

    out: Dict[str, Any] = {
        "method": method_name,
        "dataset": dataset,
        "model": model,
        "simulation_name": simulation_name,
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_txt.resolve()) if metrics_txt.exists() else "",
        "n_users": int(n_users),
        "n_pages": int(total_pages),
        "mean_pages_from_behavior": float(total_pages / max(n_users, 1)),
        "avg_exit_page": _mean([float(u.pages) for u in per_user]),
        "median_exit_page": float(statistics.median([u.pages for u in per_user])) if per_user else 0.0,
        "p90_exit_page": _quantile([float(u.pages) for u in per_user], 0.90),
        "max_exit_page_observed": int(max_exit),
        "items_per_page_inferred": mean_items_per_page,
        "watch_yes_rate": float(total_watch / max(total_exposures, 1)),
        "align_yes_rate": float(total_align / max(total_exposures, 1)),
        "like_yes_rate": float(total_like / max(total_exposures, 1)),
        "dislike_proxy_rate": float(total_dislike_proxy / max(total_exposures, 1)),
        "avg_rating_watched": float(sum_ratings / max(total_ratings, 1)),
        "avg_align_reason_chars": float(sum_align_reason_chars / max(sum_align_reason_cnt, 1)),
        "avg_watch_reason_chars": float(sum_watch_reason_chars / max(sum_watch_reason_cnt, 1)),
        "gt_positive_page_rate": float(gt_positive_pages / max(total_pages, 1)),
        "gt_hit_align_given_gt": float(gt_hit_align_pages / max(gt_positive_pages, 1)),
        "gt_hit_like_given_gt": float(gt_hit_like_pages / max(gt_positive_pages, 1)),
        "topn": int(topn),
        "topn_pair_overlap_mean": _mean(pair_scores),
        "topn_unique_count": int(topn_unique),
        "topn_unique_ratio": topn_unique_ratio,
        "retention_curve": retention_curve,
        "interview_rating_mean": _mean(interview_ratings),
        "interview_rating_count": int(len(interview_ratings)),
        # Backward-compatible references from metrics.txt if present.
        "avg_recall": _extract_first_float(raw_metrics.get("Average recall"), default=0.0),
        "avg_precision": _extract_first_float(raw_metrics.get("Average presion"), default=0.0),
        "overall_click_rate_reported": _extract_first_float(raw_metrics.get("Overall click rate"), default=0.0),
        "avg_likes_reported": _extract_first_float(raw_metrics.get("Average number of likes"), default=0.0),
        "avg_exit_page_reported": _extract_first_float(raw_metrics.get("Average exit page"), default=0.0),
        "total_k_tokens": _extract_first_float(raw_metrics.get("Total k tokens"), default=0.0),
        "total_cost": _extract_first_float(raw_metrics.get("Total cost"), default=0.0),
        "total_sim_time_s": _extract_first_float(raw_metrics.get("Total simulation time"), default=0.0),
        # New CTR normalization variants.
        "click_rate_actual_pages": float(total_watch / max(total_exposures, 1)),
        "click_rate_capacity_norm": float(ctr_by_capacity),
    }

    # Include sparse exit page histogram for debugging.
    out["exit_page_hist"] = {str(k): int(v) for k, v in sorted(retention_hist.items(), key=lambda kv: kv[0])}
    return out


def _load_runs_from_metrics_csv(path: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            metrics_path = (r.get("metrics_path") or "").strip()
            if not metrics_path:
                continue
            mpath = Path(metrics_path)
            if not mpath.is_absolute():
                mpath = (REPO_ROOT / mpath).resolve()
            rows.append((r.get("method", "") or r.get("model", ""), mpath.parent))
    return rows


def _default_output_paths(run_rows: List[Tuple[str, Path]], out_json: str, out_csv: str) -> Tuple[Path, Path]:
    if out_json and out_csv:
        return Path(out_json), Path(out_csv)
    if len(run_rows) == 1 and not out_json and not out_csv:
        base = run_rows[0][1]
        return base / "metrics_extended.json", base / "metrics_extended.csv"
    ts = time.strftime("%Y%m%d_%H%M%S")
    default_dir = REPO_ROOT / "baseline" / "results"
    default_dir.mkdir(parents=True, exist_ok=True)
    if not out_json:
        out_json = str(default_dir / f"metrics_extended_summary_{ts}.json")
    if not out_csv:
        out_csv = str(default_dir / f"metrics_extended_summary_{ts}.csv")
    return Path(out_json), Path(out_csv)


def _flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    # Flatten compact dict fields.
    for key in ["retention_curve", "exit_page_hist"]:
        val = out.get(key)
        out[key] = json.dumps(val, ensure_ascii=False, separators=(",", ":")) if isinstance(val, dict) else str(val)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute extended behavior metrics from Agent4Rec simulation logs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="One or more simulation run directories.")
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="Optional CSV containing a metrics_path column (e.g. baseline/results/*_vs_*.csv).",
    )
    parser.add_argument("--topn", type=int, default=20, help="Top-N exposure window for overlap/diversity stats.")
    parser.add_argument("--out_json", type=str, default="", help="Output JSON path.")
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV path.")
    args = parser.parse_args()

    run_rows: List[Tuple[str, Path]] = []
    for d in args.run_dirs:
        run_rows.append(("", (REPO_ROOT / d).resolve() if not Path(d).is_absolute() else Path(d).resolve()))
    if args.metrics_csv:
        mpath = Path(args.metrics_csv)
        if not mpath.is_absolute():
            mpath = (REPO_ROOT / mpath).resolve()
        run_rows.extend(_load_runs_from_metrics_csv(mpath))

    if not run_rows:
        raise ValueError("No runs provided. Use --run_dirs or --metrics_csv.")

    # Remove duplicates while preserving order.
    seen = set()
    dedup_rows: List[Tuple[str, Path]] = []
    for method, rd in run_rows:
        key = str(rd.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append((method, rd))
    run_rows = dedup_rows

    results: List[Dict[str, Any]] = []
    for method, rd in run_rows:
        row = compute_extended_metrics_for_run(run_dir=rd, topn=max(int(args.topn), 1), method_name=method)
        if method and not row.get("method"):
            row["method"] = method
        results.append(row)
        print(
            "[ok]",
            row.get("method") or row.get("model", ""),
            f"users={row['n_users']}",
            f"pages={row['n_pages']}",
            f"click_actual={row['click_rate_actual_pages']:.4f}",
            f"exit={row['avg_exit_page']:.3f}",
        )

    out_json_path, out_csv_path = _default_output_paths(run_rows, args.out_json, args.out_csv)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = [_flatten_for_csv(r) for r in results]
    columns = sorted({k for r in csv_rows for k in r.keys()})
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"[saved] {out_json_path}")
    print(f"[saved] {out_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
