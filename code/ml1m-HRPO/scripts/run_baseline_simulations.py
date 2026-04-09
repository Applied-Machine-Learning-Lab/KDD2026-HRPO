"""
Run a batch of Agent4Rec simulations (baseline recommenders) and aggregate metrics.

Example:
  python scripts/run_baseline_simulations.py --dataset all-beauty --models Random Pop --auto_suffix
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def try_acquire_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "cwd": str(REPO_ROOT)}, ensure_ascii=False))
    return True


def _safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return value


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
        metrics[key] = _safe_float(val)
    return metrics


def compute_click_exit_from_behavior(sim_dir: Path) -> Dict[str, Any]:
    """
    Compute key online metrics from `behavior/*.pkl` to avoid relying on
    legacy `metrics.txt` definitions (e.g., old click-rate denominators).

    Returns keys compatible with `metrics.txt`:
    - "Overall click rate"
    - "Average exit page"
    """

    beh_dir = sim_dir / "behavior"
    if not beh_dir.exists():
        return {}

    per_user_ctr: List[float] = []
    per_user_exit: List[int] = []

    for pkl_path in beh_dir.glob("*.pkl"):
        try:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        page_keys = sorted([k for k in data.keys() if isinstance(k, int)])
        if not page_keys:
            continue

        n_pages = len(page_keys)
        # Infer items_per_page from payload (fallback to 1).
        items_per_page = 1
        first = data.get(page_keys[0], {})
        if isinstance(first, dict):
            rid = first.get("recommended_id")
            if isinstance(rid, list) and len(rid) > 0:
                items_per_page = int(len(rid))

        clicks = 0
        for k in page_keys:
            info = data.get(k, {})
            if not isinstance(info, dict):
                continue
            wid = info.get("watch_id", [])
            if isinstance(wid, list):
                clicks += int(len(wid))

        exposures = max(int(n_pages) * int(items_per_page), 1)
        per_user_ctr.append(float(clicks) / float(exposures))
        per_user_exit.append(int(n_pages))

    if not per_user_ctr:
        return {}

    overall_ctr = float(sum(per_user_ctr) / len(per_user_ctr))
    avg_exit = float(sum(per_user_exit) / len(per_user_exit)) if per_user_exit else 0.0
    return {"Overall click rate": overall_ctr, "Average exit page": avg_exit}


def shlex_join(cmd: List[str]) -> str:
    # Minimal cross-platform pretty print.
    out = []
    for token in cmd:
        if re.search(r"\s", token):
            out.append(f"\"{token}\"")
        else:
            out.append(token)
    return " ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all-beauty")
    parser.add_argument("--models", nargs="+", default=["Random", "Pop"])
    parser.add_argument("--simulation_name", type=str, default="Baseline")
    parser.add_argument("--auto_suffix", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Saved",
        help="Weights subdir under recommenders/weights/<dataset>/<model>/ (e.g., Saved, Saved_e10).",
    )
    parser.add_argument(
        "--model_path_override",
        action="append",
        default=[],
        help="Per-model override in the form Model=Saved_subdir. Can be repeated.",
    )

    parser.add_argument("--n_avatars", type=int, default=20)
    parser.add_argument("--max_pages", type=int, default=3)
    parser.add_argument("--items_per_page", type=int, default=4)
    parser.add_argument("--execution_mode", type=str, default="parallel", choices=["serial", "parallel"])
    parser.add_argument("--seed", type=int, default=101)

    parser.add_argument("--llm_model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--llm_api_style", type=str, default=os.getenv("LLM_API_STYLE", "chat_completions"))
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        help="Avatar LLM temperature. Use 0 for stable evaluation.",
    )
    parser.add_argument(
        "--beauty_prompt_mode",
        type=str,
        default=os.getenv("BEAUTY_PROMPT_MODE", "a"),
        choices=["a", "b", "c"],
        help="Beauty prompt mode: a=single-stage, b=two-stage (scan+decide), c=strict-intent (mission-driven, conservative clicks).",
    )
    parser.add_argument("--enable_hazard_plan", action="store_true")
    parser.add_argument("--hazard_plan_dir", type=str, default=os.getenv("HAZARD_PLAN_DIR", "Saved"))
    parser.add_argument("--hazard_plan_candidate_pool", type=int, default=50)
    parser.add_argument(
        "--hazard_plan_override",
        type=str,
        default=os.getenv("HAZARD_PLAN_OVERRIDE", "auto"),
        choices=["auto", "safe_match", "recover", "explore", "balanced"],
    )

    # Convenience: set env vars for the subprocess.
    parser.add_argument("--openai_api_base", type=str, default=os.getenv("OPENAI_API_BASE", ""))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))

    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_path_overrides: Dict[str, str] = {}
    for raw in args.model_path_override:
        if "=" not in raw:
            raise ValueError(f"Invalid --model_path_override: {raw!r}. Expected Model=Saved_subdir.")
        model_name, model_path = raw.split("=", 1)
        model_name = model_name.strip().lower()
        model_path = model_path.strip()
        if not model_name or not model_path:
            raise ValueError(f"Invalid --model_path_override: {raw!r}. Expected Model=Saved_subdir.")
        model_path_overrides[model_name] = model_path

    run_name = args.simulation_name
    if args.auto_suffix:
        run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = REPO_ROOT / "storage" / args.dataset / "baseline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{run_name}_summary.json"
    out_csv = out_dir / f"{run_name}_summary.csv"
    lock_path = out_dir / f"{run_name}.lock"

    if out_json.exists() and out_csv.exists():
        print(f"[skip] summary already exists for run={run_name}")
        print(f"[skip] {out_json}")
        print(f"[skip] {out_csv}")
        return 0

    lock_acquired = try_acquire_lock(lock_path)
    if not lock_acquired:
        print(f"[skip] run is already active elsewhere: {run_name}")
        print(f"[skip] lock={lock_path}")
        return 0

    env = os.environ.copy()
    if args.openai_api_base:
        env["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key

    try:
        results: List[Dict[str, Any]] = []
        for model in args.models:
            model_path = model_path_overrides.get(str(model).lower(), args.model_path)
            cmd = [
                args.python,
                str(REPO_ROOT / "main.py"),
                "--dataset",
                args.dataset,
                "--modeltype",
                model,
                "--model_path",
                str(model_path),
                "--simulation_name",
                run_name,
                "--n_avatars",
                str(args.n_avatars),
                "--max_pages",
                str(args.max_pages),
                "--items_per_page",
                str(args.items_per_page),
                "--execution_mode",
                args.execution_mode,
                "--seed",
                str(args.seed),
                "--llm_model",
                args.llm_model,
                "--llm_api_style",
                args.llm_api_style,
                "--llm_temperature",
                str(args.llm_temperature),
                "--beauty_prompt_mode",
                args.beauty_prompt_mode,
            ]
            if args.enable_hazard_plan:
                cmd.extend(
                    [
                        "--enable_hazard_plan",
                        "--hazard_plan_dir",
                        str(args.hazard_plan_dir),
                        "--hazard_plan_candidate_pool",
                        str(args.hazard_plan_candidate_pool),
                        "--hazard_plan_override",
                        str(args.hazard_plan_override),
                    ]
                )

            print(f"[run] {args.dataset} {model} -> simulation_name={run_name} model_path={model_path}")
            print(shlex_join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

            metrics_path = REPO_ROOT / "storage" / args.dataset / model / run_name / "metrics.txt"
            metrics = parse_metrics_txt(metrics_path)
            # Prefer behavior-derived CTR/exit for consistency across code versions.
            behavior_metrics = compute_click_exit_from_behavior(metrics_path.parent)
            metrics.update(behavior_metrics)
            metrics["dataset"] = args.dataset
            metrics["modeltype"] = model
            metrics["model_path"] = model_path
            metrics["simulation_name"] = run_name
            metrics["metrics_path"] = str(metrics_path)
            metrics["seed"] = args.seed
            metrics["llm_model"] = args.llm_model
            metrics["llm_api_style"] = args.llm_api_style
            metrics["llm_temperature"] = args.llm_temperature
            metrics["beauty_prompt_mode"] = args.beauty_prompt_mode
            metrics["n_avatars_requested"] = args.n_avatars
            metrics["max_pages_requested"] = args.max_pages
            metrics["items_per_page_requested"] = args.items_per_page
            metrics["execution_mode"] = args.execution_mode
            metrics["hazard_plan_enabled"] = bool(args.enable_hazard_plan)
            results.append(metrics)

        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Union of keys across result rows.
        columns = sorted({k for row in results for k in row.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"[saved] {out_json}")
        print(f"[saved] {out_csv}")
        return 0
    finally:
        if lock_acquired and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
