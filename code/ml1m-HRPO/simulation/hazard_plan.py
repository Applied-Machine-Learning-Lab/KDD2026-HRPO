from __future__ import annotations

import math
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    "page_index",
    "page_progress",
    "max_pages",
    "activity_group",
    "conformity_group",
    "diversity_group",
    "exit_threshold",
    "proxy_negative_count",
    "watched_pages_so_far",
    "aligned_pages_so_far",
    "watched_items_so_far",
    "aligned_items_so_far",
    "recent_watch_rate_3",
    "recent_align_rate_3",
    "recent_avg_rating_3",
    "pages_since_last_watch",
    "pages_since_last_align",
    "mission_restock",
    "mission_explore",
    "item_rating",
    "item_review_count_log1p",
    "item_price_log1p",
    "item_has_price",
    "item_quality",
    "focus_match",
    "pref_match_count",
    "pref_match_any",
    "item_low_rating",
    "item_low_reviews",
    "item_high_reviews",
    "item_high_rating",
    "plan_safe_match",
    "plan_recover",
    "plan_explore",
    "plan_balanced",
]

PLANNER_FEATURE_NAMES: List[str] = [
    "page_index",
    "page_progress",
    "max_pages",
    "activity_group",
    "conformity_group",
    "diversity_group",
    "exit_threshold",
    "proxy_negative_count",
    "watched_pages_so_far",
    "aligned_pages_so_far",
    "watched_items_so_far",
    "aligned_items_so_far",
    "recent_watch_rate_3",
    "recent_align_rate_3",
    "recent_avg_rating_3",
    "pages_since_last_watch",
    "pages_since_last_align",
    "mission_restock",
    "mission_explore",
]

PLAN_OPTIONS: Tuple[str, ...] = ("safe_match", "recover", "explore", "balanced")
CF_OVERRIDE_MAX_PRIOR: float = 0.58
CF_OVERRIDE_MIN_GAIN: float = 0.035
ATTR_BASE_SCALE: float = 0.16
ATTR_MARGIN_SCALE: float = 0.08
ATTR_SURVIVAL_SAT_PENALTY: float = 0.05
MULTIOBJ_BLEND_SCALE: float = 0.18
MULTIOBJ_PARETO_SCALE: float = 0.08
MULTIOBJ_STD_PENALTY: float = 0.04
PLAN_SCORE_TEMPERATURE: float = 0.75
ATTRV2_ADV_SCALE: float = 0.12
ATTRV2_POS_SCALE: float = 0.05
ATTRV2_NEG_SCALE: float = 0.10
ATTRV2_NEG_MEMORY_SCALE: float = 0.14
ATTRV2_ITEM_REPEAT_SCALE: float = 0.10
ATTRV2_MEMORY_DECAY: float = 0.82
ATTRV3_HISTORY_SCALE: float = 0.06
ATTRV3_OT_SCALE: float = 0.10
ATTRV3_CF_SCALE: float = 0.05
ATTRV3_DECODE_SCALE: float = 0.05
ATTRV3_NEG_HISTORY_SCALE: float = 0.04
ATTRV3_TRANSPORT_RESIDUAL_SCALE: float = 0.03
ATTRV3_TRANSPORT_STD_FLOOR: float = 0.10
ATTRV3_TRANSPORT_Z_CLIP: float = 1.50
ATTRV3_TRANSPORT_MIN_PAGE_CREDIT: float = -0.12
ATTRV3_TRANSPORT_MAX_EXIT_PROB: float = 0.30
ATTRV3_TRANSPORT_ALLOWED_PLANS = {"explore", "balanced"}


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


ATTRV3_ENABLE_HISTORY_TERM: bool = _env_flag("ATTRV3_ENABLE_HISTORY", True)
ATTRV3_ENABLE_OT_TERM: bool = _env_flag("ATTRV3_ENABLE_OT", True)
ATTRV3_ENABLE_CF_TERM: bool = _env_flag("ATTRV3_ENABLE_CF", True)
ATTRV3_ENABLE_DECODE_TERM: bool = _env_flag("ATTRV3_ENABLE_DECODE", True)
ATTRV3_ENABLE_NEG_HISTORY_TERM: bool = _env_flag("ATTRV3_ENABLE_NEG_HISTORY", True)
ATTRV3_ENABLE_TRANSPORT_RESIDUAL: bool = _env_flag("ATTRV3_ENABLE_TRANSPORT_RESIDUAL", False)
OPTION_PRIOR_LOG_SCALE: float = 0.07
OPTION_DWELL_SCALE: float = 0.10
OPTION_STAY_SCALE: float = 0.08
OPTION_SWITCH_BASE: float = 0.035
EXITFIRST_STYLE_PROFILES = {
    "modeA_exitfirst",
    "modeA_tiger_exitfirst",
    "modeA_hindsight",
    "modeA_tiger_hindsight",
    "modeA_multiobj",
    "modeA_tiger_multiobj",
    "modeA_valuemix",
    "modeA_tiger_valuemix",
    "modeA_valuemix_anchor",
    "modeA_tiger_valuemix_anchor",
    "modeA_scopecf",
    "modeA_tiger_scopecf",
    "modeA_scopecf_gate",
    "modeA_tiger_scopecf_gate",
    "modeA_attr",
    "modeA_tiger_attr",
    "modeA_attrv2",
    "modeA_tiger_attrv2",
    "modeA_attrv3",
    "modeA_tiger_attrv3",
    "modeA_option",
    "modeA_tiger_option",
}
ATTR_PROFILES = {"modeA_attr", "modeA_tiger_attr"}
ATTRV2_PROFILES = {"modeA_attrv2", "modeA_tiger_attrv2"}
ATTRV3_PROFILES = {"modeA_attrv3", "modeA_tiger_attrv3"}
ATTR_SESSION_CREDIT_PROFILES = ATTRV2_PROFILES | ATTRV3_PROFILES
MULTIOBJ_PROFILES = {"modeA_multiobj", "modeA_tiger_multiobj"}
VALUEMIX_PROFILES = {
    "modeA_valuemix",
    "modeA_tiger_valuemix",
    "modeA_valuemix_anchor",
    "modeA_tiger_valuemix_anchor",
}
VALUEMIX_ANCHOR_PROFILES = {"modeA_valuemix_anchor", "modeA_tiger_valuemix_anchor"}
OPTION_PROFILES = {"modeA_option", "modeA_tiger_option"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def harmonic_mean_positive(values: Sequence[float], eps: float = 1e-6) -> float:
    vals = [max(float(v), eps) for v in values]
    if not vals:
        return 0.0
    denom = sum(1.0 / v for v in vals)
    if denom <= 0:
        return 0.0
    return float(len(vals) / denom)


def minmax_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-6:
        return [0.5 for _ in arr]
    return [float((v - lo) / (hi - lo)) for v in arr]


def softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-6)
    arr = np.asarray(values, dtype=np.float32) / temp
    arr = arr - float(np.max(arr))
    exp_arr = np.exp(arr)
    denom = float(np.sum(exp_arr))
    if denom <= 0:
        return [1.0 / len(values) for _ in values]
    return [float(v / denom) for v in exp_arr.tolist()]


def to_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [parse_int(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [parse_int(v) for v in value]
    return [parse_int(value)]


def split_tags(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def canonical_product_type(text: str) -> Optional[str]:
    t = clean_text(text).lower()
    if not t:
        return None
    if any(k in t for k in ["cleanser", "cleanse", "cleansing", "face wash"]):
        return "cleanser"
    if any(k in t for k in ["hair", "shampoo", "conditioner", "scalp"]):
        return "hair care"
    if "mask" in t:
        return "mask"
    if any(k in t for k in ["makeup", "mascara", "eyeliner", "foundation", "bb cream", "concealer", "lip", "eyelash"]):
        return "makeup"
    if "nail" in t:
        return "nail care"
    if any(k in t for k in ["fragrance", "perfume", "scent"]):
        return "fragrance"
    if any(k in t for k in ["sunscreen", "spf", "sun care"]):
        return "sun care"
    if any(k in t for k in ["body wash", "bath", "scrub", "body care"]):
        return "body care"
    if any(k in t for k in ["serum", "moistur", "cream", "toner", "oil", "lotion", "skincare", "skin care"]):
        return "skincare"
    return None


def parse_review_count(summary: Any) -> int:
    text = clean_text(summary)
    m = re.search(r"review count\s*:\s*([0-9,]+)", text, flags=re.IGNORECASE)
    if not m:
        return 0
    return parse_int(m.group(1).replace(",", ""), 0)


def parse_price(summary: Any) -> float:
    text = clean_text(summary)
    m = re.search(r"price\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not m:
        return 0.0
    return parse_float(m.group(1), 0.0)


def infer_preferred_types(taste_lines: Sequence[str], high_rating_text: str = "") -> List[str]:
    prefs: List[str] = []
    for raw in taste_lines:
        m = re.search(r"prefer\s+(.+?)\s+products", str(raw), flags=re.IGNORECASE)
        cat = m.group(1).strip().strip(".") if m else str(raw)
        if ":" in cat:
            continue
        canon = canonical_product_type(cat)
        if canon and canon not in prefs:
            prefs.append(canon)
    if not prefs:
        m = re.search(r"tends to rate\s+(.+?)\s+products\s+highly", str(high_rating_text), flags=re.IGNORECASE)
        if m:
            for part in str(m.group(1)).split(","):
                cat = part.strip().strip(".")
                if not cat or ":" in cat:
                    continue
                canon = canonical_product_type(cat)
                if canon and canon not in prefs:
                    prefs.append(canon)
    if not prefs:
        prefs.append("skincare")
    return prefs


def build_user_profile(persona_row: Any, user_stat_row: Any, avatar_id: int) -> Dict[str, Any]:
    taste_raw = clean_text(getattr(persona_row, "taste", persona_row.get("taste", "")))
    taste_lines = [part.strip() for part in taste_raw.split("|") if part.strip()]
    high_rating = clean_text(getattr(persona_row, "high_rating", persona_row.get("high_rating", "")))
    preferred_types = infer_preferred_types(taste_lines, high_rating)
    focus = preferred_types[int(avatar_id) % len(preferred_types)]

    activity_group = parse_int(getattr(user_stat_row, "activity", user_stat_row.get("activity", 2)), 2)
    conformity_group = parse_int(getattr(user_stat_row, "conformity", user_stat_row.get("conformity", 2)), 2)
    diversity_group = parse_int(getattr(user_stat_row, "diversity", user_stat_row.get("diversity", 2)), 2)
    mission = "restock" if activity_group <= 2 else "explore"
    exit_threshold = {1: 1, 2: 2, 3: 3}.get(activity_group, 2)
    return {
        "activity_group": activity_group,
        "conformity_group": conformity_group,
        "diversity_group": diversity_group,
        "exit_threshold": exit_threshold,
        "mission": mission,
        "focus": focus,
        "preferred_types": preferred_types,
    }


def build_item_catalog(movie_detail: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if "movie_id" in movie_detail.columns:
        item_ids = movie_detail["movie_id"].astype(int).tolist()
    else:
        item_ids = list(range(len(movie_detail)))

    catalog: Dict[int, Dict[str, Any]] = {}
    for idx, item_id in enumerate(item_ids):
        row = movie_detail.iloc[idx]
        tags_raw = split_tags(getattr(row, "genres", ""))
        tags = [canonical_product_type(tag) or clean_text(tag).lower() for tag in tags_raw if clean_text(tag)]
        rating = parse_float(getattr(row, "rating", 0.0), 0.0)
        review_count = parse_review_count(getattr(row, "summary", ""))
        price = parse_price(getattr(row, "summary", ""))
        quality = 0.65 * max(min(rating / 5.0, 1.0), 0.0) + 0.35 * max(
            min(math.log1p(review_count) / math.log1p(500.0), 1.0),
            0.0,
        )
        catalog[int(item_id)] = {
            "item_id": int(item_id),
            "rating": rating,
            "review_count": review_count,
            "price": price,
            "has_price": 1.0 if price > 0 else 0.0,
            "quality": quality,
            "tags": set(tags),
            "title": clean_text(getattr(row, "title", "")),
        }
    return catalog


def choose_plan(user_profile: Dict[str, Any], state: Dict[str, float], override: str = "auto") -> str:
    if override and override != "auto":
        return override
    if state.get("proxy_negative_count", 0.0) >= 1.0:
        return "recover"
    if state.get("recent_watch_rate_3", 0.0) <= 0.05 and state.get("page_index", 1.0) > 1.0:
        return "recover"
    if user_profile.get("mission") == "restock":
        return "safe_match"
    if state.get("page_progress", 0.0) <= 0.35 and int(user_profile.get("diversity_group", 2)) >= 2:
        return "explore"
    return "balanced"


def build_planner_feature_row(user_profile: Dict[str, Any], state: Dict[str, float]) -> Dict[str, float]:
    return {
        "page_index": float(state.get("page_index", 1.0)),
        "page_progress": float(state.get("page_progress", 0.0)),
        "max_pages": float(state.get("max_pages", 1.0)),
        "activity_group": float(user_profile.get("activity_group", 2)),
        "conformity_group": float(user_profile.get("conformity_group", 2)),
        "diversity_group": float(user_profile.get("diversity_group", 2)),
        "exit_threshold": float(user_profile.get("exit_threshold", 2)),
        "proxy_negative_count": float(state.get("proxy_negative_count", 0.0)),
        "watched_pages_so_far": float(state.get("watched_pages_so_far", 0.0)),
        "aligned_pages_so_far": float(state.get("aligned_pages_so_far", 0.0)),
        "watched_items_so_far": float(state.get("watched_items_so_far", 0.0)),
        "aligned_items_so_far": float(state.get("aligned_items_so_far", 0.0)),
        "recent_watch_rate_3": float(state.get("recent_watch_rate_3", 0.0)),
        "recent_align_rate_3": float(state.get("recent_align_rate_3", 0.0)),
        "recent_avg_rating_3": float(state.get("recent_avg_rating_3", 0.0)),
        "pages_since_last_watch": float(state.get("pages_since_last_watch", 0.0)),
        "pages_since_last_align": float(state.get("pages_since_last_align", 0.0)),
        "mission_restock": 1.0 if user_profile.get("mission") == "restock" else 0.0,
        "mission_explore": 1.0 if user_profile.get("mission") == "explore" else 0.0,
    }


def planner_row_to_vector(row: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(row.get(name, 0.0)) for name in PLANNER_FEATURE_NAMES], dtype=np.float32)


def infer_hindsight_plan_label(
    user_profile: Dict[str, Any],
    state: Dict[str, float],
    current_outcome: Dict[str, float],
    future_outcomes: Sequence[Dict[str, float]],
) -> str:
    current_positive = current_outcome.get("watch_count", 0.0) > 0 and current_outcome.get("avg_rating", 0.0) >= 4.0
    future_positive = any(
        out.get("watch_count", 0.0) > 0 and out.get("avg_rating", 0.0) >= 4.0 for out in future_outcomes[1:]
    )
    recover_context = (
        float(state.get("proxy_negative_count", 0.0)) >= 1.0
        or (
            float(state.get("page_index", 1.0)) > 1.0
            and float(state.get("recent_watch_rate_3", 0.0)) <= 0.01
        )
    )
    if recover_context and future_positive:
        return "recover"

    if current_positive:
        if int(user_profile.get("mission") == "restock") or int(user_profile.get("diversity_group", 2)) <= 1:
            return "safe_match"
        if int(user_profile.get("diversity_group", 2)) >= 3 and float(state.get("page_progress", 0.0)) <= 0.4:
            return "explore"

    if (
        user_profile.get("mission") == "explore"
        and int(user_profile.get("diversity_group", 2)) >= 2
        and float(state.get("page_progress", 0.0)) <= 0.45
        and (current_positive or future_positive)
    ):
        return "explore"

    if (
        float(state.get("page_progress", 0.0)) >= 0.6
        and current_outcome.get("watch_count", 0.0) <= 0
    ):
        return "safe_match"

    return "balanced"


def discounted_sum(values: Sequence[float], gamma: float = 0.72) -> float:
    total = 0.0
    weight = 1.0
    for value in values:
        total += weight * float(value)
        weight *= float(gamma)
    return float(total)


def infer_hindsight_plan_label_v2(
    user_profile: Dict[str, Any],
    state: Dict[str, float],
    current_outcome: Dict[str, float],
    future_outcomes: Sequence[Dict[str, float]],
) -> str:
    if not future_outcomes:
        return infer_hindsight_plan_label(user_profile, state, current_outcome, future_outcomes)

    mission = str(user_profile.get("mission", "restock"))
    diversity_group = int(user_profile.get("diversity_group", 2))
    progress = float(state.get("page_progress", 0.0))
    neg = float(state.get("proxy_negative_count", 0.0))
    recent_watch = float(state.get("recent_watch_rate_3", 0.0))
    pages_since_watch = float(state.get("pages_since_last_watch", 0.0))

    success_seq = [float(out.get("success_score", 0.0)) for out in future_outcomes]
    focus_seq = [float(out.get("focus_score", 0.0)) for out in future_outcomes]
    novelty_seq = [float(out.get("novelty_score", 0.0)) for out in future_outcomes]
    quality_seq = [float(out.get("quality_score", 0.0)) for out in future_outcomes]

    current_success = success_seq[0]
    future_success = discounted_sum(success_seq)
    future_focus = discounted_sum(focus_seq)
    future_novelty = discounted_sum(novelty_seq)
    future_quality = discounted_sum(quality_seq)
    continuation = min(max(len(future_outcomes) - 1, 0), 3) / 3.0
    rebound_gain = max(future_success - current_success, 0.0)

    recover_context = (
        neg >= 1.0
        or recent_watch <= 0.05
        or pages_since_watch >= 2.0
        or (progress >= 0.35 and current_success <= 0.05)
    )

    recover_score = (
        (0.72 if recover_context else 0.18) * future_success
        + 0.18 * rebound_gain
        + 0.10 * continuation
    )
    safe_score = (
        0.48 * future_focus
        + 0.22 * future_quality
        + 0.18 * future_success
        + 0.12 * continuation
        + (0.08 if mission == "restock" else 0.0)
        + (0.05 if diversity_group <= 1 else 0.0)
    )
    explore_score = (
        0.52 * future_novelty
        + 0.22 * future_success
        + 0.14 * max(1.0 - progress, 0.0)
        + (0.08 if mission == "explore" else 0.0)
        + (0.06 if diversity_group >= 3 else 0.0)
    )
    balanced_score = (
        0.36 * future_success
        + 0.24 * future_focus
        + 0.14 * future_novelty
        + 0.14 * future_quality
        + 0.12 * continuation
    )

    if progress >= 0.65:
        safe_score += 0.06 * future_focus
        balanced_score += 0.04 * future_quality
        explore_score -= 0.04 * future_novelty

    scores = {
        "recover": float(recover_score),
        "safe_match": float(safe_score),
        "explore": float(explore_score),
        "balanced": float(balanced_score),
    }

    if recover_context and recover_score >= max(safe_score, balanced_score) - 0.04:
        return "recover"
    if mission == "restock" and safe_score >= max(explore_score, balanced_score) - 0.02:
        return "safe_match"
    if mission == "explore" and diversity_group >= 2 and explore_score >= max(safe_score, balanced_score) - 0.02:
        return "explore"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def compute_hindsight_plan_scores_v3(
    user_profile: Dict[str, Any],
    state: Dict[str, float],
    current_outcome: Dict[str, float],
    future_outcomes: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    if not future_outcomes:
        fallback = infer_hindsight_plan_label_v2(user_profile, state, current_outcome, future_outcomes)
        return {plan: 1.0 if plan == fallback else 0.0 for plan in PLAN_OPTIONS}

    mission = str(user_profile.get("mission", "restock"))
    diversity_group = int(user_profile.get("diversity_group", 2))
    progress = float(state.get("page_progress", 0.0))
    neg = float(state.get("proxy_negative_count", 0.0))
    recent_watch = float(state.get("recent_watch_rate_3", 0.0))
    pages_since_watch = float(state.get("pages_since_last_watch", 0.0))
    current_success = float(current_outcome.get("success_score", 0.0))

    success_seq = [float(out.get("success_score", 0.0)) for out in future_outcomes]
    focus_seq = [float(out.get("focus_score", 0.0)) for out in future_outcomes]
    novelty_seq = [float(out.get("novelty_score", 0.0)) for out in future_outcomes]
    quality_seq = [float(out.get("quality_score", 0.0)) for out in future_outcomes]

    future_success = discounted_sum(success_seq)
    future_focus = discounted_sum(focus_seq)
    future_novelty = discounted_sum(novelty_seq)
    future_quality = discounted_sum(quality_seq)
    continuation = min(max(len(future_outcomes) - 1, 0), 4) / 4.0
    rebound_gain = max(future_success - current_success, 0.0)

    recover_context = (
        neg >= 1.0
        or (
            float(state.get("page_index", 1.0)) > 1.0
            and recent_watch <= 0.02
        )
        or (pages_since_watch >= 2.0 and current_success <= 0.10)
    )

    recover_score = (
        (0.58 if recover_context else 0.06) * future_success
        + 0.28 * rebound_gain
        + 0.14 * continuation
    )
    safe_score = (
        0.44 * future_focus
        + 0.22 * future_quality
        + 0.18 * future_success
        + 0.10 * continuation
        + (0.08 if mission == "restock" else 0.0)
        + (0.04 if diversity_group <= 1 else 0.0)
    )
    explore_score = (
        0.44 * future_novelty
        + 0.22 * future_success
        + 0.12 * max(1.0 - progress, 0.0)
        + (0.08 if mission == "explore" else 0.0)
        + (0.06 if diversity_group >= 3 else 0.0)
    )
    balanced_score = (
        0.32 * future_success
        + 0.22 * future_focus
        + 0.18 * future_novelty
        + 0.14 * future_quality
        + 0.14 * continuation
    )

    if current_success >= 0.45 and not recover_context:
        balanced_score += 0.08 * current_success
    if future_success >= 0.45 and abs(future_focus - future_novelty) <= 0.06:
        balanced_score += 0.08
    if 0.2 <= progress <= 0.75:
        balanced_score += 0.04 * future_quality
    if progress >= 0.65:
        safe_score += 0.04 * future_focus
        balanced_score += 0.03 * future_quality
        explore_score -= 0.03 * future_novelty

    return {
        "recover": float(recover_score),
        "safe_match": float(safe_score),
        "explore": float(explore_score),
        "balanced": float(balanced_score),
    }


def infer_hindsight_plan_label_v3(
    user_profile: Dict[str, Any],
    state: Dict[str, float],
    current_outcome: Dict[str, float],
    future_outcomes: Sequence[Dict[str, float]],
) -> str:
    scores = compute_hindsight_plan_scores_v3(
        user_profile=user_profile,
        state=state,
        current_outcome=current_outcome,
        future_outcomes=future_outcomes,
    )
    future_success = discounted_sum([float(out.get("success_score", 0.0)) for out in future_outcomes])
    future_focus = discounted_sum([float(out.get("focus_score", 0.0)) for out in future_outcomes])
    future_novelty = discounted_sum([float(out.get("novelty_score", 0.0)) for out in future_outcomes])
    current_success = float(current_outcome.get("success_score", 0.0))
    rebound_gain = max(future_success - current_success, 0.0)
    recover_context = (
        float(state.get("proxy_negative_count", 0.0)) >= 1.0
        or (
            float(state.get("page_index", 1.0)) > 1.0
            and float(state.get("recent_watch_rate_3", 0.0)) <= 0.02
        )
        or (float(state.get("pages_since_last_watch", 0.0)) >= 2.0 and current_success <= 0.10)
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_plan, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1e9
    margin = float(top_score - second_score)

    if recover_context and top_plan == "recover" and (rebound_gain >= 0.16 or margin >= 0.08):
        return "recover"
    if margin < 0.05:
        return "balanced"
    if top_plan == "recover" and rebound_gain < 0.10:
        return "balanced"
    if top_plan == "safe_match" and future_focus < 0.16 and future_success < 0.36:
        return "balanced"
    if top_plan == "explore" and future_novelty < 0.10:
        return "balanced"
    return str(top_plan)


def build_feature_row(
    user_profile: Dict[str, Any],
    state: Dict[str, float],
    item_info: Dict[str, Any],
    plan: str,
) -> Dict[str, float]:
    preferred_types = set(user_profile.get("preferred_types", []))
    item_tags = set(item_info.get("tags", set()))
    focus_match = 1.0 if user_profile.get("focus", "") in item_tags else 0.0
    pref_match_count = float(len(preferred_types & item_tags))
    pref_match_any = 1.0 if pref_match_count > 0 else 0.0
    review_count = float(item_info.get("review_count", 0.0))
    price = float(item_info.get("price", 0.0))
    rating = float(item_info.get("rating", 0.0))
    row = {
        "page_index": float(state.get("page_index", 1.0)),
        "page_progress": float(state.get("page_progress", 0.0)),
        "max_pages": float(state.get("max_pages", 1.0)),
        "activity_group": float(user_profile.get("activity_group", 2)),
        "conformity_group": float(user_profile.get("conformity_group", 2)),
        "diversity_group": float(user_profile.get("diversity_group", 2)),
        "exit_threshold": float(user_profile.get("exit_threshold", 2)),
        "proxy_negative_count": float(state.get("proxy_negative_count", 0.0)),
        "watched_pages_so_far": float(state.get("watched_pages_so_far", 0.0)),
        "aligned_pages_so_far": float(state.get("aligned_pages_so_far", 0.0)),
        "watched_items_so_far": float(state.get("watched_items_so_far", 0.0)),
        "aligned_items_so_far": float(state.get("aligned_items_so_far", 0.0)),
        "recent_watch_rate_3": float(state.get("recent_watch_rate_3", 0.0)),
        "recent_align_rate_3": float(state.get("recent_align_rate_3", 0.0)),
        "recent_avg_rating_3": float(state.get("recent_avg_rating_3", 0.0)),
        "pages_since_last_watch": float(state.get("pages_since_last_watch", 0.0)),
        "pages_since_last_align": float(state.get("pages_since_last_align", 0.0)),
        "mission_restock": 1.0 if user_profile.get("mission") == "restock" else 0.0,
        "mission_explore": 1.0 if user_profile.get("mission") == "explore" else 0.0,
        "item_rating": rating,
        "item_review_count_log1p": math.log1p(max(review_count, 0.0)),
        "item_price_log1p": math.log1p(max(price, 0.0)),
        "item_has_price": float(item_info.get("has_price", 0.0)),
        "item_quality": float(item_info.get("quality", 0.0)),
        "focus_match": focus_match,
        "pref_match_count": pref_match_count,
        "pref_match_any": pref_match_any,
        "item_low_rating": 1.0 if rating > 0 and rating < 4.0 else 0.0,
        "item_low_reviews": 1.0 if review_count > 0 and review_count < 50.0 else 0.0,
        "item_high_reviews": 1.0 if review_count >= 200.0 else 0.0,
        "item_high_rating": 1.0 if rating >= 4.4 else 0.0,
        "plan_safe_match": 1.0 if plan == "safe_match" else 0.0,
        "plan_recover": 1.0 if plan == "recover" else 0.0,
        "plan_explore": 1.0 if plan == "explore" else 0.0,
        "plan_balanced": 1.0 if plan == "balanced" else 0.0,
    }
    return row


def row_to_vector(row: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(row.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)


def update_rollout_state(
    state: Dict[str, Any],
    align_count: int,
    watch_count: int,
    avg_rating: float,
    negative_increment: int,
) -> None:
    state["watch_flags"].append(1 if watch_count > 0 or avg_rating > 0 else 0)
    state["align_flags"].append(1 if align_count > 0 else 0)
    state["avg_ratings"].append(float(avg_rating))
    state["watch_counts"].append(int(watch_count))
    state["align_counts"].append(int(align_count))
    state["watched_items_so_far"] += int(watch_count)
    state["aligned_items_so_far"] += int(align_count)
    state["watched_pages_so_far"] += 1 if watch_count > 0 or avg_rating > 0 else 0
    state["aligned_pages_so_far"] += 1 if align_count > 0 else 0
    if watch_count > 0 or avg_rating > 0:
        state["pages_since_last_watch"] = 0
    else:
        state["pages_since_last_watch"] += 1
    if align_count > 0:
        state["pages_since_last_align"] = 0
    else:
        state["pages_since_last_align"] += 1
    state["proxy_negative_count"] += int(negative_increment)


def summarize_rollout_state(
    state: Dict[str, Any],
    page_index: int,
    max_pages: int,
) -> Dict[str, float]:
    recent_watch = state["watch_flags"][-3:]
    recent_align = state["align_flags"][-3:]
    recent_ratings = [r for r in state["avg_ratings"][-3:] if r > 0]
    return {
        "page_index": float(page_index),
        "page_progress": float(page_index / max(max_pages, 1)),
        "max_pages": float(max_pages),
        "proxy_negative_count": float(state["proxy_negative_count"]),
        "watched_pages_so_far": float(state["watched_pages_so_far"]),
        "aligned_pages_so_far": float(state["aligned_pages_so_far"]),
        "watched_items_so_far": float(state["watched_items_so_far"]),
        "aligned_items_so_far": float(state["aligned_items_so_far"]),
        "recent_watch_rate_3": mean_or_zero(recent_watch),
        "recent_align_rate_3": mean_or_zero(recent_align),
        "recent_avg_rating_3": mean_or_zero(recent_ratings),
        "pages_since_last_watch": float(state["pages_since_last_watch"]),
        "pages_since_last_align": float(state["pages_since_last_align"]),
    }


def initial_rollout_state() -> Dict[str, Any]:
    return {
        "watch_flags": [],
        "align_flags": [],
        "avg_ratings": [],
        "watch_counts": [],
        "align_counts": [],
        "proxy_negative_count": 0,
        "watched_pages_so_far": 0,
        "aligned_pages_so_far": 0,
        "watched_items_so_far": 0,
        "aligned_items_so_far": 0,
        "pages_since_last_watch": 0,
        "pages_since_last_align": 0,
    }


def heuristic_negative_increment(watch_count: int, avg_rating: float) -> int:
    if watch_count <= 0:
        return 1
    if avg_rating > 0 and avg_rating < 4.0:
        return 1
    return 0


class HazardPlanReranker:
    def __init__(
        self,
        *,
        dataset: str,
        movie_detail: pd.DataFrame,
        persona_df: pd.DataFrame,
        user_statistic: pd.DataFrame,
        artifact_dir: Optional[Path],
        candidate_pool: int = 50,
        override_plan: str = "auto",
    ) -> None:
        self.dataset = dataset
        self.movie_detail = movie_detail
        self.persona_df = persona_df
        self.user_statistic = user_statistic
        self.item_catalog = build_item_catalog(movie_detail)
        self.candidate_pool = max(int(candidate_pool), 1)
        self.override_plan = override_plan or "auto"
        self.artifact_dir = artifact_dir
        self.model = None
        self.enjoy_model = None
        self.planner_model = None
        self.planner_dwell_model = None
        self.credit_value_model = None
        self.credit_model = None
        self.planner_model_kind = "classifier"
        self.metadata: Dict[str, Any] = {}
        self.score_profile = "default"
        self.plan_memory: Dict[int, Dict[str, Any]] = {}
        self.credit_memory: Dict[int, Dict[str, Any]] = {}
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "hazard_model.pkl"
            enjoy_model_path = Path(artifact_dir) / "enjoy_model.pkl"
            planner_model_path = Path(artifact_dir) / "planner_model.pkl"
            planner_dwell_model_path = Path(artifact_dir) / "planner_dwell_model.pkl"
            credit_value_model_path = Path(artifact_dir) / "credit_value_model.pkl"
            credit_model_path = Path(artifact_dir) / "credit_model.pkl"
            meta_path = Path(artifact_dir) / "hazard_model_meta.pkl"
            if model_path.exists():
                with model_path.open("rb") as f:
                    self.model = pickle.load(f)
            if enjoy_model_path.exists():
                with enjoy_model_path.open("rb") as f:
                    self.enjoy_model = pickle.load(f)
            if planner_model_path.exists():
                with planner_model_path.open("rb") as f:
                    self.planner_model = pickle.load(f)
            if planner_dwell_model_path.exists():
                with planner_dwell_model_path.open("rb") as f:
                    self.planner_dwell_model = pickle.load(f)
            if credit_value_model_path.exists():
                with credit_value_model_path.open("rb") as f:
                    self.credit_value_model = pickle.load(f)
            if credit_model_path.exists():
                with credit_model_path.open("rb") as f:
                    self.credit_model = pickle.load(f)
            if meta_path.exists():
                with meta_path.open("rb") as f:
                    self.metadata = pickle.load(f)
        self.planner_model_kind = str(self.metadata.get("planner_model_kind", "classifier"))
        self.score_profile = str(self.metadata.get("score_profile", self.score_profile or "default"))
        if self.artifact_dir is not None:
            artifact_name = Path(self.artifact_dir).name.lower()
            profile_aliases = [
                ("modeA_tiger_valuemix_anchor", ("modea_tiger_valuemix_anchor", "scope_valuemix_anchor_tiger")),
                ("modeA_valuemix_anchor", ("modea_valuemix_anchor", "scope_valuemix_anchor")),
                ("modeA_tiger_attrv3", ("modea_tiger_attrv3", "scope_attrv3_tiger")),
                ("modeA_attrv3", ("modea_attrv3", "scope_attrv3")),
                ("modeA_tiger_attrv2", ("modea_tiger_attrv2", "scope_attrv2_tiger")),
                ("modeA_attrv2", ("modea_attrv2", "scope_attrv2")),
                ("modeA_tiger_option", ("modea_tiger_option", "scope_option_tiger")),
                ("modeA_option", ("modea_option", "scope_option")),
                ("modeA_tiger_valuemix", ("modea_tiger_valuemix", "scope_valuemix_tiger")),
                ("modeA_valuemix", ("modea_valuemix", "scope_valuemix")),
                ("modeA_tiger_multiobj", ("modea_tiger_multiobj", "scope_multiobj_tiger")),
                ("modeA_multiobj", ("modea_multiobj", "scope_multiobj")),
                ("modeA_tiger_hindsight", ("modea_tiger_hindsight", "scope_hindsight_tiger")),
                ("modeA_hindsight", ("modea_hindsight", "scope_hindsight")),
                ("modeA_tiger_attr", ("modea_tiger_attr", "scope_attr_tiger")),
                ("modeA_attr", ("modea_attr", "scope_attr")),
                ("modeA_tiger_scopecf_gate", ("modea_tiger_scopecf_gate",)),
                ("modeA_scopecf_gate", ("modea_scopecf_gate", "scope_cf_gate")),
                ("modeA_tiger_scopecf", ("modea_tiger_scopecf",)),
                ("modeA_scopecf", ("modea_scopecf", "scope_cf")),
            ]
            for profile_name, aliases in profile_aliases:
                if any(alias in artifact_name for alias in aliases):
                    self.score_profile = profile_name
                    break
            else:
                if self.score_profile == "default" and "exitfirst" in artifact_name:
                    self.score_profile = "modeA_tiger_exitfirst"
                elif self.score_profile == "default" and "modea_tiger" in artifact_name:
                    self.score_profile = "modeA_tiger"

    def get_user_profile(self, avatar_id: int) -> Dict[str, Any]:
        if avatar_id in self.user_statistic.index:
            stat_row = self.user_statistic.loc[avatar_id]
        else:
            stat_row = self.user_statistic.iloc[int(avatar_id)]
        if hasattr(self.persona_df, "iloc"):
            persona_row = self.persona_df.iloc[int(avatar_id)]
        else:
            persona_row = self.persona_df.loc[int(avatar_id)]
        return build_user_profile(persona_row, stat_row, int(avatar_id))

    def build_runtime_state(self, arena: Any, avatar_id: int, page_index: int) -> Dict[str, float]:
        avatar = arena.avatars[avatar_id]
        ratings = list(arena.ratings.get(avatar_id, []))
        aligns = list(arena.n_likes.get(avatar_id, []))
        state = initial_rollout_state()
        for page_idx, avg_rating in enumerate(ratings, start=1):
            align_count = parse_int(aligns[page_idx - 1] if page_idx - 1 < len(aligns) else 0, 0)
            watch_count = 1 if parse_float(avg_rating, 0.0) > 0 else 0
            neg_inc = heuristic_negative_increment(watch_count, parse_float(avg_rating, 0.0))
            update_rollout_state(
                state,
                align_count=align_count,
                watch_count=watch_count,
                avg_rating=parse_float(avg_rating, 0.0),
                negative_increment=neg_inc,
            )
        summary = summarize_rollout_state(state, page_index=page_index, max_pages=int(arena.max_pages))
        summary["proxy_negative_count"] = float(getattr(avatar, "negative_feedback_count", summary["proxy_negative_count"]))
        return summary

    def predict_exit_prob(self, batch: np.ndarray) -> np.ndarray:
        if self.model is None:
            rating = batch[:, FEATURE_NAMES.index("item_rating")]
            review_log = batch[:, FEATURE_NAMES.index("item_review_count_log1p")]
            focus = batch[:, FEATURE_NAMES.index("focus_match")]
            neg = batch[:, FEATURE_NAMES.index("proxy_negative_count")]
            progress = batch[:, FEATURE_NAMES.index("page_progress")]
            risk = 1.8 - 0.7 * rating - 0.18 * review_log - 0.4 * focus + 0.65 * neg + 0.9 * progress
            return 1.0 / (1.0 + np.exp(-risk))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(batch)[:, 1]
        raw = self.model.predict(batch)
        return np.asarray(raw, dtype=np.float32)

    def predict_enjoy_prob(self, batch: np.ndarray) -> np.ndarray:
        if self.enjoy_model is None:
            quality = batch[:, FEATURE_NAMES.index("item_quality")]
            focus = batch[:, FEATURE_NAMES.index("focus_match")]
            pref_match = batch[:, FEATURE_NAMES.index("pref_match_any")]
            low_rating = batch[:, FEATURE_NAMES.index("item_low_rating")]
            low_reviews = batch[:, FEATURE_NAMES.index("item_low_reviews")]
            progress = batch[:, FEATURE_NAMES.index("page_progress")]
            logits = (
                -1.3
                + 1.35 * quality
                + 0.95 * focus
                + 0.45 * pref_match
                - 0.70 * low_rating
                - 0.35 * low_reviews
                - 0.15 * progress
            )
            return 1.0 / (1.0 + np.exp(-logits))
        if hasattr(self.enjoy_model, "predict_proba"):
            return self.enjoy_model.predict_proba(batch)[:, 1]
        raw = self.enjoy_model.predict(batch)
        return np.asarray(raw, dtype=np.float32)

    def predict_credit_outputs(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows = int(batch.shape[0]) if hasattr(batch, "shape") else 0
        if self.credit_model is None or n_rows <= 0:
            zeros = np.zeros(n_rows, dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy()
        raw = np.asarray(self.credit_model.predict(batch), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        adv = raw[:, 0].astype(np.float32) if raw.shape[1] >= 1 else np.zeros(n_rows, dtype=np.float32)
        pos = np.maximum(raw[:, 1].astype(np.float32), 0.0) if raw.shape[1] >= 2 else np.maximum(adv, 0.0)
        neg = np.maximum(raw[:, 2].astype(np.float32), 0.0) if raw.shape[1] >= 3 else np.maximum(-adv, 0.0)
        return adv, pos, neg

    def _ensure_credit_memory(self, avatar_id: int) -> Dict[str, Any]:
        avatar_id = int(avatar_id)
        if avatar_id not in self.credit_memory:
            self.credit_memory[avatar_id] = {
                "processed_pages": 0,
                "last_negative_count": 0,
                "page_records": {},
                "negative_tags": {},
                "negative_items": {},
                "positive_tags": {},
            }
        return self.credit_memory[avatar_id]

    def _register_page_credit_prediction(
        self,
        *,
        avatar_id: int,
        page_index: int,
        selected_item_id: int,
        adv_pred: float,
        pos_pred: float,
        neg_pred: float,
    ) -> None:
        memory = self._ensure_credit_memory(int(avatar_id))
        item_info = self.item_catalog.get(int(selected_item_id), {})
        memory["page_records"][int(page_index)] = {
            "item_id": int(selected_item_id),
            "tags": list(item_info.get("tags", set())),
            "adv_pred": float(adv_pred),
            "pos_pred": float(pos_pred),
            "neg_pred": float(neg_pred),
        }

    def _decay_memory_scores(self, scores: Dict[Any, float], decay: float) -> Dict[Any, float]:
        out: Dict[Any, float] = {}
        for key, value in scores.items():
            decayed = float(value) * float(decay)
            if decayed >= 1e-4:
                out[key] = decayed
        return out

    def _sync_credit_feedback(self, arena: Any, avatar_id: int) -> None:
        memory = self._ensure_credit_memory(int(avatar_id))
        ratings = list(arena.ratings.get(int(avatar_id), []))
        aligns = list(arena.n_likes.get(int(avatar_id), []))
        avatar = arena.avatars.get(int(avatar_id))
        current_negative_count = int(getattr(avatar, "negative_feedback_count", memory.get("last_negative_count", 0)))
        while int(memory.get("processed_pages", 0)) < len(ratings):
            page_no = int(memory.get("processed_pages", 0)) + 1
            record = memory.get("page_records", {}).get(page_no)
            avg_rating = parse_float(ratings[page_no - 1], 0.0)
            align_count = parse_int(aligns[page_no - 1] if page_no - 1 < len(aligns) else 0, 0)
            watch_count = 1 if avg_rating > 0 else 0
            negative_increment = heuristic_negative_increment(watch_count, avg_rating)
            neg_delta = max(int(current_negative_count) - int(memory.get("last_negative_count", 0)), 0)
            if record is not None:
                base_neg = max(float(record.get("neg_pred", 0.0)), 0.0)
                base_pos = max(float(record.get("pos_pred", 0.0)), 0.0)
                neg_strength = 0.55 * float(negative_increment) + 0.45 * base_neg
                if avg_rating > 0 and avg_rating < 4.0:
                    neg_strength += 0.20
                if neg_delta > 0:
                    neg_strength += 0.25 * float(neg_delta)
                pos_strength = 0.35 * base_pos
                if watch_count > 0:
                    pos_strength += 0.20
                if align_count > 0:
                    pos_strength += 0.20
                if avg_rating >= 4.0:
                    pos_strength += 0.20 * min(avg_rating / 5.0, 1.0)

                memory["negative_tags"] = self._decay_memory_scores(memory.get("negative_tags", {}), ATTRV2_MEMORY_DECAY)
                memory["negative_items"] = self._decay_memory_scores(memory.get("negative_items", {}), ATTRV2_MEMORY_DECAY)
                memory["positive_tags"] = self._decay_memory_scores(memory.get("positive_tags", {}), 0.90)

                if neg_strength >= max(pos_strength, 0.20):
                    item_id = int(record.get("item_id", -1))
                    if item_id >= 0:
                        memory["negative_items"][item_id] = float(memory["negative_items"].get(item_id, 0.0) + neg_strength)
                    tags = list(record.get("tags", []))
                    tag_denom = max(len(tags), 1)
                    for tag in tags:
                        memory["negative_tags"][tag] = float(
                            memory["negative_tags"].get(tag, 0.0) + neg_strength / float(tag_denom)
                        )
                elif pos_strength > 0.25:
                    for tag in list(record.get("tags", [])):
                        memory["positive_tags"][tag] = float(memory["positive_tags"].get(tag, 0.0) + pos_strength)

            memory["processed_pages"] = page_no
            memory["last_negative_count"] = int(current_negative_count)

    def _negative_memory_penalty(self, avatar_id: int, item_id: int, item_info: Dict[str, Any]) -> float:
        memory = self._ensure_credit_memory(int(avatar_id))
        tags = set(item_info.get("tags", set()))
        tag_penalty = sum(float(memory.get("negative_tags", {}).get(tag, 0.0)) for tag in tags)
        pos_relief = sum(float(memory.get("positive_tags", {}).get(tag, 0.0)) for tag in tags)
        item_penalty = float(memory.get("negative_items", {}).get(int(item_id), 0.0))
        adjusted = max(tag_penalty - 0.35 * pos_relief, 0.0)
        return float(adjusted + ATTRV2_ITEM_REPEAT_SCALE * item_penalty)

    def _register_selected_credit_from_result(self, avatar_id: int, page_index: int, result: Dict[str, Any]) -> None:
        if self.score_profile not in ATTR_SESSION_CREDIT_PROFILES:
            return
        selected = list(result.get("selected", []))
        kept_ids = list(result.get("kept_ids", []))
        if not selected or not kept_ids:
            return
        item_id = int(selected[0])
        try:
            idx = kept_ids.index(item_id)
        except ValueError:
            idx = 0
        adv_list = list(result.get("credit_adv_preds", []))
        pos_list = list(result.get("credit_pos_preds", []))
        neg_list = list(result.get("credit_neg_preds", []))
        self._register_page_credit_prediction(
            avatar_id=int(avatar_id),
            page_index=int(page_index),
            selected_item_id=item_id,
            adv_pred=float(adv_list[idx]) if idx < len(adv_list) else 0.0,
            pos_pred=float(pos_list[idx]) if idx < len(pos_list) else 0.0,
            neg_pred=float(neg_list[idx]) if idx < len(neg_list) else 0.0,
        )

    def _get_runtime_history_items(self, arena: Any, avatar_id: int) -> List[int]:
        if arena is None:
            return []
        source = getattr(arena, "new_train_dict", None)
        if isinstance(source, dict) and int(avatar_id) in source:
            history = source.get(int(avatar_id), [])
        else:
            data_obj = getattr(arena, "data", None)
            train_user_list = getattr(data_obj, "train_user_list", {}) if data_obj is not None else {}
            history = train_user_list.get(int(avatar_id), [])
        out: List[int] = []
        for value in history:
            iid = parse_int(value, -1)
            if iid >= 0:
                out.append(int(iid))
        return out

    def _get_tiger_candidate_attribution(self, arena: Any, avatar_id: int, item_id: int) -> Dict[str, Any]:
        if arena is None:
            return {}
        model = getattr(arena, "model", None)
        if model is None or not hasattr(model, "get_candidate_attribution"):
            return {}
        history_items = self._get_runtime_history_items(arena, int(avatar_id))
        try:
            return model.get_candidate_attribution(int(item_id), history_items=history_items, user_id=int(avatar_id))
        except Exception:
            return {}

    def predict_plan(self, user_profile: Dict[str, Any], state: Dict[str, float]) -> str:
        if self.planner_model is None:
            return choose_plan(user_profile, state, override=self.override_plan)
        if self.override_plan and self.override_plan != "auto":
            return self.override_plan
        dist = self.predict_plan_distribution(user_profile, state)
        if not dist:
            return choose_plan(user_profile, state, override="auto")
        return max(dist.items(), key=lambda kv: kv[1])[0]

    def predict_plan_distribution(self, user_profile: Dict[str, Any], state: Dict[str, float]) -> Dict[str, float]:
        if self.override_plan and self.override_plan != "auto":
            return {plan: 1.0 if plan == self.override_plan else 0.0 for plan in PLAN_OPTIONS}
        if self.planner_model is None:
            chosen = choose_plan(user_profile, state, override="auto")
            return {plan: 1.0 if plan == chosen else 0.0 for plan in PLAN_OPTIONS}
        row = build_planner_feature_row(user_profile, state)
        vec = planner_row_to_vector(row).reshape(1, -1)
        if self.planner_model_kind == "regressor_scores":
            raw = np.asarray(self.planner_model.predict(vec), dtype=np.float32).reshape(-1)
            if raw.shape[0] != len(PLAN_OPTIONS):
                chosen = choose_plan(user_profile, state, override="auto")
                return {plan: 1.0 if plan == chosen else 0.0 for plan in PLAN_OPTIONS}
            probs = softmax(raw.tolist(), temperature=float(self.metadata.get("planner_score_temperature", PLAN_SCORE_TEMPERATURE)))
            return {plan: float(prob) for plan, prob in zip(PLAN_OPTIONS, probs)}
        if not hasattr(self.planner_model, "predict_proba"):
            pred = self.planner_model.predict(vec)
            chosen = str(pred[0]) if len(pred) else choose_plan(user_profile, state, override="auto")
            return {plan: 1.0 if plan == chosen else 0.0 for plan in PLAN_OPTIONS}
        probs = self.planner_model.predict_proba(vec)[0]
        dist = {plan: 0.0 for plan in PLAN_OPTIONS}
        classes = list(getattr(self.planner_model, "classes_", []))
        for cls, prob in zip(classes, probs):
            name = str(cls)
            if name in dist:
                dist[name] = float(prob)
        total = sum(dist.values())
        if total <= 0:
            chosen = self.predict_plan(user_profile, state)
            return {plan: 1.0 if plan == chosen else 0.0 for plan in PLAN_OPTIONS}
        return {plan: float(val / total) for plan, val in dist.items()}

    def predict_plan_dwell_scores(self, user_profile: Dict[str, Any], state: Dict[str, float]) -> Dict[str, float]:
        if self.override_plan and self.override_plan != "auto":
            return {plan: 1.0 if plan == self.override_plan else 0.0 for plan in PLAN_OPTIONS}
        if self.planner_dwell_model is None:
            return self.predict_plan_distribution(user_profile, state)
        row = build_planner_feature_row(user_profile, state)
        vec = planner_row_to_vector(row).reshape(1, -1)
        raw = np.asarray(self.planner_dwell_model.predict(vec), dtype=np.float32).reshape(-1)
        if raw.shape[0] != len(PLAN_OPTIONS):
            return self.predict_plan_distribution(user_profile, state)
        raw = np.maximum(raw, 0.0)
        total = float(np.sum(raw))
        if total <= 1e-6:
            return self.predict_plan_distribution(user_profile, state)
        return {plan: float(val / total) for plan, val in zip(PLAN_OPTIONS, raw.tolist())}

    def _option_risk_score(self, state: Dict[str, float]) -> float:
        neg = min(max(float(state.get("proxy_negative_count", 0.0)), 0.0), 3.0) / 3.0
        recent_watch = min(max(float(state.get("recent_watch_rate_3", 0.0)), 0.0), 1.0)
        pages_since_watch = min(max(float(state.get("pages_since_last_watch", 0.0)), 0.0), 3.0) / 3.0
        progress = min(max(float(state.get("page_progress", 0.0)), 0.0), 1.0)
        recent_align = min(max(float(state.get("recent_align_rate_3", 0.0)), 0.0), 1.0)
        risk = (
            0.40 * neg
            + 0.25 * max(0.45 - recent_watch, 0.0) / 0.45
            + 0.18 * pages_since_watch
            + 0.09 * progress
            + 0.08 * max(0.35 - recent_align, 0.0) / 0.35
        )
        return float(min(max(risk, 0.0), 1.0))

    def _plan_utility(self, row: Dict[str, float], base_rank_score: float, plan: str) -> float:
        quality = float(row["item_quality"])
        match = 0.75 * float(row["focus_match"]) + 0.25 * min(float(row["pref_match_count"]) / 2.0, 1.0)
        trust = 0.55 * float(row["item_high_reviews"]) + 0.45 * float(row["item_high_rating"])
        novelty = max(float(row["pref_match_any"]) - float(row["focus_match"]), 0.0)
        low_risk = 1.0 - max(float(row["item_low_rating"]), float(row["item_low_reviews"]))
        if self.score_profile in EXITFIRST_STYLE_PROFILES:
            # Exit-first profile: bias toward safe continuation and preserve more of
            # the base rank, especially when the planner is not strongly confident.
            if plan == "recover":
                return 0.35 * match + 0.23 * quality + 0.18 * trust + 0.12 * low_risk + 0.12 * base_rank_score
            if plan == "explore":
                return 0.20 * match + 0.18 * quality + 0.12 * trust + 0.10 * novelty + 0.40 * base_rank_score
            if plan == "safe_match":
                return 0.42 * match + 0.23 * quality + 0.16 * trust + 0.10 * low_risk + 0.09 * base_rank_score
            return 0.30 * match + 0.22 * quality + 0.14 * trust + 0.06 * novelty + 0.28 * base_rank_score
        if self.score_profile == "modeA_tiger":
            # Strong base rankers like TIGER in relaxed modeA should keep more of the
            # original ordering signal; reranking should only make local corrections.
            if plan == "recover":
                return 0.33 * match + 0.22 * quality + 0.17 * trust + 0.08 * low_risk + 0.20 * base_rank_score
            if plan == "explore":
                return 0.22 * match + 0.20 * quality + 0.12 * trust + 0.14 * novelty + 0.32 * base_rank_score
            if plan == "safe_match":
                return 0.38 * match + 0.22 * quality + 0.16 * trust + 0.08 * low_risk + 0.16 * base_rank_score
            return 0.28 * match + 0.22 * quality + 0.12 * trust + 0.08 * novelty + 0.30 * base_rank_score
        if plan == "recover":
            return 0.42 * match + 0.24 * quality + 0.20 * trust + 0.10 * low_risk + 0.04 * base_rank_score
        if plan == "explore":
            return 0.28 * match + 0.24 * quality + 0.18 * trust + 0.18 * novelty + 0.12 * base_rank_score
        if plan == "safe_match":
            return 0.46 * match + 0.24 * quality + 0.18 * trust + 0.08 * low_risk + 0.04 * base_rank_score
        return 0.36 * match + 0.26 * quality + 0.16 * trust + 0.08 * novelty + 0.14 * base_rank_score

    def _component_weights(self, *, plan: str, state: Dict[str, float]) -> Tuple[float, float, float, float]:
        progress = float(state.get("page_progress", 0.0))
        fatigue_scale = 1.0
        if plan == "recover":
            survival_w, enjoy_w, utility_w = 0.44, 0.38, 0.18
        elif plan == "safe_match":
            survival_w, enjoy_w, utility_w = 0.31, 0.45, 0.24
        elif plan == "explore":
            survival_w, enjoy_w, utility_w = 0.24, 0.34, 0.42
        else:
            survival_w, enjoy_w, utility_w = 0.31, 0.42, 0.27

        survival_w = max(survival_w - 0.12 * progress, 0.14)
        enjoy_w = min(enjoy_w + 0.14 * progress, 0.62)
        utility_w = max(1.0 - survival_w - enjoy_w, 0.12)
        if self.score_profile in EXITFIRST_STYLE_PROFILES:
            if plan == "recover":
                survival_w, enjoy_w, utility_w = 0.52, 0.30, 0.18
            elif plan == "safe_match":
                survival_w, enjoy_w, utility_w = 0.43, 0.34, 0.23
            elif plan == "explore":
                survival_w, enjoy_w, utility_w = 0.32, 0.24, 0.44
            else:
                survival_w, enjoy_w, utility_w = 0.40, 0.32, 0.28
            survival_w = max(survival_w - 0.04 * progress, 0.24)
            enjoy_w = min(enjoy_w + 0.06 * progress, 0.54)
            utility_w = max(1.0 - survival_w - enjoy_w, 0.16)
            fatigue_scale = 1.20
        elif self.score_profile == "modeA_tiger":
            survival_w = max(survival_w - 0.08, 0.14)
            enjoy_w = min(enjoy_w + 0.04, 0.66)
            utility_w = max(1.0 - survival_w - enjoy_w, 0.20)
            fatigue_scale = 0.55
        return float(survival_w), float(enjoy_w), float(utility_w), float(fatigue_scale)

    def _build_score_breakdown(
        self,
        *,
        row: Dict[str, float],
        exit_prob: float,
        enjoy_prob: float,
        utility: float,
        plan: str,
        state: Dict[str, float],
    ) -> Dict[str, float]:
        progress = float(state.get("page_progress", 0.0))
        neg = float(state.get("proxy_negative_count", 0.0))
        survival_w, enjoy_w, utility_w, fatigue_scale = self._component_weights(plan=plan, state=state)
        low_risk = 1.0 - max(float(row["item_low_rating"]), float(row["item_low_reviews"]))
        survival_term = 1.0 - float(exit_prob)
        satisfaction_term = 0.72 * float(enjoy_prob) + 0.18 * float(row["item_quality"]) + 0.10 * low_risk
        fatigue_penalty = (
            progress * max(0.0, 0.35 - float(enjoy_prob)) * 0.42
            + min(neg, 3.0) * max(0.0, 0.45 - float(enjoy_prob)) * 0.05
        ) * fatigue_scale
        survival_contrib = survival_w * survival_term
        satisfaction_contrib = enjoy_w * satisfaction_term
        utility_contrib = utility_w * float(utility)
        return {
            "survival_term": float(survival_term),
            "satisfaction_term": float(satisfaction_term),
            "utility_term": float(utility),
            "survival_contrib": float(survival_contrib),
            "satisfaction_contrib": float(satisfaction_contrib),
            "utility_contrib": float(utility_contrib),
            "fatigue_penalty": float(fatigue_penalty),
            "base_score": float(survival_contrib + satisfaction_contrib + utility_contrib - fatigue_penalty),
        }

    def _click_proxy_score(
        self,
        *,
        row: Dict[str, float],
        detail: Dict[str, float],
        plan: str,
        state: Dict[str, float],
    ) -> float:
        progress = float(state.get("page_progress", 0.0))
        neg = float(state.get("proxy_negative_count", 0.0))
        match = 0.72 * float(row["focus_match"]) + 0.28 * min(float(row["pref_match_count"]) / 2.0, 1.0)
        trust = 0.55 * float(row["item_high_reviews"]) + 0.45 * float(row["item_high_rating"])
        novelty = max(float(row["pref_match_any"]) - float(row["focus_match"]), 0.0)
        low_risk = 1.0 - max(float(row["item_low_rating"]), float(row["item_low_reviews"]))
        recent_watch = float(state.get("recent_watch_rate_3", 0.0))
        recent_align = float(state.get("recent_align_rate_3", 0.0))
        recent_rating = min(max(float(state.get("recent_avg_rating_3", 0.0)) / 5.0, 0.0), 1.0)

        score = (
            0.28 * match
            + 0.20 * float(row["item_quality"])
            + 0.14 * low_risk
            + 0.12 * trust
            + 0.10 * float(detail["satisfaction_term"])
            + 0.08 * recent_watch
            + 0.04 * recent_align
            + 0.04 * recent_rating
        )
        if plan == "recover":
            score += 0.06 * low_risk + 0.04 * trust - 0.02 * novelty
        elif plan == "safe_match":
            score += 0.08 * match + 0.03 * trust
        elif plan == "explore":
            score += 0.09 * novelty + 0.02 * float(row["item_quality"])
        else:
            score += 0.04 * match + 0.04 * novelty

        if progress >= 0.55:
            score += 0.03 * float(row["item_quality"])
        if neg >= 1.0:
            score += 0.04 * low_risk - 0.02 * novelty
        return float(score)

    def _multiobjective_bonus(
        self,
        *,
        rows: Sequence[Dict[str, float]],
        score_details: Sequence[Dict[str, float]],
        state: Dict[str, float],
        plan: str,
        final_scores: Sequence[float],
    ) -> List[float]:
        if not rows or not score_details:
            return []

        progress = float(state.get("page_progress", 0.0))
        neg = float(state.get("proxy_negative_count", 0.0))
        if progress <= 0.35:
            exit_w, click_w, sat_w = 0.18, 0.46, 0.36
        elif progress <= 0.70:
            exit_w = 0.24 + 0.04 * min(neg, 2.0) / 2.0
            click_w = 0.40 - 0.03 * min(neg, 2.0) / 2.0
            sat_w = 1.0 - exit_w - click_w
        else:
            exit_w = 0.30 + 0.04 * min(neg, 2.0) / 2.0
            click_w = 0.28
            sat_w = 1.0 - exit_w - click_w
        if plan == "recover":
            exit_w += 0.04
            click_w -= 0.02
        elif plan == "safe_match":
            sat_w += 0.04
        elif plan == "explore":
            click_w += 0.04
            sat_w -= 0.02
        total_w = max(exit_w + click_w + sat_w, 1e-6)
        exit_w, click_w, sat_w = exit_w / total_w, click_w / total_w, sat_w / total_w

        base_norm = minmax_scale(final_scores)
        survival_norm = minmax_scale([detail["survival_term"] for detail in score_details])
        click_scores = [
            self._click_proxy_score(row=row, detail=detail, plan=plan, state=state)
            for row, detail in zip(rows, score_details)
        ]
        click_norm = minmax_scale(click_scores)
        satisfaction_norm = minmax_scale([detail["satisfaction_term"] for detail in score_details])

        bonuses: List[float] = []
        for idx, detail in enumerate(score_details):
            mo_objective = (
                0.35 * base_norm[idx]
                + 0.65 * (
                    exit_w * survival_norm[idx]
                    + click_w * click_norm[idx]
                    + sat_w * satisfaction_norm[idx]
                )
            )
            pareto = 0.4 * min(survival_norm[idx], click_norm[idx], satisfaction_norm[idx]) + 0.6 * min(
                click_norm[idx], satisfaction_norm[idx]
            )
            spread = float(np.std(np.asarray(
                [survival_norm[idx], click_norm[idx], satisfaction_norm[idx]],
                dtype=np.float32,
            )))
            survival_dominance = max(
                survival_norm[idx] - 0.5 * (click_norm[idx] + satisfaction_norm[idx]),
                0.0,
            )
            bonus = (
                (MULTIOBJ_BLEND_SCALE + 0.02) * (mo_objective - 0.5)
                + MULTIOBJ_PARETO_SCALE * pareto
                - MULTIOBJ_STD_PENALTY * spread
                - 0.07 * survival_dominance
            )
            detail["click_proxy"] = float(click_scores[idx])
            detail["mo_objective"] = float(mo_objective)
            detail["mo_pareto"] = float(pareto)
            detail["mo_spread"] = float(spread)
            detail["mo_survival_dominance"] = float(survival_dominance)
            detail["mo_bonus"] = float(bonus)
            bonuses.append(float(bonus))
        return bonuses

    def _score_with_survival_and_enjoyment(
        self,
        *,
        row: Dict[str, float],
        exit_prob: float,
        enjoy_prob: float,
        utility: float,
        plan: str,
        state: Dict[str, float],
    ) -> float:
        return float(
            self._build_score_breakdown(
                row=row,
                exit_prob=exit_prob,
                enjoy_prob=enjoy_prob,
                utility=utility,
                plan=plan,
                state=state,
            )["base_score"]
        )

    def _score_candidates_for_plan(
        self,
        *,
        arena: Optional[Any],
        user_profile: Dict[str, Any],
        state: Dict[str, float],
        candidate_ids: Sequence[int],
        plan: str,
        items_per_page: int,
        avatar_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, float]] = []
        kept_ids: List[int] = []
        utilities: List[float] = []
        for rank_idx, item_id in enumerate(candidate_ids):
            item_info = self.item_catalog.get(int(item_id))
            if item_info is None:
                continue
            row = build_feature_row(user_profile, state, item_info, plan)
            rows.append(row)
            kept_ids.append(int(item_id))
            base_rank_score = 1.0 - (float(rank_idx) / max(float(len(candidate_ids) - 1), 1.0))
            utilities.append(self._plan_utility(row, base_rank_score=base_rank_score, plan=plan))

        if not rows:
            return {
                "plan": plan,
                "kept_ids": [],
                "rows": [],
                "exit_probs": np.asarray([], dtype=np.float32),
                "enjoy_probs": np.asarray([], dtype=np.float32),
                "final_scores": [],
                "selected": list(candidate_ids)[:items_per_page],
                "plan_value": -1e9,
                "top_candidates": [],
            }

        batch = np.vstack([row_to_vector(row) for row in rows]).astype(np.float32)
        exit_probs = self.predict_exit_prob(batch)
        enjoy_probs = self.predict_enjoy_prob(batch)
        credit_adv_preds, credit_pos_preds, credit_neg_preds = self.predict_credit_outputs(batch)
        final_scores: List[float] = []
        score_details: List[Dict[str, float]] = []
        for idx, row in enumerate(rows):
            detail = self._build_score_breakdown(
                row=row,
                exit_prob=float(exit_probs[idx]),
                enjoy_prob=float(enjoy_probs[idx]),
                utility=float(utilities[idx]),
                plan=plan,
                state=state,
            )
            score_details.append(detail)
            final_scores.append(float(detail["base_score"]))

        if self.score_profile in ATTR_SESSION_CREDIT_PROFILES and score_details:
            memory_penalties: List[float] = []
            tiger_attr_bundles: List[Dict[str, Any]] = []
            for idx, iid in enumerate(kept_ids):
                item_info = self.item_catalog.get(int(iid), {})
                memory_penalty = (
                    self._negative_memory_penalty(int(avatar_id), int(iid), item_info)
                    if avatar_id is not None
                    else 0.0
                )
                credit_bonus = (
                    ATTRV2_ADV_SCALE * float(credit_adv_preds[idx])
                    + ATTRV2_POS_SCALE * float(credit_pos_preds[idx])
                    - ATTRV2_NEG_SCALE * float(credit_neg_preds[idx])
                    - ATTRV2_NEG_MEMORY_SCALE * float(memory_penalty)
                )
                score_details[idx]["credit_adv"] = float(credit_adv_preds[idx])
                score_details[idx]["credit_pos"] = float(credit_pos_preds[idx])
                score_details[idx]["credit_neg"] = float(credit_neg_preds[idx])
                score_details[idx]["neg_memory_penalty"] = float(memory_penalty)
                score_details[idx]["credit_bonus"] = float(credit_bonus)
                memory_penalties.append(float(memory_penalty))
                final_scores[idx] = float(final_scores[idx] + credit_bonus)
                if self.score_profile in ATTRV3_PROFILES and avatar_id is not None:
                    tiger_bundle = self._get_tiger_candidate_attribution(arena, int(avatar_id), int(iid))
                    block_weights = np.asarray(tiger_bundle.get("block_weights", []), dtype=np.float32)
                    block_support = np.clip(np.asarray(tiger_bundle.get("block_support", []), dtype=np.float32), 0.0, 1.0)
                    pred_block_credit = np.asarray(tiger_bundle.get("pred_block_credit", []), dtype=np.float32)
                    pred_prefix_credit = np.asarray(tiger_bundle.get("pred_prefix_credit", []), dtype=np.float32)
                    pred_page_credit = float(tiger_bundle.get("pred_page_credit", 0.0))
                    transport_available = bool(tiger_bundle.get("transport_available", False))
                    history_weights = np.asarray(tiger_bundle.get("history_weights", []), dtype=np.float32)
                    support = float(tiger_bundle.get("support", 0.0))
                    residual_support = float(tiger_bundle.get("residual_support", 0.0))
                    counterfactual_gap = float(tiger_bundle.get("counterfactual_gap", 0.0))
                    decode_support = float(tiger_bundle.get("decode_bonus", 0.0))
                    positive_credit = max(float(credit_bonus), 0.0)
                    negative_credit = max(-float(credit_bonus), 0.0)
                    block_alignment = float(np.dot(block_weights, block_support)) if block_weights.size and block_support.size else 0.0
                    history_concentration = float(np.max(history_weights)) if history_weights.size else 0.0
                    ot_bonus = (
                        ATTRV3_OT_SCALE * positive_credit * block_alignment
                        - 0.5 * ATTRV3_OT_SCALE * negative_credit * max(1.0 - block_alignment, 0.0)
                    )
                    history_bonus = ATTRV3_HISTORY_SCALE * (0.70 * support + 0.30 * residual_support)
                    cf_bonus = ATTRV3_CF_SCALE * (residual_support - 0.75 * counterfactual_gap)
                    decode_bonus = ATTRV3_DECODE_SCALE * decode_support
                    neg_history_penalty = ATTRV3_NEG_HISTORY_SCALE * negative_credit * max(
                        history_concentration - residual_support,
                        0.0,
                    )
                    attrv3_bonus = float(history_bonus + ot_bonus + cf_bonus + decode_bonus - neg_history_penalty)
                    block_credit = (
                        [float(v) for v in pred_block_credit.tolist()]
                        if pred_block_credit.size
                        else (float(credit_bonus) * block_weights).tolist() if block_weights.size else []
                    )
                    score_details[idx]["tiger_support"] = float(support)
                    score_details[idx]["tiger_residual_support"] = float(residual_support)
                    score_details[idx]["tiger_cf_gap"] = float(counterfactual_gap)
                    score_details[idx]["tiger_decode_bonus"] = float(decode_support)
                    score_details[idx]["tiger_block_alignment"] = float(block_alignment)
                    score_details[idx]["tiger_transport_available"] = 1.0 if transport_available else 0.0
                    score_details[idx]["tiger_transport_page_credit"] = float(pred_page_credit)
                    score_details[idx]["tiger_transport_prefix_credit"] = [float(v) for v in pred_prefix_credit.tolist()] if pred_prefix_credit.size else []
                    score_details[idx]["tiger_transport_centered"] = 0.0
                    score_details[idx]["tiger_transport_z"] = 0.0
                    score_details[idx]["tiger_transport_residual_bonus"] = 0.0
                    score_details[idx]["tiger_transport_gate_open"] = 0.0
                    score_details[idx]["tiger_transport_gate_plan_ok"] = 0.0
                    score_details[idx]["tiger_transport_gate_credit_ok"] = 0.0
                    score_details[idx]["tiger_transport_gate_risk_ok"] = 0.0
                    score_details[idx]["attrv3_bonus"] = float(attrv3_bonus)
                    score_details[idx]["tiger_history_concentration"] = float(history_concentration)
                    score_details[idx]["tiger_top_history"] = list(tiger_bundle.get("top_history", []))
                    score_details[idx]["tiger_block_credit"] = [float(v) for v in block_credit]
                    score_details[idx]["tiger_block_weights"] = [float(v) for v in block_weights.tolist()] if block_weights.size else []
                    final_scores[idx] = float(final_scores[idx] + attrv3_bonus)
                    tiger_attr_bundles.append(tiger_bundle)
                else:
                    tiger_attr_bundles.append({})
            if self.score_profile in ATTRV3_PROFILES:
                transport_indices: List[int] = []
                transport_values: List[float] = []
                for idx, detail in enumerate(score_details):
                    if float(detail.get("tiger_transport_available", 0.0)) <= 0.0:
                        continue
                    transport_indices.append(int(idx))
                    transport_values.append(float(detail.get("tiger_transport_page_credit", 0.0)))
                if len(transport_indices) >= 2:
                    transport_arr = np.asarray(transport_values, dtype=np.float32)
                    transport_mean = float(np.mean(transport_arr))
                    transport_std = max(float(np.std(transport_arr)), ATTRV3_TRANSPORT_STD_FLOOR)
                    transport_z = np.clip(
                        (transport_arr - transport_mean) / transport_std,
                        -ATTRV3_TRANSPORT_Z_CLIP,
                        ATTRV3_TRANSPORT_Z_CLIP,
                    )
                    for pos, score_idx in enumerate(transport_indices):
                        centered = float(transport_arr[pos] - transport_mean)
                        z_value = float(transport_z[pos])
                        raw_page_credit = float(score_details[score_idx].get("tiger_transport_page_credit", 0.0))
                        exit_prob_value = float(exit_probs[score_idx])
                        plan_ok = plan in ATTRV3_TRANSPORT_ALLOWED_PLANS
                        credit_ok = raw_page_credit >= ATTRV3_TRANSPORT_MIN_PAGE_CREDIT
                        risk_ok = exit_prob_value <= ATTRV3_TRANSPORT_MAX_EXIT_PROB
                        gate_open = bool(plan_ok and credit_ok and risk_ok)
                        residual_bonus = (
                            float(ATTRV3_TRANSPORT_RESIDUAL_SCALE * max(z_value, 0.0))
                            if gate_open
                            else 0.0
                        )
                        score_details[score_idx]["tiger_transport_centered"] = centered
                        score_details[score_idx]["tiger_transport_z"] = z_value
                        score_details[score_idx]["tiger_transport_gate_open"] = 1.0 if gate_open else 0.0
                        score_details[score_idx]["tiger_transport_gate_plan_ok"] = 1.0 if plan_ok else 0.0
                        score_details[score_idx]["tiger_transport_gate_credit_ok"] = 1.0 if credit_ok else 0.0
                        score_details[score_idx]["tiger_transport_gate_risk_ok"] = 1.0 if risk_ok else 0.0
                        score_details[score_idx]["tiger_transport_residual_bonus"] = residual_bonus
                        final_scores[score_idx] = float(final_scores[score_idx] + residual_bonus)
        else:
            memory_penalties = [0.0 for _ in kept_ids]
            tiger_attr_bundles = [{} for _ in kept_ids]

        if self.score_profile in ATTR_PROFILES and score_details:
            progress = float(state.get("page_progress", 0.0))
            survival_mean = mean_or_zero([detail["survival_contrib"] for detail in score_details])
            satisfaction_mean = mean_or_zero([detail["satisfaction_contrib"] for detail in score_details])
            utility_mean = mean_or_zero([detail["utility_contrib"] for detail in score_details])
            for idx, detail in enumerate(score_details):
                contribs = [
                    float(detail["survival_contrib"]),
                    float(detail["satisfaction_contrib"]),
                    float(detail["utility_contrib"]),
                ]
                mean_contrib = max(mean_or_zero(contribs), 1e-6)
                balance = min(contribs) / mean_contrib
                robust_hmean = harmonic_mean_positive(contribs)
                min_margin = min(
                    float(detail["survival_contrib"]) - survival_mean,
                    float(detail["satisfaction_contrib"]) - satisfaction_mean,
                    float(detail["utility_contrib"]) - utility_mean,
                )
                survival_sat_gap = max(
                    float(detail["survival_contrib"]) - float(detail["satisfaction_contrib"]),
                    0.0,
                )
                attr_bonus = (
                    (ATTR_BASE_SCALE + 0.04 * progress) * balance * robust_hmean
                    + ATTR_MARGIN_SCALE * max(min_margin, 0.0)
                    - ATTR_SURVIVAL_SAT_PENALTY * survival_sat_gap * (0.5 + 0.5 * progress)
                )
                detail["attr_balance"] = float(balance)
                detail["attr_hmean"] = float(robust_hmean)
                detail["attr_min_margin"] = float(min_margin)
                detail["attr_bonus"] = float(attr_bonus)
                final_scores[idx] = float(final_scores[idx] + attr_bonus)

        if self.score_profile in MULTIOBJ_PROFILES and score_details:
            mo_bonuses = self._multiobjective_bonus(
                rows=rows,
                score_details=score_details,
                state=state,
                plan=plan,
                final_scores=final_scores,
            )
            for idx, bonus in enumerate(mo_bonuses):
                final_scores[idx] = float(final_scores[idx] + bonus)

        order = np.argsort(-np.asarray(final_scores, dtype=np.float32))
        selected = [kept_ids[int(i)] for i in order[: max(int(items_per_page), 1)]]
        top_scores = [float(final_scores[int(i)]) for i in order[: max(int(items_per_page), 1)]]
        plan_value = mean_or_zero(top_scores)
        debug_rows = []
        for i in order[: min(len(order), 5)]:
            iid = kept_ids[int(i)]
            info = self.item_catalog.get(iid, {})
            debug_rows.append(
                {
                    "item_id": iid,
                    "title": info.get("title", ""),
                    "score": round(float(final_scores[int(i)]), 4),
                    "exit_prob": round(float(exit_probs[int(i)]), 4),
                    "enjoy_prob": round(float(enjoy_probs[int(i)]), 4),
                    "quality": round(float(info.get("quality", 0.0)), 4),
                    "focus_match": int(rows[int(i)]["focus_match"]),
                    "survival_contrib": round(float(score_details[int(i)]["survival_contrib"]), 4),
                    "satisfaction_contrib": round(float(score_details[int(i)]["satisfaction_contrib"]), 4),
                    "utility_contrib": round(float(score_details[int(i)]["utility_contrib"]), 4),
                    "attr_bonus": round(float(score_details[int(i)].get("attr_bonus", 0.0)), 4),
                    "credit_bonus": round(float(score_details[int(i)].get("credit_bonus", 0.0)), 4),
                    "credit_adv": round(float(score_details[int(i)].get("credit_adv", 0.0)), 4),
                    "credit_pos": round(float(score_details[int(i)].get("credit_pos", 0.0)), 4),
                    "credit_neg": round(float(score_details[int(i)].get("credit_neg", 0.0)), 4),
                    "neg_memory_penalty": round(float(score_details[int(i)].get("neg_memory_penalty", 0.0)), 4),
                    "attrv3_bonus": round(float(score_details[int(i)].get("attrv3_bonus", 0.0)), 4),
                    "tiger_support": round(float(score_details[int(i)].get("tiger_support", 0.0)), 4),
                    "tiger_residual_support": round(float(score_details[int(i)].get("tiger_residual_support", 0.0)), 4),
                    "tiger_cf_gap": round(float(score_details[int(i)].get("tiger_cf_gap", 0.0)), 4),
                    "tiger_decode_bonus": round(float(score_details[int(i)].get("tiger_decode_bonus", 0.0)), 4),
                    "tiger_block_alignment": round(float(score_details[int(i)].get("tiger_block_alignment", 0.0)), 4),
                    "tiger_transport_residual_bonus": round(float(score_details[int(i)].get("tiger_transport_residual_bonus", 0.0)), 4),
                    "tiger_transport_page_credit": round(float(score_details[int(i)].get("tiger_transport_page_credit", 0.0)), 4),
                    "tiger_transport_centered": round(float(score_details[int(i)].get("tiger_transport_centered", 0.0)), 4),
                    "tiger_transport_z": round(float(score_details[int(i)].get("tiger_transport_z", 0.0)), 4),
                    "tiger_transport_gate_open": round(float(score_details[int(i)].get("tiger_transport_gate_open", 0.0)), 4),
                    "tiger_transport_gate_plan_ok": round(float(score_details[int(i)].get("tiger_transport_gate_plan_ok", 0.0)), 4),
                    "tiger_transport_gate_credit_ok": round(float(score_details[int(i)].get("tiger_transport_gate_credit_ok", 0.0)), 4),
                    "tiger_transport_gate_risk_ok": round(float(score_details[int(i)].get("tiger_transport_gate_risk_ok", 0.0)), 4),
                    "tiger_transport_prefix_credit": [round(float(v), 4) for v in score_details[int(i)].get("tiger_transport_prefix_credit", [])],
                    "tiger_top_history": score_details[int(i)].get("tiger_top_history", []),
                    "tiger_block_credit": [round(float(v), 4) for v in score_details[int(i)].get("tiger_block_credit", [])],
                    "click_proxy": round(float(score_details[int(i)].get("click_proxy", 0.0)), 4),
                    "mo_bonus": round(float(score_details[int(i)].get("mo_bonus", 0.0)), 4),
                    "mo_objective": round(float(score_details[int(i)].get("mo_objective", 0.0)), 4),
                }
            )
        return {
            "plan": plan,
            "kept_ids": kept_ids,
            "rows": rows,
            "exit_probs": exit_probs,
            "enjoy_probs": enjoy_probs,
            "credit_adv_preds": [float(v) for v in credit_adv_preds.tolist()],
            "credit_pos_preds": [float(v) for v in credit_pos_preds.tolist()],
            "credit_neg_preds": [float(v) for v in credit_neg_preds.tolist()],
            "neg_memory_penalties": [float(v) for v in memory_penalties],
            "final_scores": final_scores,
            "selected": selected,
            "plan_value": float(plan_value),
            "top_candidates": debug_rows,
        }

    def score_candidates(
        self,
        *,
        arena: Any,
        avatar_id: int,
        candidate_ids: Sequence[int],
        page_index: int,
        items_per_page: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        if int(page_index) <= 1:
            self.credit_memory.pop(int(avatar_id), None)
        self._sync_credit_feedback(arena, int(avatar_id))
        user_profile = self.get_user_profile(avatar_id)
        state = self.build_runtime_state(arena, avatar_id=avatar_id, page_index=page_index)
        cf_profiles = {
            "modeA_scopecf",
            "modeA_tiger_scopecf",
            "modeA_scopecf_gate",
            "modeA_tiger_scopecf_gate",
        }
        gated_cf_profiles = {"modeA_scopecf_gate", "modeA_tiger_scopecf_gate"}

        if self.score_profile in OPTION_PROFILES:
            if int(page_index) <= 1:
                self.plan_memory.pop(int(avatar_id), None)
            plan_prior = self.predict_plan_distribution(user_profile, state)
            dwell_scores = self.predict_plan_dwell_scores(user_profile, state)
            risk_score = self._option_risk_score(state)
            previous_plan = str(self.plan_memory.get(int(avatar_id), {}).get("plan", ""))
            previous_dwell = float(dwell_scores.get(previous_plan, 0.0)) if previous_plan else 0.0
            plan_results: Dict[str, Dict[str, Any]] = {}
            plan_scores: Dict[str, float] = {}

            for plan_name in PLAN_OPTIONS:
                result = self._score_candidates_for_plan(
                    arena=arena,
                    user_profile=user_profile,
                    state=state,
                    candidate_ids=candidate_ids,
                    plan=plan_name,
                    items_per_page=items_per_page,
                    avatar_id=int(avatar_id),
                )
                prior = max(float(plan_prior.get(plan_name, 0.0)), 1e-6)
                dwell = float(dwell_scores.get(plan_name, 0.0))
                combined = float(result["plan_value"]) + OPTION_PRIOR_LOG_SCALE * math.log(prior) + OPTION_DWELL_SCALE * dwell

                if plan_name == "recover":
                    combined += 0.035 * risk_score + 0.03 * risk_score * dwell
                elif plan_name == "safe_match":
                    combined += 0.02 * risk_score + 0.02 * dwell
                elif plan_name == "explore":
                    combined += 0.018 * max(1.0 - risk_score, 0.0) * dwell
                else:
                    combined += 0.015 * dwell

                if previous_plan:
                    if plan_name == previous_plan:
                        combined += OPTION_STAY_SCALE * max(1.0 - 0.55 * risk_score, 0.0) * (0.6 + 0.4 * dwell)
                    else:
                        switch_cost = (OPTION_SWITCH_BASE + 0.05 * previous_dwell) * max(1.0 - risk_score, 0.0)
                        if plan_name in {"recover", "safe_match"} and risk_score >= 0.55:
                            switch_cost *= 0.35
                        combined -= switch_cost

                plan_results[plan_name] = result
                plan_scores[plan_name] = combined

            if not plan_results:
                fallback_plan = self.predict_plan(user_profile, state)
                return list(candidate_ids)[:items_per_page], {"plan": fallback_plan, "used_model": self.model is not None}

            best_plan = max(plan_scores.items(), key=lambda kv: kv[1])[0]
            selected_plan = best_plan
            switch_applied = bool(previous_plan and previous_plan != best_plan)
            stay_margin = 0.0
            if previous_plan and previous_plan in plan_scores:
                stay_margin = 0.015 + 0.07 * previous_dwell * max(1.0 - risk_score, 0.0)
                gain_over_prev = float(plan_scores[best_plan] - plan_scores[previous_plan])
                if gain_over_prev <= stay_margin:
                    selected_plan = previous_plan
                    switch_applied = False

            self.plan_memory[int(avatar_id)] = {
                "plan": selected_plan,
                "page_index": int(page_index),
                "risk_score": float(risk_score),
                "dwell_score": float(dwell_scores.get(selected_plan, 0.0)),
            }
            chosen = plan_results[selected_plan]
            self._register_selected_credit_from_result(int(avatar_id), int(page_index), chosen)
            return chosen["selected"], {
                "plan": selected_plan,
                "used_model": self.model is not None,
                "used_enjoy_model": self.enjoy_model is not None,
                "used_planner_model": self.planner_model is not None,
                "used_planner_dwell_model": self.planner_dwell_model is not None,
                "previous_plan": previous_plan,
                "switch_applied": switch_applied,
                "stay_margin": round(float(stay_margin), 4),
                "risk_score": round(float(risk_score), 4),
                "planner_prior": {k: round(float(v), 4) for k, v in plan_prior.items()},
                "planner_dwell": {k: round(float(v), 4) for k, v in dwell_scores.items()},
                "plan_values": {k: round(float(plan_results[k]["plan_value"]), 4) for k in PLAN_OPTIONS},
                "plan_scores": {k: round(float(plan_scores[k]), 4) for k in PLAN_OPTIONS},
                "top_candidates": chosen["top_candidates"],
            }

        if self.score_profile in VALUEMIX_PROFILES:
            plan_prior = self.predict_plan_distribution(user_profile, state)
            if self.score_profile in VALUEMIX_ANCHOR_PROFILES:
                anchor_plan = choose_plan(user_profile, state, override="auto")
                anchored = {plan: 0.0 for plan in PLAN_OPTIONS}
                progress = float(state.get("page_progress", 0.0))
                anchor_strength = 0.55 if progress <= 0.25 else (0.35 if progress <= 0.5 else 0.15)
                for plan_name in PLAN_OPTIONS:
                    anchored[plan_name] = (1.0 - anchor_strength) * float(plan_prior.get(plan_name, 0.0))
                anchored[anchor_plan] = anchored.get(anchor_plan, 0.0) + anchor_strength
                total_anchor = sum(anchored.values())
                if total_anchor > 0:
                    plan_prior = {plan_name: float(val / total_anchor) for plan_name, val in anchored.items()}
            plan_results: Dict[str, Dict[str, Any]] = {}
            aggregate_scores: Dict[int, float] = {}
            aggregate_support: Dict[int, List[float]] = {}
            aggregate_plan_contrib: Dict[int, Dict[str, float]] = {}

            for plan_name in PLAN_OPTIONS:
                result = self._score_candidates_for_plan(
                    arena=arena,
                    user_profile=user_profile,
                    state=state,
                    candidate_ids=candidate_ids,
                    plan=plan_name,
                    items_per_page=items_per_page,
                    avatar_id=int(avatar_id),
                )
                plan_results[plan_name] = result
                prior = float(plan_prior.get(plan_name, 0.0))
                if prior <= 0.0 or not result["kept_ids"]:
                    continue
                norm_scores = minmax_scale(result["final_scores"])
                for idx, iid in enumerate(result["kept_ids"]):
                    blended = 0.6 * norm_scores[idx] + 0.4 * max(float(result["final_scores"][idx]), -1.0)
                    aggregate_scores[iid] = float(aggregate_scores.get(iid, 0.0) + prior * blended)
                    aggregate_support.setdefault(iid, []).append(float(norm_scores[idx]))
                    aggregate_plan_contrib.setdefault(iid, {})[plan_name] = round(float(prior * blended), 4)

            if not aggregate_scores:
                fallback_plan = self.predict_plan(user_profile, state)
                return list(candidate_ids)[:items_per_page], {"plan": fallback_plan, "used_model": self.model is not None}

            final_mix_scores: Dict[int, float] = {}
            for iid, score in aggregate_scores.items():
                support = aggregate_support.get(iid, [])
                consensus = harmonic_mean_positive(support) if support else 0.0
                final_mix_scores[iid] = float(score + 0.06 * consensus)

            ordered_items = [iid for iid, _ in sorted(final_mix_scores.items(), key=lambda kv: kv[1], reverse=True)]
            selected = ordered_items[: max(int(items_per_page), 1)]
            debug_rows = []
            for iid in ordered_items[:5]:
                info = self.item_catalog.get(iid, {})
                debug_rows.append(
                    {
                        "item_id": iid,
                        "title": info.get("title", ""),
                        "mix_score": round(float(final_mix_scores[iid]), 4),
                        "plan_contrib": aggregate_plan_contrib.get(iid, {}),
                    }
                )
            return selected, {
                "plan": "mixture",
                "used_model": self.model is not None,
                "used_enjoy_model": self.enjoy_model is not None,
                "used_planner_model": self.planner_model is not None,
                "planner_model_kind": self.planner_model_kind,
                "planner_prior": {k: round(float(v), 4) for k, v in plan_prior.items()},
                "top_candidates": debug_rows,
            }

        if self.score_profile in cf_profiles:
            plan_prior = self.predict_plan_distribution(user_profile, state)
            plan_results: Dict[str, Dict[str, Any]] = {}
            plan_scores: Dict[str, float] = {}
            for plan_name in PLAN_OPTIONS:
                result = self._score_candidates_for_plan(
                    arena=arena,
                    user_profile=user_profile,
                    state=state,
                    candidate_ids=candidate_ids,
                    plan=plan_name,
                    items_per_page=items_per_page,
                )
                prior = max(float(plan_prior.get(plan_name, 0.0)), 1e-6)
                combined = float(result["plan_value"]) + 0.08 * math.log(prior)
                plan_results[plan_name] = result
                plan_scores[plan_name] = combined

            if not plan_results:
                fallback_plan = self.predict_plan(user_profile, state)
                return list(candidate_ids)[:items_per_page], {"plan": fallback_plan, "used_model": self.model is not None}

            cf_best_plan = max(plan_scores.items(), key=lambda kv: kv[1])[0]
            plan = cf_best_plan
            planner_choice = max(plan_prior.items(), key=lambda kv: kv[1])[0]
            planner_confidence = max(float(v) for v in plan_prior.values()) if plan_prior else 0.0
            cf_gain = float(plan_scores.get(cf_best_plan, -1e9) - plan_scores.get(planner_choice, -1e9))
            override_applied = bool(cf_best_plan != planner_choice)

            if self.score_profile in gated_cf_profiles:
                if (
                    cf_best_plan != planner_choice
                    and (planner_confidence >= CF_OVERRIDE_MAX_PRIOR or cf_gain < CF_OVERRIDE_MIN_GAIN)
                ):
                    plan = planner_choice
                    override_applied = False

            chosen = plan_results[plan]
            self._register_selected_credit_from_result(int(avatar_id), int(page_index), chosen)
            return chosen["selected"], {
                "plan": plan,
                "used_model": self.model is not None,
                "used_enjoy_model": self.enjoy_model is not None,
                "used_planner_model": self.planner_model is not None,
                "planner_choice": planner_choice,
                "counterfactual_best_plan": cf_best_plan,
                "planner_confidence": round(float(planner_confidence), 4),
                "cf_gain": round(float(cf_gain), 4),
                "override_applied": override_applied,
                "planner_prior": {k: round(float(v), 4) for k, v in plan_prior.items()},
                "plan_values": {k: round(float(plan_results[k]["plan_value"]), 4) for k in PLAN_OPTIONS},
                "plan_scores": {k: round(float(plan_scores[k]), 4) for k in PLAN_OPTIONS},
                "top_candidates": chosen["top_candidates"],
            }

        plan = self.predict_plan(user_profile, state)
        chosen = self._score_candidates_for_plan(
            arena=arena,
            user_profile=user_profile,
            state=state,
            candidate_ids=candidate_ids,
            plan=plan,
            items_per_page=items_per_page,
            avatar_id=int(avatar_id),
        )
        if not chosen["rows"]:
            return list(candidate_ids)[:items_per_page], {"plan": plan, "used_model": self.model is not None}
        self._register_selected_credit_from_result(int(avatar_id), int(page_index), chosen)
        return chosen["selected"], {
            "plan": plan,
            "used_model": self.model is not None,
            "used_enjoy_model": self.enjoy_model is not None,
            "used_planner_model": self.planner_model is not None,
            "top_candidates": chosen["top_candidates"],
        }
