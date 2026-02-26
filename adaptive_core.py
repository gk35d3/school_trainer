import random
from typing import Any, Dict, List, Tuple

from trainer_data import now_ts


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def update_tag_stats(state: Dict[str, Any], tags: List[str], correct: bool, rt: float, tag_window: int) -> None:
    for tag in tags:
        t = state["tags"].setdefault(tag, {"attempts": []})
        t["attempts"].append({"correct": bool(correct), "rt": float(rt), "ts": now_ts()})
        if len(t["attempts"]) > tag_window:
            t["attempts"] = t["attempts"][-tag_window:]


def tag_metrics(state: Dict[str, Any], tag: str, default_acc: float, default_rt: float) -> Tuple[float, float, int]:
    t = state["tags"].get(tag, {}).get("attempts", [])
    n = len(t)
    if n == 0:
        return (default_acc, default_rt, 0)
    acc = sum(1 for a in t if a.get("correct")) / n
    avg_rt = sum(a.get("rt", default_rt) for a in t) / n
    return (acc, avg_rt, n)


def update_overall_difficulty(
    state: Dict[str, Any],
    *,
    default_acc: float,
    default_rt: float,
    rt_good: float,
    rt_bad: float,
    smooth_old: float,
    smooth_new: float,
) -> None:
    tags = list(state["tags"].keys())
    if not tags:
        return

    scores = []
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag, default_acc, default_rt)
        rt_norm = clamp((avg_rt - rt_good) / (rt_bad - rt_good), 0.0, 1.0)
        score = (acc * 0.75) + ((1.0 - rt_norm) * 0.25)
        scores.append((score, n ** 0.5))

    total_w = sum(w for _, w in scores) or 1.0
    overall_score = sum(s * w for s, w in scores) / total_w
    target = clamp((overall_score - 0.50) / 0.50, 0.0, 1.0)
    state["difficulty"] = clamp(smooth_old * float(state["difficulty"]) + smooth_new * target, 0.0, 1.0)


def build_state_from_events(
    events: List[Dict[str, Any]],
    *,
    app_id: str,
    initial_difficulty: float,
    default_acc: float,
    default_rt: float,
    tag_window: int,
    rt_good: float,
    rt_bad: float,
    smooth_old: float,
    smooth_new: float,
    total_seen_key: str,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"difficulty": initial_difficulty, total_seen_key: 0, "tags": {}}
    for ev in events:
        if ev.get("app") != app_id or ev.get("type") != "attempt":
            continue
        tags = ev.get("tags") or []
        correct = bool(ev.get("correct", False))
        rt = float(ev.get("rt", default_rt))
        update_tag_stats(state, tags, correct, rt, tag_window)
        if correct:
            state[total_seen_key] = int(state[total_seen_key]) + 1

    update_overall_difficulty(
        state,
        default_acc=default_acc,
        default_rt=default_rt,
        rt_good=rt_good,
        rt_bad=rt_bad,
        smooth_old=smooth_old,
        smooth_new=smooth_new,
    )
    return state


def weighted_pick_tag(
    state: Dict[str, Any],
    allowed_tags: List[str],
    *,
    default_acc: float,
    default_rt: float,
    rt_good: float,
    rt_bad: float,
    base_weight: float,
    explore_bonus: float,
    focus_boosts: Dict[str, float],
) -> str:
    weights = []
    for tag in allowed_tags:
        acc, avg_rt, n = tag_metrics(state, tag, default_acc, default_rt)
        rt_norm = clamp((avg_rt - rt_good) / (rt_bad - rt_good), 0.0, 1.0)
        weakness = (1.0 - acc) * 0.7 + rt_norm * 0.3
        explore = explore_bonus if n == 0 else 0.0
        w = base_weight + weakness + explore + focus_boosts.get(tag, 0.0)
        weights.append((tag, w))

    total = sum(w for _, w in weights)
    r = random.random() * total
    upto = 0.0
    for tag, w in weights:
        upto += w
        if upto >= r:
            return tag
    return weights[-1][0]
