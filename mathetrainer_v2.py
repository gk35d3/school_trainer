import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import pygame

# =========================
# Config
# =========================
FPS = 60
SESSION_QUESTIONS = 50
CORRECT_PAUSE_SECONDS = 0.6

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(APP_DIR, "state.json")
SESSIONS_DIR = os.path.join(APP_DIR, "sessions")

# Warm-up: start a bit easier than stored skill
WARMUP_DIFFICULTY_OFFSET = 0.08  # subtract from stored difficulty at session start

# Constraints for kid-friendliness
ALLOW_NEGATIVES = False

# Difficulty ramp inside a session
# session_progress in [0..1], we map to +0.00..+0.25 added difficulty
RAMP_MAX_BONUS = 0.25

# Rolling stats per tag
TAG_WINDOW = 60  # keep last N attempts per tag in state


# =========================
# Persistence
# =========================
def ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def now_ts() -> float:
    return time.time()


def load_state() -> Dict[str, Any]:
    default = {
        "difficulty": 0.15,     # overall 0..1
        "total_questions_seen": 0,
        "tags": {
            # tag: {"attempts":[{"correct":bool,"rt":float,"ts":float}]}
        }
    }
    if not os.path.exists(STATE_PATH):
        return default
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # merge defaults
        for k, v in default.items():
            if k not in data:
                data[k] = v
        if "tags" not in data or not isinstance(data["tags"], dict):
            data["tags"] = default["tags"]
        return data
    except Exception:
        return default


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def update_tag_stats(state: Dict[str, Any], tags: List[str], correct: bool, rt: float) -> None:
    for tag in tags:
        t = state["tags"].setdefault(tag, {"attempts": []})
        t["attempts"].append({"correct": bool(correct), "rt": float(rt), "ts": now_ts()})
        if len(t["attempts"]) > TAG_WINDOW:
            t["attempts"] = t["attempts"][-TAG_WINDOW:]


def tag_metrics(state: Dict[str, Any], tag: str) -> Tuple[float, float, int]:
    """
    Returns (accuracy 0..1, avg_rt seconds, n)
    If no data, return neutral-ish defaults.
    """
    t = state["tags"].get(tag, {}).get("attempts", [])
    n = len(t)
    if n == 0:
        return (0.65, 9.0, 0)  # unknown => slightly weak and slow
    acc = sum(1 for a in t if a.get("correct")) / n
    avg_rt = sum(a.get("rt", 9.0) for a in t) / n
    return (acc, avg_rt, n)


def update_overall_difficulty(state: Dict[str, Any]) -> None:
    """
    Compute an overall difficulty from tag performance, gently.
    - If most practiced tags are strong, increase.
    - If many are weak, decrease slightly.
    """
    tags = list(state["tags"].keys())
    if not tags:
        return

    scores = []
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        # score: accuracy high good, rt high bad
        # normalize rt: 3s good, 12s bad
        rt_norm = clamp((avg_rt - 3.0) / (12.0 - 3.0), 0.0, 1.0)
        score = (acc * 0.75) + ((1.0 - rt_norm) * 0.25)
        # weigh by sqrt(n) so more evidence counts
        weight = (n ** 0.5)
        scores.append((score, weight))

    total_w = sum(w for _, w in scores) or 1.0
    overall_score = sum(s * w for s, w in scores) / total_w  # 0..1-ish

    # Map overall_score to difficulty gently
    # overall_score 0.55 -> ~0.15, 0.75 -> ~0.45, 0.90 -> ~0.75
    target = clamp((overall_score - 0.50) / 0.50, 0.0, 1.0)

    # Smooth update
    state["difficulty"] = clamp(0.90 * float(state["difficulty"]) + 0.10 * target, 0.0, 1.0)


# =========================
# Problem model + tagging
# =========================
@dataclass
class Problem:
    a: int
    b: int
    op: str
    tags: List[str]

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b


def is_crossing_10_add(a: int, b: int) -> bool:
    return (a % 10) + (b % 10) >= 10


def is_carry_add(a: int, b: int) -> bool:
    # carry in ones for two-digit add
    return a >= 10 and b >= 10 and (a % 10) + (b % 10) >= 10


def is_borrow_sub(a: int, b: int) -> bool:
    # borrow from tens in two-digit subtract
    return a >= 10 and b >= 10 and (a % 10) < (b % 10)


def is_tens(n: int) -> bool:
    return n % 10 == 0


def assign_tags(a: int, b: int, op: str) -> List[str]:
    tags = []
    if op == "+":
        tags.append("add")
        if a <= 10 and b <= 10:
            tags.append("add_small")
        if is_tens(a) and is_tens(b):
            tags.append("tens")
            tags.append("add_tens")
        if a >= 10 and b >= 10:
            tags.append("add_two_digit")
            if is_carry_add(a, b):
                tags.append("add_carry")
            else:
                tags.append("add_no_carry")
        if is_crossing_10_add(a, b):
            tags.append("add_cross10")
    else:
        tags.append("sub")
        if a <= 10 and b <= 10:
            tags.append("sub_small")
        if is_tens(a) and is_tens(b):
            tags.append("tens")
            tags.append("sub_tens")
        if a >= 10 and b >= 10:
            tags.append("sub_two_digit")
            if is_borrow_sub(a, b):
                tags.append("sub_borrow")
            else:
                tags.append("sub_no_borrow")
    return tags


# =========================
# Problem generation by "target tag" + difficulty
# =========================
def choose_op(difficulty: float) -> str:
    # start more addition; increase subtraction as skill rises
    p_sub = 0.25 + 0.35 * difficulty
    return "-" if random.random() < p_sub else "+"


def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    """
    Choose a tag to practice.
    Prefer weaker tags (low acc, high rt), but keep some variety.
    """
    weights = []
    for tag in allowed_tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        # weakness: low acc and high rt
        rt_norm = clamp((avg_rt - 3.0) / 9.0, 0.0, 1.0)  # 0..1
        weakness = (1.0 - acc) * 0.7 + rt_norm * 0.3
        # if no data, treat as medium-weak to explore
        explore_boost = 0.15 if n == 0 else 0.0
        w = 0.20 + weakness + explore_boost
        weights.append((tag, w))

    # add a bit of randomness / exploration
    total = sum(w for _, w in weights)
    r = random.random() * total
    upto = 0.0
    for tag, w in weights:
        upto += w
        if upto >= r:
            return tag
    return weights[-1][0]


def difficulty_to_limits(difficulty: float) -> Dict[str, Any]:
    """
    Translate difficulty to number ranges + feature allowances.
    """
    # smooth ranges
    max_small = int(8 + 22 * difficulty)          # 8..30
    max_mid = int(15 + 55 * difficulty)           # 15..70
    max_two = int(20 + 79 * difficulty)           # 20..99

    allow_over_99 = difficulty >= 0.80  # later allow addition > 99
    return {
        "max_small": max_small,
        "max_mid": max_mid,
        "max_two": max_two,
        "allow_over_99": allow_over_99
    }


def make_problem_for_target(state: Dict[str, Any], difficulty: float, target_tag: str) -> Problem:
    """
    Generate a problem intended to match a target tag (weakness area),
    while respecting difficulty constraints and kid-friendliness.
    """
    limits = difficulty_to_limits(difficulty)
    allow_over_99 = limits["allow_over_99"]
    max_small = limits["max_small"]
    max_mid = limits["max_mid"]
    max_two = limits["max_two"]

    # Basic plan: choose op, then shape operands to likely hit target tag.
    # We generate until the tags contain target_tag and constraints are met.
    # (This keeps code simpler & adaptive.)
    for _ in range(400):
        # Decide op: if target is add_* or sub_* force it; else use choose_op
        if target_tag.startswith("add"):
            op = "+"
        elif target_tag.startswith("sub"):
            op = "-"
        else:
            op = choose_op(difficulty)

        # operand generation "styles" based on target
        if target_tag in ("add_small", "sub_small"):
            a = random.randint(0, max_small)
            b = random.randint(0, max_small)

        elif target_tag in ("tens", "add_tens", "sub_tens"):
            tens_max = max(10, (int(1 + difficulty * 9)) * 10)  # 10..90
            choices = list(range(10, min(90, tens_max) + 1, 10))
            a = random.choice(choices)
            b = random.choice(choices)

        elif target_tag in ("add_cross10",):
            # force crossing 10 in ones place: a1 in 1..9 and b1 >= (10 - a1)
            # so that (a%10) + (b%10) >= 10 is always achievable.
            a10 = random.randint(0, max_mid // 10)
            a1 = random.randint(1, 9)  # IMPORTANT: never 0
            a = a10 * 10 + a1
            a = max(1, min(a, max_mid))

            low_b1 = 10 - (a % 10)      # in 1..9 now
            low_b1 = max(0, min(9, low_b1))
            # low_b1 is guaranteed <= 9 because a1 != 0
            b1 = random.randint(low_b1, 9)

            b10 = random.randint(0, max(0, (max_mid - b1) // 10))
            b = b10 * 10 + b1
            b = min(b, max_mid)

        elif target_tag in ("add_carry", "add_no_carry", "sub_borrow", "sub_no_borrow", "add_two_digit", "sub_two_digit"):
            # focus on two-digit
            a = random.randint(10, max_two)
            b = random.randint(10, max_two)

            if target_tag == "add_carry":
                # enforce carry in ones
                a1 = random.randint(0, 9)
                b1 = random.randint(max(0, 10 - a1), 9)
                a10 = random.randint(1, 9)
                b10 = random.randint(1, 9)
                a = a10 * 10 + a1
                b = b10 * 10 + b1

            if target_tag == "add_no_carry":
                a1 = random.randint(0, 9)
                b1 = random.randint(0, 9 - a1)
                a10 = random.randint(1, 9)
                b10 = random.randint(1, 9)
                a = a10 * 10 + a1
                b = b10 * 10 + b1

            if target_tag == "sub_borrow":
                # enforce borrow in ones: a1 < b1 but keep a>=b
                a10 = random.randint(2, 9)
                b10 = random.randint(1, a10)
                a1 = random.randint(0, 8)
                b1 = random.randint(a1 + 1, 9)
                a = a10 * 10 + a1
                b = b10 * 10 + b1
                if b > a:
                    a, b = b, a  # keep non-negative

            if target_tag == "sub_no_borrow":
                a10 = random.randint(1, 9)
                b10 = random.randint(1, a10)
                a1 = random.randint(0, 9)
                b1 = random.randint(0, a1)
                a = a10 * 10 + a1
                b = b10 * 10 + b1
                if b > a:
                    a, b = b, a

        else:
            # fallback mixed
            a = random.randint(0, max_mid)
            b = random.randint(0, max_mid)

        # enforce kid-friendly subtraction non-negative
        if op == "-" and not ALLOW_NEGATIVES:
            if b > a:
                a, b = b, a

        p_tags = assign_tags(a, b, op)
        ans = a + b if op == "+" else a - b

        # constraints: keep results <= 99 until difficulty high
        if not allow_over_99 and ans > 99:
            continue
        if not ALLOW_NEGATIVES and ans < 0:
            continue

        if target_tag not in p_tags:
            # allow some variety: small chance accept non-matching to avoid “stuck”
            if random.random() > 0.12:
                continue

        return Problem(a=a, b=b, op=op, tags=p_tags)

    # if we fail to match, fallback simple
    op = choose_op(difficulty)
    a = random.randint(0, limits["max_mid"])
    b = random.randint(0, limits["max_mid"])
    if op == "-" and b > a:
        a, b = b, a
    return Problem(a=a, b=b, op=op, tags=assign_tags(a, b, op))


def allowed_tags_for_difficulty(difficulty: float) -> List[str]:
    """
    Which skill categories are appropriate at this difficulty?
    We keep this small & smooth.
    """
    tags = ["add_small", "sub_small", "tens", "add_cross10"]

    if difficulty >= 0.30:
        tags += ["add_two_digit", "sub_two_digit", "add_no_carry", "sub_no_borrow"]
    if difficulty >= 0.55:
        tags += ["add_carry", "sub_borrow"]
    if difficulty >= 0.75:
        # still same tags, but allow_over_99 will unlock bigger results (addition)
        tags += ["add_carry", "sub_borrow"]

    # de-duplicate while preserving order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# =========================
# UI helpers
# =========================
def draw_progress_bar(surface, rect: pygame.Rect, frac_0_1: float):
    pygame.draw.rect(surface, (30, 30, 40), rect, border_radius=10)
    inner = rect.inflate(-6, -6)
    fill_w = int(inner.width * clamp(frac_0_1, 0.0, 1.0))
    fill_rect = pygame.Rect(inner.left, inner.top, fill_w, inner.height)
    pygame.draw.rect(surface, (80, 220, 120), fill_rect, border_radius=8)
    pygame.draw.rect(surface, (70, 70, 90), rect, width=2, border_radius=10)


def digits_only_append(user_text: str, key: int) -> str:
    if pygame.K_0 <= key <= pygame.K_9:
        return user_text + chr(key)
    if pygame.K_KP0 <= key <= pygame.K_KP9:
        return user_text + chr(key - pygame.K_KP0 + ord("0"))
    return user_text


# =========================
# Main app
# =========================
def main():
    ensure_dirs()
    state = load_state()

    # Open session logs
    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    attempts_path = os.path.join(SESSIONS_DIR, session_name + "_attempts.jsonl")
    questions_path = os.path.join(SESSIONS_DIR, session_name + "_questions.jsonl")

    attempts_f = open(attempts_path, "a", encoding="utf-8")
    questions_f = open(questions_path, "a", encoding="utf-8")

    pygame.init()
    pygame.display.set_caption("Math Trainer (Weakness + 50)")

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pygame.font.SysFont(None, int(base * 1.6))
    font_input = pygame.font.SysFont(None, int(base * 1.1))
    font_hint = pygame.font.SysFont(None, int(base * 0.40))
    font_small = pygame.font.SysFont(None, int(base * 0.32))

    # Session difficulty start: slightly easier than stored
    session_base_difficulty = clamp(float(state["difficulty"]) - WARMUP_DIFFICULTY_OFFSET, 0.0, 1.0)

    # Session counters
    q_index = 0  # 0..49
    completed = False

    # Current question state
    user_text = ""
    feedback: Optional[str] = None  # "correct"/"wrong"
    feedback_since = 0.0
    attempts_for_problem = 0
    problem_start = now_ts()
    problem_solved = False

    # Pre-generate first problem
    def current_session_difficulty() -> float:
        # ramp inside session
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        ramp = RAMP_MAX_BONUS * prog
        return clamp(session_base_difficulty + ramp, 0.0, 1.0)

    def pick_next_problem() -> Problem:
        d = current_session_difficulty()
        allowed = allowed_tags_for_difficulty(d)
        target = pick_target_tag(state, allowed)
        return make_problem_for_target(state, d, target)

    problem = pick_next_problem()

    def log_attempt(correct: bool, typed: str, rt: float):
        rec = {
            "t": now_ts(),
            "q_index": q_index + 1,
            "a": problem.a,
            "b": problem.b,
            "op": problem.op,
            "answer": problem.answer,
            "typed": typed,
            "correct": bool(correct),
            "attempt": attempts_for_problem,
            "rt": float(rt),
            "session_difficulty": float(current_session_difficulty()),
            "overall_difficulty": float(state["difficulty"]),
            "tags": problem.tags,
        }
        attempts_f.write(json.dumps(rec) + "\n")
        attempts_f.flush()

    def log_question_summary(final_correct: bool, total_attempts: int, total_time: float):
        rec = {
            "t": now_ts(),
            "q_index": q_index + 1,
            "a": problem.a,
            "b": problem.b,
            "op": problem.op,
            "answer": problem.answer,
            "final_correct": bool(final_correct),
            "attempts": int(total_attempts),
            "time_total": float(total_time),
            "session_difficulty": float(current_session_difficulty()),
            "tags": problem.tags,
        }
        questions_f.write(json.dumps(rec) + "\n")
        questions_f.flush()

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                # once session completed: ignore all keys except ESC
                if completed:
                    continue

                # during correct pause: ignore typing
                if feedback == "correct":
                    continue

                if event.key == pygame.K_RETURN:
                    if user_text.strip() == "":
                        continue

                    attempts_for_problem += 1
                    rt = now_ts() - problem_start

                    try:
                        val = int(user_text)
                    except ValueError:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, problem.tags, correct=False, rt=rt)
                        save_state(state)
                        continue

                    if val == problem.answer:
                        feedback = "correct"
                        feedback_since = now_ts()
                        log_attempt(True, user_text, rt)

                        if not problem_solved:
                            problem_solved = True
                            update_tag_stats(state, problem.tags, correct=True, rt=rt)
                            state["total_questions_seen"] = int(state.get("total_questions_seen", 0)) + 1
                            update_overall_difficulty(state)
                            save_state(state)

                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, problem.tags, correct=False, rt=rt)
                        update_overall_difficulty(state)
                        save_state(state)

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if len(user_text) < 4:
                        user_text = digits_only_append(user_text, event.key)

        # After correct pause: advance to next question OR finish session
        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                # record question summary
                total_time = now_ts() - problem_start
                log_question_summary(final_correct=True, total_attempts=attempts_for_problem, total_time=total_time)

                q_index += 1
                if q_index >= SESSION_QUESTIONS:
                    completed = True
                else:
                    # next question
                    problem = pick_next_problem()
                    user_text = ""
                    feedback = None
                    attempts_for_problem = 0
                    problem_start = now_ts()
                    problem_solved = False

        # If you want “move on even if not solved” after some attempts,
        # you can add a rule here. For now, it stays until solved.
        # (But still only counts as 1 of the 50 questions.)

        # ----------------------------
        # DRAW
        # ----------------------------
        screen.fill((10, 10, 14))

        if completed:
            # Session complete screen
            msg = "SESSION COMPLETE"
            surf = font_task.render(msg, True, (80, 220, 120))
            rect = surf.get_rect(center=(w // 2, h // 2))
            screen.blit(surf, rect)

            msg2 = "Press ESC to exit"
            surf2 = font_hint.render(msg2, True, (160, 160, 170))
            rect2 = surf2.get_rect(center=(w // 2, int(h * 0.60)))
            screen.blit(surf2, rect2)

            # progress bar full
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)

        else:
            # Problem
            task_text = f"{problem.a}  {problem.op}  {problem.b}  ="
            surf_task = font_task.render(task_text, True, (240, 240, 240))
            rect_task = surf_task.get_rect(center=(w // 2, int(h * 0.40)))
            screen.blit(surf_task, rect_task)

            # Input color
            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text != "" else " "
            surf_in = font_input.render(shown_input, True, input_color)
            rect_in = surf_in.get_rect(center=(w // 2, int(h * 0.55)))
            screen.blit(surf_in, rect_in)

            # Feedback message (ASCII only)
            if feedback == "wrong":
                msg = "Try again"
                surf_msg = font_hint.render(msg, True, (240, 90, 90))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "correct":
                msg = "Correct"
                surf_msg = font_hint.render(msg, True, (80, 220, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)

            # Bottom progress bar: question progress 1..50
            frac = (q_index / SESSION_QUESTIONS)
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, frac)

            # small UI text
            hint = "ESC: exit   ENTER: check   BACKSPACE: delete"
            screen.blit(font_hint.render(hint, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.86)))

            # question count
            qtxt = f"Question {q_index + 1}/{SESSION_QUESTIONS}"
            screen.blit(font_small.render(qtxt, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.06)))

            # (Optional) show difficulty number for parents (small)
            dtxt = f"Difficulty {current_session_difficulty():.2f}"
            screen.blit(font_small.render(dtxt, True, (120, 120, 140)), (int(w * 0.10), int(h * 0.09)))

        pygame.display.flip()

    attempts_f.close()
    questions_f.close()
    pygame.quit()


if __name__ == "__main__":
    main()
