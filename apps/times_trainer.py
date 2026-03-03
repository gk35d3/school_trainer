import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame

from core.adaptive_core import (
    build_state_from_events,
    clamp,
    latest_logged_difficulty,
    tag_metrics,
    update_overall_difficulty,
    update_tag_stats,
    weighted_pick_tag,
)
from core.trainer_data import append_event, load_recent_events, now_ts

# =========================
# Config
# =========================
# Objective: Define session pacing and adaptive tuning for fast 1x1 recall.
FPS = 60
SESSION_QUESTIONS = 20
SESSION_MAX_SECONDS = 15 * 60
CORRECT_PAUSE_SECONDS = 1.35
REVEAL_PAUSE_SECONDS = 2.60
SLOW_HINT_SECONDS = 6.0
MAX_TRIES_PER_PROBLEM = 2

RAMP_MAX_BONUS = 0.18
TAG_WINDOW = 100
MAX_INPUT_CHARS = 3

APP_ID = "times"

# Objective: Prioritize difficult multiplication rows/facts for repetition.
FOCUS_BOOSTS = {
    "anchor": 0.30,
    "doubles": 0.18,
    "near10": 0.25,
    "hard": 0.45,
    "mul_7": 0.40,
    "mul_8": 0.25,
    "mul_9": 0.30,
    "fact_6x7": 0.40,
    "fact_6x8": 0.28,
    "fact_7x8": 0.46,
    "fact_7x9": 0.46,
    "fact_8x9": 0.42,
}

# Objective: Configure default prior estimates for unseen tags.
DEFAULT_ACC = 0.60
DEFAULT_RT = 5.0

ANCHOR_NUMBERS = {1, 2, 5, 10}
DOUBLE_PATTERN_NUMBERS = {2, 4, 8}
HARD_FACTS = {(6, 7), (6, 8), (7, 8), (7, 9), (8, 9)}


# =========================
# Helpers
# =========================
# Objective: Normalize unicode text so umlauts render as single characters.
def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# Objective: Pick a font that reliably supports German umlauts.
FONT_CACHE: Dict[int, pygame.font.Font] = {}


def pick_unicode_font(size: int):
    if size in FONT_CACHE:
        return FONT_CACHE[size]
    for name in ("DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            FONT_CACHE[size] = f
            return f
    fallback = pygame.font.SysFont(None, size)
    FONT_CACHE[size] = fallback
    return fallback


# Objective: Rebuild times state from the shared event log.
def build_state_from_log(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return build_state_from_events(
        events,
        app_id=APP_ID,
        initial_difficulty=0.18,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        tag_window=TAG_WINDOW,
        rt_good=2.5,
        rt_bad=8.0,
        smooth_old=0.90,
        smooth_new=0.10,
        total_seen_key="total_questions_seen",
    )


# Objective: Build canonical fact identity for spaced repetition across orientations.
def canonical_fact(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


# Objective: Convert fact pair to stable tag name.
def fact_tag(a: int, b: int) -> str:
    x, y = canonical_fact(a, b)
    return f"fact_{x}x{y}"


# Objective: Identify high-friction facts that need extra practice.
def is_hard_fact(a: int, b: int) -> bool:
    return canonical_fact(a, b) in HARD_FACTS


# =========================
# Problem model + tagging
# =========================
# Objective: Represent one generated multiplication task.
@dataclass
class Problem:
    a: int
    b: int
    op: str
    tags: List[str]

    @property
    def answer(self) -> int:
        return self.a * self.b


# Objective: Assign row/fact/pattern tags used by the adaptive engine.
def assign_tags(a: int, b: int) -> List[str]:
    tags: List[str] = [
        "mul",
        f"mul_{a}",
        f"mul_{b}",
        fact_tag(a, b),
    ]

    if a in ANCHOR_NUMBERS or b in ANCHOR_NUMBERS:
        tags.append("anchor")
    if a == 9 or b == 9:
        tags.append("near10")
    if a in DOUBLE_PATTERN_NUMBERS or b in DOUBLE_PATTERN_NUMBERS:
        tags.append("doubles")
    if is_hard_fact(a, b):
        tags.append("hard")

    return tags


# =========================
# Generation
# =========================
# Objective: Gate curriculum rows and partner range by current difficulty.
def allowed_rows_and_partner_max(difficulty: float) -> Tuple[List[int], int]:
    if difficulty < 0.25:
        return ([1, 2, 5, 10], 5)
    if difficulty < 0.55:
        return ([1, 2, 3, 4, 5, 6, 8, 10], 10)
    if difficulty < 0.80:
        return ([1, 2, 3, 4, 5, 6, 8, 9, 10], 10)
    return ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)


# Objective: Enumerate canonical facts available in the current curriculum gate.
def fact_candidates_for_difficulty(difficulty: float) -> List[Tuple[int, int]]:
    rows, partner_max = allowed_rows_and_partner_max(difficulty)
    facts: Set[Tuple[int, int]] = set()
    for row in rows:
        for n in range(1, partner_max + 1):
            facts.add(canonical_fact(row, n))
    return sorted(facts)


# Objective: Build adaptive target tag pool from rows, facts, and patterns.
def allowed_target_tags_for_difficulty(difficulty: float) -> List[str]:
    rows, _ = allowed_rows_and_partner_max(difficulty)
    tags: Set[str] = {"anchor", "doubles"}

    for row in rows:
        tags.add(f"mul_{row}")

    for x, y in fact_candidates_for_difficulty(difficulty):
        tags.add(f"fact_{x}x{y}")

    if 9 in rows:
        tags.add("near10")

    if 7 in rows:
        tags.add("hard")
        for x, y in HARD_FACTS:
            tags.add(f"fact_{x}x{y}")

    return sorted(tags)


# Objective: Pick next weak tag with exploration and focus boosts.
def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    return weighted_pick_tag(
        state,
        allowed_tags,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        rt_good=2.5,
        rt_bad=8.0,
        base_weight=0.20,
        explore_bonus=0.18,
        focus_boosts=FOCUS_BOOSTS,
    )


# Objective: Choose orientation while occasionally testing commutativity.
def choose_orientation(x: int, y: int, difficulty: float, recent_first_operands: List[int]) -> Tuple[int, int]:
    if x == y:
        return (x, y)

    ask_swapped = difficulty >= 0.35 and random.random() < 0.35
    primary = (y, x) if ask_swapped else (x, y)
    alternative = (x, y) if primary == (y, x) else (y, x)

    if len(recent_first_operands) >= 2 and primary[0] == recent_first_operands[-1] == recent_first_operands[-2]:
        return alternative
    return primary


# Objective: Generate one multiplication problem aligned to selected weak tag.
def make_problem_for_target(difficulty: float, target_tag: str, recent_first_operands: List[int]) -> Problem:
    facts = fact_candidates_for_difficulty(difficulty)

    targeted = []
    for x, y in facts:
        tags = assign_tags(x, y)
        if target_tag in tags:
            targeted.append((x, y))

    source = targeted if targeted else facts

    for _ in range(300):
        x, y = random.choice(source)
        a, b = choose_orientation(x, y, difficulty, recent_first_operands)
        tags = assign_tags(a, b)

        if target_tag not in tags and random.random() > 0.10:
            continue

        if len(recent_first_operands) >= 2 and a == recent_first_operands[-1] == recent_first_operands[-2]:
            if random.random() < 0.85:
                continue

        return Problem(a=a, b=b, op="*", tags=tags)

    x, y = random.choice(facts)
    a, b = choose_orientation(x, y, difficulty, recent_first_operands)
    return Problem(a=a, b=b, op="*", tags=assign_tags(a, b))


# Objective: Provide tiny German micro-hints only when needed.
def micro_hint(problem: Problem) -> str:
    a, b = problem.a, problem.b
    if 1 in (a, b):
        return "Mal 1 bleibt gleich."
    if 10 in (a, b):
        return "Häng 0 dran."
    if 2 in (a, b):
        return "Doppelt."
    if 5 in (a, b):
        return "0 oder 5 am Ende."
    if 9 in (a, b):
        return "9er: eins weniger."
    if 4 in (a, b):
        return "x4: doppelt doppelt."
    if 8 in (a, b):
        return "x8: dreimal doppeln."
    if 6 in (a, b):
        return "x6 = x5 + Zahl."
    if a != b:
        return "Tausch egal."
    if is_hard_fact(a, b):
        return "Ruhig in Schritten."
    return ""


# Objective: Trigger hints for slow answers or known weak tags.
def should_show_hint(state: Dict[str, Any], tags: List[str], rt: float) -> bool:
    if rt > SLOW_HINT_SECONDS:
        return True
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag, DEFAULT_ACC, DEFAULT_RT)
        if n >= 2 and (acc < 0.68 or avg_rt > SLOW_HINT_SECONDS):
            return True
    return False


# =========================
# UI
# =========================
# Objective: Render a consistent progress bar used by the trainer UI.
def draw_progress_bar(surface, rect: pygame.Rect, frac_0_1: float):
    pygame.draw.rect(surface, (30, 30, 40), rect, border_radius=10)
    inner = rect.inflate(-6, -6)
    fill_w = int(inner.width * clamp(frac_0_1, 0.0, 1.0))
    fill_rect = pygame.Rect(inner.left, inner.top, fill_w, inner.height)
    pygame.draw.rect(surface, (80, 220, 120), fill_rect, border_radius=8)
    pygame.draw.rect(surface, (70, 70, 90), rect, width=2, border_radius=10)


# Objective: Append digit keys (including numpad) to input text.
def digits_only_append(user_text: str, key: int) -> str:
    if pygame.K_0 <= key <= pygame.K_9:
        return user_text + chr(key)
    if pygame.K_KP0 <= key <= pygame.K_KP9:
        return user_text + chr(key - pygame.K_KP0 + ord("0"))
    return user_text


# Objective: Run one adaptive fullscreen times-table training session.
def main():
    events = load_recent_events()
    state = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("1x1 Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pick_unicode_font(int(base * 1.6))
    font_input = pick_unicode_font(int(base * 1.1))
    font_hint = pick_unicode_font(int(base * 0.40))
    font_small = pick_unicode_font(int(base * 0.32))

    session_id = f"times_{int(now_ts())}"
    session_start_ts = now_ts()
    session_base_difficulty = latest_logged_difficulty(events, APP_ID, float(state["difficulty"]))

    q_index = 0
    solved_count = 0
    completed = False
    session_end_logged = False

    user_text = ""
    feedback: Optional[str] = None
    feedback_since = 0.0
    attempts_for_problem = 0
    problem_start = now_ts()
    problem_solved = False
    hint_text = ""
    wait_for_enter_advance = False

    recent_first_operands: List[int] = []

    # Objective: Increase challenge slightly as the session progresses.
    def current_session_difficulty() -> float:
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        return clamp(session_base_difficulty + (RAMP_MAX_BONUS * prog), 0.0, 1.0)

    # Objective: Pick the next multiplication task from adaptive target tags.
    def pick_next_problem() -> Problem:
        d = current_session_difficulty()
        target = pick_target_tag(state, allowed_target_tags_for_difficulty(d))
        return make_problem_for_target(d, target, recent_first_operands)

    problem = pick_next_problem()

    append_event(
        {
            "type": "session_start",
            "app": APP_ID,
            "session_id": session_id,
            "questions_target": SESSION_QUESTIONS,
            "duration_limit_sec": SESSION_MAX_SECONDS,
            "difficulty_start": float(current_session_difficulty()),
        }
    )

    # Objective: Persist one attempt event for each ENTER press.
    def log_attempt(correct: bool, typed: str, rt: float):
        append_event(
            {
                "type": "attempt",
                "app": APP_ID,
                "session_id": session_id,
                "q_index": q_index + 1,
                "a": problem.a,
                "b": problem.b,
                "op": "*",
                "answer": problem.answer,
                "typed": typed,
                "correct": bool(correct),
                "attempt": attempts_for_problem,
                "rt": float(rt),
                "difficulty": float(current_session_difficulty()),
                "tags": problem.tags,
            }
        )

    # Objective: Update adaptive statistics after each attempt.
    def update_adaptive(correct: bool, rt: float):
        update_tag_stats(state, problem.tags, correct=correct, rt=rt, tag_window=TAG_WINDOW)
        update_overall_difficulty(
            state,
            default_acc=DEFAULT_ACC,
            default_rt=DEFAULT_RT,
            rt_good=2.5,
            rt_bad=8.0,
            smooth_old=0.90,
            smooth_new=0.10,
        )

    # Objective: Log session end once when finished by question count or time limit.
    def finish_session():
        nonlocal session_end_logged
        if session_end_logged:
            return
        session_end_logged = True
        append_event(
            {
                "type": "session_end",
                "app": APP_ID,
                "session_id": session_id,
                "questions_done": q_index,
                "correct_solved": solved_count,
                "elapsed_sec": float(now_ts() - session_start_ts),
                "difficulty_end": float(state["difficulty"]),
            }
        )

    # Objective: Move to next question while preserving compact runtime state.
    def advance_problem(solved: bool):
        nonlocal q_index, solved_count, completed, user_text, feedback, attempts_for_problem
        nonlocal problem_start, problem_solved, hint_text, problem, wait_for_enter_advance

        recent_first_operands.append(problem.a)
        if len(recent_first_operands) > 6:
            recent_first_operands[:] = recent_first_operands[-6:]

        q_index += 1
        if solved:
            solved_count += 1

        time_up = (now_ts() - session_start_ts) >= SESSION_MAX_SECONDS
        if q_index >= SESSION_QUESTIONS or time_up:
            completed = True
            finish_session()
            return

        problem = pick_next_problem()
        user_text = ""
        feedback = None
        attempts_for_problem = 0
        problem_start = now_ts()
        problem_solved = False
        hint_text = ""
        wait_for_enter_advance = False

    running = True
    while running:
        clock.tick(FPS)

        if not completed and (now_ts() - session_start_ts) >= SESSION_MAX_SECONDS:
            completed = True
            finish_session()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if completed:
                    continue

                if feedback in ("correct", "reveal"):
                    if wait_for_enter_advance and event.key == pygame.K_RETURN:
                        advance_problem(solved=(feedback == "correct"))
                    continue

                if event.key == pygame.K_RETURN:
                    if user_text.strip() == "":
                        continue

                    attempts_for_problem += 1
                    rt = now_ts() - problem_start

                    try:
                        val = int(user_text)
                    except ValueError:
                        val = -1

                    is_correct = val == problem.answer
                    log_attempt(is_correct, user_text, rt)
                    update_adaptive(is_correct, rt)

                    if is_correct:
                        feedback = "correct"
                        feedback_since = now_ts()
                        if should_show_hint(state, problem.tags, rt):
                            hint_text = micro_hint(problem)
                            wait_for_enter_advance = True
                        else:
                            hint_text = ""
                            wait_for_enter_advance = False

                        if not problem_solved:
                            problem_solved = True
                            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1
                    else:
                        hint_text = micro_hint(problem)
                        feedback_since = now_ts()
                        if attempts_for_problem >= MAX_TRIES_PER_PROBLEM:
                            feedback = "reveal"
                            wait_for_enter_advance = True
                        else:
                            feedback = "wrong"
                            wait_for_enter_advance = False
                        user_text = ""

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                    if feedback == "wrong":
                        feedback = None
                else:
                    if len(user_text) < MAX_INPUT_CHARS:
                        user_text = digits_only_append(user_text, event.key)
                        if feedback == "wrong":
                            feedback = None

        if not completed and feedback == "correct" and not wait_for_enter_advance:
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                advance_problem(solved=True)

        if not completed and feedback == "reveal" and not wait_for_enter_advance:
            if now_ts() - feedback_since >= REVEAL_PAUSE_SECONDS:
                advance_problem(solved=False)

        screen.fill((10, 10, 14))

        if completed:
            surf = font_task.render(to_nfc("FERTIG"), True, (80, 220, 120))
            rect = surf.get_rect(center=(w // 2, h // 2))
            screen.blit(surf, rect)

            msg2 = font_hint.render(to_nfc("ESC"), True, (160, 160, 170))
            rect2 = msg2.get_rect(center=(w // 2, int(h * 0.60)))
            screen.blit(msg2, rect2)

            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)
        else:
            top_line = "Tippe das Ergebnis. ENTER = prüfen."
            surf_top = font_hint.render(to_nfc(top_line), True, (190, 190, 205))
            rect_top = surf_top.get_rect(center=(w // 2, int(h * 0.10)))
            screen.blit(surf_top, rect_top)

            task_text = f"{problem.a}  ×  {problem.b}  ="
            surf_task = font_task.render(to_nfc(task_text), True, (240, 240, 240))
            rect_task = surf_task.get_rect(center=(w // 2, int(h * 0.40)))
            screen.blit(surf_task, rect_task)

            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback in ("wrong", "reveal"):
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text else " "
            surf_in = font_input.render(to_nfc(shown_input), True, input_color)
            rect_in = surf_in.get_rect(center=(w // 2, int(h * 0.55)))
            screen.blit(surf_in, rect_in)

            if feedback == "wrong":
                surf_msg = font_hint.render(to_nfc("Nochmal"), True, (240, 90, 90))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "correct":
                surf_msg = font_hint.render(to_nfc("Richtig"), True, (80, 220, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "reveal":
                reveal_text = f"Richtig: {problem.a}×{problem.b}={problem.answer}"
                surf_msg = font_hint.render(to_nfc(reveal_text), True, (235, 210, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)

            if hint_text and (feedback in ("wrong", "correct", "reveal")):
                surf_hint = font_input.render(to_nfc(hint_text), True, (170, 170, 185))
                rect_hint = surf_hint.get_rect(center=(w // 2, int(h * 0.72)))
                screen.blit(surf_hint, rect_hint)

            frac = q_index / SESSION_QUESTIONS
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, frac)

            qtxt = f"Frage {q_index + 1}/{SESSION_QUESTIONS}"
            screen.blit(font_small.render(to_nfc(qtxt), True, (160, 160, 170)), (int(w * 0.10), int(h * 0.86)))

        pygame.display.flip()

    finish_session()
    pygame.quit()


if __name__ == "__main__":
    main()
