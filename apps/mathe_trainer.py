import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pygame

from core.adaptive_core import (
    build_state_from_events,
    clamp,
    latest_logged_difficulty,
    update_overall_difficulty,
    update_tag_stats,
    weighted_pick_tag,
)
from core.trainer_data import append_event, load_recent_events, now_ts

# =========================
# Config
# =========================
# Objective: Define session pacing and adaptive-tuning constants.
FPS = 60
SESSION_QUESTIONS = 30
CORRECT_PAUSE_SECONDS = 1.35

RAMP_MAX_BONUS = 0.20
ALLOW_NEGATIVES = False
TAG_WINDOW = 80
MIN_VALUE = 1
MAX_VALUE = 100

APP_ID = "math"

# Focus boost based on observed mistakes in recent handwritten exercises:
# - carry in addition
# - borrowing in subtraction
# - place-value slips in two-digit mental math
FOCUS_BOOSTS = {
    "add_carry": 0.45,
    "sub_borrow": 0.45,
    "add_two_digit": 0.20,
    "sub_two_digit": 0.20,
}


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


# Objective: Configure default prior estimates for unseen tags.
DEFAULT_ACC = 0.60
DEFAULT_RT = 9.0


# Objective: Rebuild math state from the shared event log.
def build_state_from_log(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return build_state_from_events(
        events,
        app_id=APP_ID,
        initial_difficulty=0.15,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        tag_window=TAG_WINDOW,
        rt_good=3.0,
        rt_bad=12.0,
        smooth_old=0.90,
        smooth_new=0.10,
        total_seen_key="total_questions_seen",
    )


# =========================
# Problem model + tagging
# =========================
# Objective: Represent one generated arithmetic task.
@dataclass
class Problem:
    a: int
    b: int
    op: str
    tags: List[str]

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b


# Objective: Detect whether addition crosses the next ten.
def is_crossing_10_add(a: int, b: int) -> bool:
    return (a % 10) + (b % 10) >= 10


# Objective: Detect carry in ones column for two-digit addition.
def is_carry_add(a: int, b: int) -> bool:
    return a >= 10 and b >= 10 and (a % 10) + (b % 10) >= 10


# Objective: Detect borrow in ones column for two-digit subtraction.
def is_borrow_sub(a: int, b: int) -> bool:
    return a >= 10 and b >= 10 and (a % 10) < (b % 10)


# Objective: Detect multiples of ten for tens-focused practice.
def is_tens(n: int) -> bool:
    return n % 10 == 0


# Objective: Assign granular skill tags to each generated problem.
def assign_tags(a: int, b: int, op: str) -> List[str]:
    tags: List[str] = []
    if op == "+":
        tags.append("add")
        if a <= 10 and b <= 10:
            tags.append("add_small")
        if is_tens(a) and is_tens(b):
            tags += ["tens", "add_tens"]
        if a >= 10 and b >= 10:
            tags.append("add_two_digit")
            tags.append("add_carry" if is_carry_add(a, b) else "add_no_carry")
        if is_crossing_10_add(a, b):
            tags.append("add_cross10")
    else:
        tags.append("sub")
        if a <= 10 and b <= 10:
            tags.append("sub_small")
        if is_tens(a) and is_tens(b):
            tags += ["tens", "sub_tens"]
        if a >= 10 and b >= 10:
            tags.append("sub_two_digit")
            tags.append("sub_borrow" if is_borrow_sub(a, b) else "sub_no_borrow")
    return tags


# =========================
# Generation
# =========================
# Objective: Balance addition vs subtraction by current difficulty.
def choose_op(difficulty: float) -> str:
    p_sub = 0.25 + 0.35 * difficulty
    return "-" if random.random() < p_sub else "+"


# Objective: Select the next weakness tag to practice.
def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    return weighted_pick_tag(
        state,
        allowed_tags,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        rt_good=3.0,
        rt_bad=12.0,
        base_weight=0.20,
        explore_bonus=0.15,
        focus_boosts=FOCUS_BOOSTS,
    )


# Objective: Translate abstract difficulty into numeric operand limits.
def difficulty_to_limits(difficulty: float) -> Dict[str, Any]:
    # Keep all generated values in a stable 1..100 curriculum while scaling complexity.
    if difficulty < 0.20:
        max_small, max_mid, max_two = 10, 20, 35
    elif difficulty < 0.40:
        max_small, max_mid, max_two = 15, 35, 50
    elif difficulty < 0.60:
        max_small, max_mid, max_two = 20, 50, 70
    elif difficulty < 0.80:
        max_small, max_mid, max_two = 20, 70, 90
    else:
        max_small, max_mid, max_two = 20, 85, 100

    return {
        "max_small": max_small,
        "max_mid": max_mid,
        "max_two": max_two,
    }


# Objective: Generate a child-friendly problem aligned to a target tag.
def make_problem_for_target(difficulty: float, target_tag: str) -> Problem:
    limits = difficulty_to_limits(difficulty)
    max_small = limits["max_small"]
    max_mid = limits["max_mid"]
    max_two = limits["max_two"]

    for _ in range(350):
        if target_tag.startswith("add"):
            op = "+"
        elif target_tag.startswith("sub"):
            op = "-"
        else:
            op = choose_op(difficulty)

        if target_tag in ("add_small", "sub_small"):
            a = random.randint(MIN_VALUE, max_small)
            b = random.randint(MIN_VALUE, max_small)

        elif target_tag in ("tens", "add_tens", "sub_tens"):
            choices = list(range(10, min(90, max_two) + 1, 10))
            a, b = random.choice(choices), random.choice(choices)

        elif target_tag == "add_cross10":
            a10 = random.randint(0, max(1, max_mid // 10))
            a1 = random.randint(1, 9)
            a = min(a10 * 10 + a1, max_mid)
            if a % 10 == 0:
                a = max(MIN_VALUE, a - 1)
            b1 = random.randint(max(1, 10 - (a % 10)), 9)
            b10 = random.randint(0, max(0, (max_mid - b1) // 10))
            b = min(b10 * 10 + b1, max_mid)

        elif target_tag in ("add_carry", "add_no_carry", "sub_borrow", "sub_no_borrow", "add_two_digit", "sub_two_digit"):
            a = random.randint(10, max_two)
            b = random.randint(10, max_two)

            if target_tag == "add_carry":
                a1 = random.randint(2, 9)
                b1 = random.randint(max(10 - a1, 1), 9)
                a = random.randint(1, 9) * 10 + a1
                b = random.randint(1, 9) * 10 + b1

            if target_tag == "add_no_carry":
                a1 = random.randint(0, 9)
                b1 = random.randint(0, 9 - a1)
                a = random.randint(1, 9) * 10 + a1
                b = random.randint(1, 9) * 10 + b1

            if target_tag == "sub_borrow":
                a10 = random.randint(2, 9)
                b10 = random.randint(1, a10)
                a1 = random.randint(0, 8)
                b1 = random.randint(a1 + 1, 9)
                a = a10 * 10 + a1
                b = b10 * 10 + b1
                if b > a:
                    a, b = b, a

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
            a = random.randint(MIN_VALUE, max_mid)
            b = random.randint(MIN_VALUE, max_mid)

        if op == "-" and not ALLOW_NEGATIVES and b > a:
            a, b = b, a

        ans = a + b if op == "+" else a - b
        if ans < MIN_VALUE or ans > MAX_VALUE:
            continue

        if a < MIN_VALUE or b < MIN_VALUE or a > MAX_VALUE or b > MAX_VALUE:
            continue

        tags = assign_tags(a, b, op)
        if target_tag not in tags and random.random() > 0.10:
            continue
        return Problem(a=a, b=b, op=op, tags=tags)

    op = choose_op(difficulty)
    a, b = random.randint(MIN_VALUE, max_mid), random.randint(MIN_VALUE, max_mid)
    if op == "-" and b > a:
        a, b = b, a
    if a + b > MAX_VALUE and op == "+":
        b = max(MIN_VALUE, MAX_VALUE - a)
    if a - b < MIN_VALUE and op == "-":
        b = max(MIN_VALUE, a - MIN_VALUE)
    return Problem(a=a, b=b, op=op, tags=assign_tags(a, b, op))


# Objective: Gate which tags are eligible at each difficulty range.
def allowed_tags_for_difficulty(difficulty: float) -> List[str]:
    # Clear difficulty bands so hardness rises consistently.
    if difficulty < 0.20:
        return ["add_small", "sub_small"]
    if difficulty < 0.40:
        return ["add_small", "sub_small", "add_cross10", "tens"]
    if difficulty < 0.60:
        return ["add_cross10", "tens", "add_two_digit", "sub_two_digit", "add_no_carry", "sub_no_borrow"]
    if difficulty < 0.80:
        return ["add_two_digit", "sub_two_digit", "add_no_carry", "sub_no_borrow", "add_carry", "sub_borrow"]
    return ["add_carry", "sub_borrow", "add_two_digit", "sub_two_digit", "add_cross10", "add_no_carry", "sub_no_borrow"]


# =========================
# UI
# =========================
# Objective: Render a consistent progress bar used by the session UI.
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


# Objective: Run one adaptive fullscreen math training session.
def main():
    events = load_recent_events()
    state = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("Math Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pick_unicode_font(int(base * 1.6))
    font_input = pick_unicode_font(int(base * 1.1))
    font_hint = pick_unicode_font(int(base * 0.40))
    font_small = pick_unicode_font(int(base * 0.32))

    session_id = f"math_{int(now_ts())}"
    session_base_difficulty = latest_logged_difficulty(events, APP_ID, float(state["difficulty"]))

    q_index = 0
    completed = False

    user_text = ""
    feedback: Optional[str] = None
    feedback_since = 0.0
    attempts_for_problem = 0
    problem_start = now_ts()
    problem_solved = False

    def current_session_difficulty() -> float:
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        return clamp(session_base_difficulty + (RAMP_MAX_BONUS * prog), 0.0, 1.0)

    def pick_next_problem() -> Problem:
        d = current_session_difficulty()
        target = pick_target_tag(state, allowed_tags_for_difficulty(d))
        return make_problem_for_target(d, target)

    problem = pick_next_problem()

    append_event({
        "type": "session_start",
        "app": APP_ID,
        "session_id": session_id,
        "questions_target": SESSION_QUESTIONS,
        "difficulty_start": float(current_session_difficulty()),
    })

    def log_attempt(correct: bool, typed: str, rt: float):
        append_event({
            "type": "attempt",
            "app": APP_ID,
            "session_id": session_id,
            "q_index": q_index + 1,
            "a": problem.a,
            "b": problem.b,
            "op": problem.op,
            "answer": problem.answer,
            "typed": typed,
            "correct": bool(correct),
            "attempt": attempts_for_problem,
            "rt": float(rt),
            "difficulty": float(current_session_difficulty()),
            "tags": problem.tags,
        })

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

                if completed:
                    continue

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
                        update_tag_stats(state, problem.tags, correct=False, rt=rt, tag_window=TAG_WINDOW)
                        update_overall_difficulty(state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT, rt_good=3.0, rt_bad=12.0, smooth_old=0.90, smooth_new=0.10)
                        continue

                    if val == problem.answer:
                        feedback = "correct"
                        feedback_since = now_ts()
                        log_attempt(True, user_text, rt)

                        if not problem_solved:
                            problem_solved = True
                            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1
                            update_tag_stats(state, problem.tags, correct=True, rt=rt, tag_window=TAG_WINDOW)
                            update_overall_difficulty(state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT, rt_good=3.0, rt_bad=12.0, smooth_old=0.90, smooth_new=0.10)
                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, problem.tags, correct=False, rt=rt, tag_window=TAG_WINDOW)
                        update_overall_difficulty(state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT, rt_good=3.0, rt_bad=12.0, smooth_old=0.90, smooth_new=0.10)

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if len(user_text) < 4:
                        user_text = digits_only_append(user_text, event.key)

        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                q_index += 1
                if q_index >= SESSION_QUESTIONS:
                    completed = True
                    append_event({
                        "type": "session_end",
                        "app": APP_ID,
                        "session_id": session_id,
                        "questions_done": q_index,
                        "difficulty_end": float(state["difficulty"]),
                    })
                else:
                    problem = pick_next_problem()
                    user_text = ""
                    feedback = None
                    attempts_for_problem = 0
                    problem_start = now_ts()
                    problem_solved = False

        screen.fill((10, 10, 14))

        if completed:
            surf = font_task.render(to_nfc("SESSION COMPLETE"), True, (80, 220, 120))
            rect = surf.get_rect(center=(w // 2, h // 2))
            screen.blit(surf, rect)
            msg2 = font_hint.render(to_nfc("Press ESC to exit"), True, (160, 160, 170))
            rect2 = msg2.get_rect(center=(w // 2, int(h * 0.60)))
            screen.blit(msg2, rect2)
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)
        else:
            task_text = f"{problem.a}  {problem.op}  {problem.b}  ="
            surf_task = font_task.render(to_nfc(task_text), True, (240, 240, 240))
            rect_task = surf_task.get_rect(center=(w // 2, int(h * 0.40)))
            screen.blit(surf_task, rect_task)

            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text else " "
            surf_in = font_input.render(to_nfc(shown_input), True, input_color)
            rect_in = surf_in.get_rect(center=(w // 2, int(h * 0.55)))
            screen.blit(surf_in, rect_in)

            if feedback == "wrong":
                surf_msg = font_hint.render(to_nfc("Try again"), True, (240, 90, 90))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "correct":
                surf_msg = font_hint.render(to_nfc("Correct"), True, (80, 220, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.65)))
                screen.blit(surf_msg, rect_msg)

            frac = (q_index / SESSION_QUESTIONS)
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, frac)

            hint = "ESC: exit   ENTER: check   BACKSPACE: delete"
            screen.blit(font_hint.render(to_nfc(hint), True, (160, 160, 170)), (int(w * 0.10), int(h * 0.86)))
            qtxt = f"Question {q_index + 1}/{SESSION_QUESTIONS}"
            screen.blit(font_small.render(to_nfc(qtxt), True, (160, 160, 170)), (int(w * 0.10), int(h * 0.06)))

            dtxt = f"Difficulty {current_session_difficulty():.2f}"
            screen.blit(font_small.render(to_nfc(dtxt), True, (120, 120, 140)), (int(w * 0.10), int(h * 0.09)))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
