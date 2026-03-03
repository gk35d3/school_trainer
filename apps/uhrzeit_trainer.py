import math
import random
import re
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
# Objective: Configure session pacing and adaptive behavior for analog clock reading.
FPS = 60
SESSION_QUESTIONS = 30
SESSION_MAX_SECONDS = 15 * 60
CORRECT_PAUSE_SECONDS = 1.35
REVEAL_PAUSE_SECONDS = 2.60
SLOW_HINT_SECONDS = 7.0
MAX_TRIES_PER_QUESTION = 2

RAMP_MAX_BONUS = 0.16
TAG_WINDOW = 100

APP_ID = "clock"

# Objective: Keep prior defaults for unseen tags.
DEFAULT_ACC = 0.58
DEFAULT_RT = 8.0

# Objective: Define stage order and unlock criteria.
STAGES = ["A", "B", "C", "D", "E"]
STAGE_TAGS = {
    "A": "level_A_hour",
    "B": "level_B_half",
    "C": "level_C_quarter",
    "D": "level_D_five",
    "E": "level_E_minute",
}

# Objective: Stronger focus on typical weak areas.
FOCUS_BOOSTS = {
    "level_A_hour": 0.24,
    "level_B_half": 0.28,
    "level_C_quarter": 0.33,
    "level_D_five": 0.36,
    "level_E_minute": 0.30,
    "full": 0.16,
    "half": 0.20,
    "quarter": 0.25,
    "five_step": 0.28,
    "single_minute": 0.35,
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


# Objective: Rebuild uhrzeit state from the shared event log.
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


# Objective: Render time in normalized H:MM format.
def format_time(hour: int, minute: int) -> str:
    return f"{int(hour)}:{int(minute):02d}"


# Objective: Parse flexible child input into normalized H:MM.
def normalize_time_input(raw: str) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None

    text = text.replace(".", ":")
    text = re.sub(r"\s+", ":", text)

    hour: int
    minute: int

    if ":" in text:
        parts = [p for p in text.split(":") if p != ""]
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        hour = int(parts[0])
        minute = int(parts[1])
    else:
        if not text.isdigit():
            return None
        if len(text) <= 2:
            hour = int(text)
            minute = 0
        elif len(text) == 3:
            hour = int(text[0])
            minute = int(text[1:])
        elif len(text) == 4:
            hour = int(text[:2])
            minute = int(text[2:])
        else:
            return None

    if hour < 1 or hour > 12:
        return None
    if minute < 0 or minute > 59:
        return None

    return format_time(hour, minute)


# Objective: Parse short numeric answer for Q3 understanding questions.
def normalize_number_input(raw: str) -> Optional[int]:
    text = raw.strip()
    if not text or not text.isdigit():
        return None
    n = int(text)
    if n < 0 or n > 59:
        return None
    return n


# Objective: Convert stage id to deterministic minute set.
def minutes_for_stage(stage: str) -> List[int]:
    if stage == "A":
        return [0]
    if stage == "B":
        return [0, 30]
    if stage == "C":
        return [0, 15, 30, 45]
    if stage == "D":
        return list(range(0, 60, 5))
    return list(range(0, 60))


# Objective: Map difficulty to maximum intended stage index.
def stage_index_from_difficulty(difficulty: float) -> int:
    if difficulty < 0.25:
        return 0
    if difficulty < 0.50:
        return 1
    if difficulty < 0.70:
        return 2
    if difficulty < 0.85:
        return 3
    return 4


# Objective: Check whether the learner has mastered one stage recently.
def stage_mastered(state: Dict[str, Any], stage: str) -> bool:
    tag = STAGE_TAGS[stage]
    attempts = state.get("tags", {}).get(tag, {}).get("attempts", [])
    if len(attempts) < 10:
        return False

    recent = attempts[-20:]
    acc = sum(1 for a in recent if a.get("correct")) / max(1, len(recent))
    avg_rt = sum(float(a.get("rt", DEFAULT_RT)) for a in recent) / max(1, len(recent))
    return acc >= 0.85 and avg_rt <= 8.5


# Objective: Gate stage unlocks by difficulty target + mastery requirements.
def max_unlocked_stage_index(state: Dict[str, Any], difficulty: float) -> int:
    target_idx = stage_index_from_difficulty(difficulty)
    unlocked = 0
    while unlocked < target_idx and stage_mastered(state, STAGES[unlocked]):
        unlocked += 1
    return unlocked


# Objective: Build target-tag pool from unlocked stages + observed weak facts.
def allowed_target_tags(state: Dict[str, Any], max_stage_idx: int) -> List[str]:
    tags: Set[str] = {"full", "half", "quarter", "five_step", "single_minute"}
    stages = STAGES[: max_stage_idx + 1]

    for stage in stages:
        tags.add(STAGE_TAGS[stage])
        for minute in minutes_for_stage(stage):
            tags.add(f"m_{minute:02d}")

    weak_fact_candidates = []
    for tag_name, payload in state.get("tags", {}).items():
        if not tag_name.startswith("t_"):
            continue
        attempts = payload.get("attempts", [])
        if len(attempts) < 2:
            continue
        acc, avg_rt, _ = tag_metrics(state, tag_name, DEFAULT_ACC, DEFAULT_RT)
        weakness = (1.0 - acc) * 0.7 + clamp((avg_rt - 3.0) / 9.0, 0.0, 1.0) * 0.3
        weak_fact_candidates.append((weakness, tag_name))

    weak_fact_candidates.sort(reverse=True)
    for _, tag_name in weak_fact_candidates[:20]:
        tags.add(tag_name)

    return sorted(tags)


# =========================
# Problem model + tagging
# =========================
# Objective: Represent one displayed clock question with expected answer metadata.
@dataclass
class Question:
    hour: int
    minute: int
    question_type: str
    prompt: str
    answer_mode: str
    expected_time: Optional[str]
    expected_number: Optional[int]
    stage: str
    tags: List[str]


# Objective: Assign stage + minute/fact/pattern tags for adaptive focus.
def assign_tags(hour: int, minute: int, stage: str) -> List[str]:
    tags = [
        STAGE_TAGS[stage],
        f"h_{hour}",
        f"m_{minute:02d}",
        f"t_{hour}:{minute:02d}",
    ]

    if minute == 0:
        tags.append("full")
    if minute == 30:
        tags.append("half")
    if minute in (15, 45):
        tags.append("quarter")
    if minute % 5 == 0:
        tags.append("five_step")
    else:
        tags.append("single_minute")

    return tags


# =========================
# Generation
# =========================
# Objective: Choose stage with a bias toward current frontier while keeping review.
def choose_stage(max_stage_idx: int) -> str:
    if max_stage_idx <= 0:
        return STAGES[0]

    frontier = STAGES[max_stage_idx]
    if random.random() < 0.70:
        return frontier
    return random.choice(STAGES[: max_stage_idx + 1])


# Objective: Build one Q3 understanding question if current time supports it.
def build_q3(hour: int, minute: int, stage: str, tags: List[str]) -> Question:
    variants: List[Tuple[str, int]] = []
    variants.append(("Wie viele Minuten bis zur vollen Stunde?", (60 - minute) % 60))
    variants.append(("Wie viele Minuten sind vergangen?", minute))
    if minute <= 30:
        variants.append(("Wie viele Minuten bis halb?", 30 - minute))

    prompt, expected = random.choice(variants)
    return Question(
        hour=hour,
        minute=minute,
        question_type="Q3",
        prompt=prompt,
        answer_mode="number",
        expected_time=None,
        expected_number=expected,
        stage=stage,
        tags=tags + ["q3"],
    )


# Objective: Generate next question aligned to the selected adaptive target tag.
def make_question_for_target(
    max_stage_idx: int,
    target_tag: str,
    q_index: int,
    recent_signatures: List[str],
) -> Question:
    q3_turn = ((q_index + 1) % 5 == 0)

    for _ in range(500):
        stage = choose_stage(max_stage_idx)
        minute = random.choice(minutes_for_stage(stage))
        hour = random.randint(1, 12)
        tags = assign_tags(hour, minute, stage)

        if q3_turn:
            q = build_q3(hour, minute, stage, tags)
        else:
            q = Question(
                hour=hour,
                minute=minute,
                question_type="Q1",
                prompt="",
                answer_mode="time",
                expected_time=format_time(hour, minute),
                expected_number=None,
                stage=stage,
                tags=tags,
            )

        if target_tag not in q.tags and random.random() > 0.12:
            continue

        signature = f"{q.question_type}:{q.hour}:{q.minute}:{q.stage}"
        if len(recent_signatures) >= 2 and signature == recent_signatures[-1] == recent_signatures[-2]:
            continue

        return q

    stage = STAGES[max_stage_idx]
    hour = random.randint(1, 12)
    minute = random.choice(minutes_for_stage(stage))
    tags = assign_tags(hour, minute, stage)
    if q3_turn:
        return build_q3(hour, minute, stage, tags)
    return Question(
        hour=hour,
        minute=minute,
        question_type="Q1",
        prompt="",
        answer_mode="time",
        expected_time=format_time(hour, minute),
        expected_number=None,
        stage=stage,
        tags=tags,
    )


# Objective: Choose a tiny hint line based on current pattern.
def micro_hint(question: Question) -> str:
    m = question.minute

    if question.stage == "E" and (m % 5 != 0):
        return "Striche zählen: 1 Minute."
    if m == 30:
        return "Bei 6 = :30 (halb)."
    if m == 15:
        return "Bei 3 = :15 (viertel)."
    if m == 45:
        return "Bei 9 = :45 (dreiviertel)."
    if m % 5 == 0 and m != 0:
        return "Minuten: Zahl × 5."
    if question.question_type == "Q3":
        return "Langer Zeiger = Minuten."
    if m == 0:
        return "Kurzer Zeiger = Stunden."
    return "Stunde liegt dazwischen."


# Objective: Trigger hint on wrong/slow/weak situations.
def should_show_hint(state: Dict[str, Any], tags: List[str], rt: float, was_wrong: bool) -> bool:
    if was_wrong:
        return True
    if rt > SLOW_HINT_SECONDS:
        return True
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag, DEFAULT_ACC, DEFAULT_RT)
        if n >= 2 and (acc < 0.68 or avg_rt > SLOW_HINT_SECONDS):
            return True
    return False


# =========================
# Draw Clock
# =========================
# Objective: Convert clock angle (degrees, 0 at top) to cartesian point.
def polar_point(cx: int, cy: int, radius: float, deg_clock: float) -> Tuple[int, int]:
    rad = math.radians(deg_clock - 90.0)
    x = cx + int(math.cos(rad) * radius)
    y = cy + int(math.sin(rad) * radius)
    return (x, y)


# Objective: Draw one analog 12-hour clock with moving hour hand.
def draw_clock(screen, cx: int, cy: int, radius: int, hour: int, minute: int, font_num):
    pygame.draw.circle(screen, (24, 24, 34), (cx, cy), radius)
    pygame.draw.circle(screen, (90, 90, 120), (cx, cy), radius, width=4)

    for i in range(60):
        deg = i * 6
        outer = polar_point(cx, cy, radius - 4, deg)
        if i % 5 == 0:
            inner = polar_point(cx, cy, radius - 20, deg)
            w = 3
        else:
            inner = polar_point(cx, cy, radius - 12, deg)
            w = 1
        pygame.draw.line(screen, (130, 130, 150), inner, outer, w)

    for n in range(1, 13):
        deg = n * 30
        nx, ny = polar_point(cx, cy, radius - 38, deg)
        txt = font_num.render(to_nfc(str(n)), True, (230, 230, 235))
        rect = txt.get_rect(center=(nx, ny))
        screen.blit(txt, rect)

    minute_deg = (minute / 60.0) * 360.0
    hour_deg = ((hour % 12) / 12.0) * 360.0 + (minute / 60.0) * 30.0

    m_end = polar_point(cx, cy, radius * 0.78, minute_deg)
    h_end = polar_point(cx, cy, radius * 0.52, hour_deg)

    pygame.draw.line(screen, (240, 240, 245), (cx, cy), m_end, 6)
    pygame.draw.line(screen, (200, 205, 220), (cx, cy), h_end, 10)
    pygame.draw.circle(screen, (240, 240, 245), (cx, cy), 8)


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


# Objective: Append typed characters according to active question mode.
def append_input(user_text: str, key: int, unicode_char: str, answer_mode: str) -> str:
    if pygame.K_0 <= key <= pygame.K_9:
        return user_text + chr(key)
    if pygame.K_KP0 <= key <= pygame.K_KP9:
        return user_text + chr(key - pygame.K_KP0 + ord("0"))

    if answer_mode == "time" and unicode_char in (":", ".", " "):
        return user_text + unicode_char

    return user_text


# Objective: Run one adaptive fullscreen analog clock training session.
def main():
    events = load_recent_events()
    state = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("Uhrzeit Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pick_unicode_font(int(base * 0.66))
    font_input = pick_unicode_font(int(base * 1.05))
    font_hint = pick_unicode_font(int(base * 0.38))
    font_small = pick_unicode_font(int(base * 0.32))
    font_clock_num = pick_unicode_font(int(base * 0.44))

    session_id = f"clock_{int(now_ts())}"
    session_start_ts = now_ts()
    session_base_difficulty = latest_logged_difficulty(events, APP_ID, float(state["difficulty"]))

    q_index = 0
    solved_count = 0
    completed = False
    session_end_logged = False

    user_text = ""
    feedback: Optional[str] = None
    feedback_since = 0.0
    attempts_for_question = 0
    question_start = now_ts()
    hint_text = ""
    reveal_text = ""

    recent_signatures: List[str] = []

    # Objective: Increase challenge slightly as the session progresses.
    def current_session_difficulty() -> float:
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        return clamp(session_base_difficulty + (RAMP_MAX_BONUS * prog), 0.0, 1.0)

    # Objective: Select next question by weighted weak-tag targeting.
    def pick_next_question() -> Question:
        d = current_session_difficulty()
        max_stage_idx = max_unlocked_stage_index(state, d)
        tags = allowed_target_tags(state, max_stage_idx)
        target = weighted_pick_tag(
            state,
            tags,
            default_acc=DEFAULT_ACC,
            default_rt=DEFAULT_RT,
            rt_good=3.0,
            rt_bad=12.0,
            base_weight=0.20,
            explore_bonus=0.18,
            focus_boosts=FOCUS_BOOSTS,
        )
        return make_question_for_target(max_stage_idx, target, q_index, recent_signatures)

    question = pick_next_question()

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
    def log_attempt(correct: bool, normalized_input: Optional[str], rt: float):
        append_event(
            {
                "type": "attempt",
                "app": APP_ID,
                "session_id": session_id,
                "q_index": q_index + 1,
                "shown_hour": question.hour,
                "shown_minute": question.minute,
                "question_type": question.question_type,
                "typed_input": user_text,
                "normalized_input": normalized_input,
                "correct": bool(correct),
                "attempt": attempts_for_question,
                "attempts_count": attempts_for_question,
                "reaction_time": float(rt),
                "rt": float(rt),
                "stage": question.stage,
                "difficulty": float(current_session_difficulty()),
                "tags": question.tags,
            }
        )

    # Objective: Update adaptive statistics after each attempt.
    def update_adaptive(correct: bool, rt: float):
        update_tag_stats(state, question.tags, correct=correct, rt=rt, tag_window=TAG_WINDOW)
        update_overall_difficulty(
            state,
            default_acc=DEFAULT_ACC,
            default_rt=DEFAULT_RT,
            rt_good=3.0,
            rt_bad=12.0,
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

    # Objective: Advance to next question and reset per-question UI state.
    def advance_question(solved: bool):
        nonlocal q_index, solved_count, completed, question, user_text, feedback
        nonlocal attempts_for_question, question_start, hint_text, reveal_text

        recent_signatures.append(f"{question.question_type}:{question.hour}:{question.minute}:{question.stage}")
        if len(recent_signatures) > 8:
            recent_signatures[:] = recent_signatures[-8:]

        q_index += 1
        if solved:
            solved_count += 1
            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1

        time_up = (now_ts() - session_start_ts) >= SESSION_MAX_SECONDS
        if q_index >= SESSION_QUESTIONS or time_up:
            completed = True
            finish_session()
            return

        question = pick_next_question()
        user_text = ""
        feedback = None
        attempts_for_question = 0
        question_start = now_ts()
        hint_text = ""
        reveal_text = ""

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
                    continue

                if event.key == pygame.K_RETURN:
                    if user_text.strip() == "":
                        continue

                    attempts_for_question += 1
                    rt = now_ts() - question_start

                    normalized: Optional[str] = None
                    correct = False

                    if question.answer_mode == "time":
                        normalized = normalize_time_input(user_text)
                        correct = normalized is not None and normalized == question.expected_time
                    else:
                        n = normalize_number_input(user_text)
                        normalized = str(n) if n is not None else None
                        correct = n is not None and n == question.expected_number

                    log_attempt(correct, normalized, rt)
                    update_adaptive(correct, rt)

                    if correct:
                        feedback = "correct"
                        feedback_since = now_ts()
                        hint_text = micro_hint(question) if should_show_hint(state, question.tags, rt, False) else ""
                    else:
                        feedback_since = now_ts()
                        hint_text = micro_hint(question)
                        if attempts_for_question >= MAX_TRIES_PER_QUESTION:
                            feedback = "reveal"
                            reveal_text = (
                                f"Richtig: {question.expected_time}"
                                if question.answer_mode == "time"
                                else f"Richtig: {question.expected_number}"
                            )
                        else:
                            feedback = "wrong"
                        user_text = ""

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                    if feedback == "wrong":
                        feedback = None
                else:
                    nxt = append_input(user_text, event.key, event.unicode or "", question.answer_mode)
                    max_len = 5 if question.answer_mode == "time" else 2
                    if len(nxt) <= max_len:
                        user_text = nxt
                        if feedback == "wrong":
                            feedback = None

        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                advance_question(solved=True)

        if not completed and feedback == "reveal":
            if now_ts() - feedback_since >= REVEAL_PAUSE_SECONDS:
                advance_question(solved=False)

        screen.fill((10, 10, 14))

        if completed:
            surf = font_input.render(to_nfc("FERTIG"), True, (80, 220, 120))
            rect = surf.get_rect(center=(w // 2, h // 2))
            screen.blit(surf, rect)

            msg2 = font_hint.render(to_nfc("ESC"), True, (160, 160, 170))
            rect2 = msg2.get_rect(center=(w // 2, int(h * 0.60)))
            screen.blit(msg2, rect2)

            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)
        else:
            top_line = "Lies die Uhr. Tippe die Antwort. ENTER=prüfen."
            surf_top = font_hint.render(to_nfc(top_line), True, (190, 190, 205))
            rect_top = surf_top.get_rect(center=(w // 2, int(h * 0.08)))
            screen.blit(surf_top, rect_top)

            radius = int(min(w, h) * 0.21)
            draw_clock(screen, w // 2, int(h * 0.36), radius, question.hour, question.minute, font_clock_num)

            if question.question_type == "Q3":
                prompt_surf = font_task.render(to_nfc(question.prompt), True, (230, 230, 235))
                prompt_rect = prompt_surf.get_rect(center=(w // 2, int(h * 0.62)))
                screen.blit(prompt_surf, prompt_rect)

            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback in ("wrong", "reveal"):
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text else " "
            surf_in = font_input.render(to_nfc(shown_input), True, input_color)
            rect_in = surf_in.get_rect(center=(w // 2, int(h * 0.73)))
            screen.blit(surf_in, rect_in)

            if feedback == "wrong":
                surf_msg = font_hint.render(to_nfc("Nochmal"), True, (240, 90, 90))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.80)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "correct":
                surf_msg = font_hint.render(to_nfc("Richtig"), True, (80, 220, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.80)))
                screen.blit(surf_msg, rect_msg)
            elif feedback == "reveal":
                surf_msg = font_hint.render(to_nfc(reveal_text), True, (235, 210, 120))
                rect_msg = surf_msg.get_rect(center=(w // 2, int(h * 0.80)))
                screen.blit(surf_msg, rect_msg)

            if hint_text and feedback in ("wrong", "correct", "reveal"):
                surf_hint = font_small.render(to_nfc(hint_text), True, (170, 170, 185))
                rect_hint = surf_hint.get_rect(center=(w // 2, int(h * 0.85)))
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
