import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame

from adaptive_core import (
    build_state_from_events,
    clamp,
    latest_logged_difficulty,
    update_overall_difficulty,
    update_tag_stats,
    weighted_pick_tag,
)
from trainer_data import append_event, load_recent_events, now_ts

# =========================
# Config
# =========================
# Objective: Configure German session timing and adaptive behavior.
FPS = 60
SESSION_QUESTIONS = 40
CORRECT_PAUSE_SECONDS = 0.6
RAMP_MAX_BONUS = 0.20
TAG_WINDOW = 100
MAX_INPUT_CHARS = 160
NORMALIZE_MULTI_SPACES = True

APP_ID = "german"

# Objective: Increase focus on weaknesses visible in handwriting samples.
FOCUS_BOOSTS = {
    "noun_cap": 0.35,
    "verb_end": 0.30,
    "double_consonant": 0.35,
    "cluster_sch_ch": 0.25,
    "punct": 0.25,
    "umlaut_esz": 0.25,
    "sentence_flow": 0.20,
}

# Objective: Keep prior defaults for unseen tags.
DEFAULT_ACC = 0.58
DEFAULT_RT = 10.0

# Objective: Define open questions with semantic anchors, not single fixed sentences.
QUESTION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "question": "Was machen Bienen?",
        "keyword_groups": [["bienen"], ["fliegen", "summen", "sammeln"]],
        "tags": ["noun_cap", "verb_end", "sentence_flow", "punct"],
        "example": "Bienen fliegen und sammeln Nektar.",
    },
    {
        "question": "Was macht ein Hund im Park?",
        "keyword_groups": [["hund"], ["läuft", "springt", "spielt", "rennt"]],
        "tags": ["noun_cap", "verb_end", "double_consonant", "punct"],
        "example": "Ein Hund läuft und spielt im Park.",
    },
    {
        "question": "Was machen Kinder in der Schule?",
        "keyword_groups": [["kinder"], ["lernen", "lesen", "schreiben"]],
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "punct"],
        "example": "Kinder lernen, lesen und schreiben in der Schule.",
    },
    {
        "question": "Wie ist die Straße nach dem Regen?",
        "keyword_groups": [["straße"], ["nass", "glatt", "rutschig"]],
        "tags": ["noun_cap", "umlaut_esz", "punct", "sentence_flow"],
        "example": "Die Straße ist nass und glatt.",
    },
    {
        "question": "Was machen Vögel am Morgen?",
        "keyword_groups": [["vögel"], ["fliegen", "singen"]],
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "punct"],
        "example": "Vögel fliegen und singen am Morgen.",
    },
    {
        "question": "Was macht die Katze in der Nacht?",
        "keyword_groups": [["katze"], ["schleicht", "jagt", "läuft"]],
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "punct"],
        "example": "Die Katze schleicht und jagt in der Nacht.",
    },
]

ALLOWED_TAGS = [
    "noun_cap",
    "verb_end",
    "double_consonant",
    "cluster_sch_ch",
    "umlaut_esz",
    "punct",
    "sentence_flow",
]

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


# =========================
# Helpers
# =========================
# Objective: Normalize unicode into one stable representation.
def to_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Objective: Normalize typed text before grammar/spelling checks.
def normalize_text(s: str) -> str:
    s = to_nfc(s).strip()
    if NORMALIZE_MULTI_SPACES:
        s = " ".join(s.split())
    return s


# Objective: Split text into word tokens and preserve order for highlighting.
def extract_words(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(to_nfc(text))]


# Objective: Compute edit distance to detect likely misspellings of expected keywords.
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


# Objective: Rebuild German adaptive state from shared JSONL attempts.
def build_state_from_log(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return build_state_from_events(
        events,
        app_id=APP_ID,
        initial_difficulty=0.18,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        tag_window=TAG_WINDOW,
        rt_good=3.0,
        rt_bad=14.0,
        smooth_old=0.90,
        smooth_new=0.10,
        total_seen_key="total_questions_seen",
    )


# =========================
# Content model
# =========================
# Objective: Represent one open-question exercise with flexible answer checks.
@dataclass
class WritingItem:
    instruction: str
    prompt: str
    keyword_groups: List[List[str]]
    tags: List[str]
    kind: str
    example: str


# Objective: Select next weakest tag for adaptive targeting.
def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    return weighted_pick_tag(
        state,
        allowed_tags,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        rt_good=3.0,
        rt_bad=14.0,
        base_weight=0.20,
        explore_bonus=0.18,
        focus_boosts=FOCUS_BOOSTS,
    )


# Objective: Build one open composition question aligned to a target tag.
def make_open_question_item(target_tag: str) -> WritingItem:
    targeted = [tpl for tpl in QUESTION_TEMPLATES if target_tag in tpl["tags"]]
    tpl = random.choice(targeted if targeted else QUESTION_TEMPLATES)
    return WritingItem(
        instruction="Schreibe einen ganzen Antwortsatz (freie Form):",
        prompt=f"Frage: {tpl['question']}",
        keyword_groups=tpl["keyword_groups"],
        tags=list(sorted(set(tpl["tags"] + [target_tag]))),
        kind="open_question",
        example=f"Beispiel: {tpl['example']}",
    )


# Objective: Evaluate free-composition answer for grammar, punctuation, and keyword spelling.
def evaluate_free_answer(item: WritingItem, typed: str) -> Tuple[bool, Set[int], str]:
    text = normalize_text(typed)
    if not text:
        return (False, set(), "Bitte schreibe einen Antwortsatz.")

    words = extract_words(text)
    words_lower = [w.lower() for w in words]
    error_word_idxs: Set[int] = set()
    issues: List[str] = []

    # Grammar: sentence start capitalization.
    first_alpha = next((c for c in text if c.isalpha()), "")
    if first_alpha and not first_alpha.isupper():
        if words:
            error_word_idxs.add(0)
        issues.append("Satzanfang groß schreiben")

    # Grammar: sentence-ending punctuation.
    if not text.endswith((".", "!", "?")):
        if words:
            error_word_idxs.add(len(words) - 1)
        issues.append("Satzzeichen am Ende fehlt")

    # Grammar: minimal sentence length.
    if len(words) < 3:
        error_word_idxs.update(range(len(words)))
        issues.append("Antwort ist zu kurz")

    # Semantic + spelling anchors: each group needs at least one word.
    for group in item.keyword_groups:
        group_l = [g.lower() for g in group]
        if any(g in words_lower for g in group_l):
            continue

        best_idx = -1
        best_dist = 99
        best_target = ""
        for i, w in enumerate(words_lower):
            for g in group_l:
                d = levenshtein(w, g)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    best_target = g

        if best_idx >= 0 and best_dist <= 2:
            error_word_idxs.add(best_idx)
            issues.append(f"Rechtschreibung prüfen: '{words[best_idx]}' (nahe bei '{best_target}')")
        else:
            # No near-miss token -> likely missing concept in sentence.
            error_word_idxs.update(range(len(words)))
            issues.append(f"Inhalt ergänzen: eines von {', '.join(group)}")

    # Grammar: ensure at least one finite-looking verb.
    if not any(w.endswith(("t", "en")) for w in words_lower):
        error_word_idxs.update(range(len(words)))
        issues.append("Ein Verb fehlt")

    ok = len(issues) == 0
    message = " | ".join(issues[:2]) if issues else ""
    return (ok, error_word_idxs, message)


# Objective: Keep only open-question mode as requested.
def pick_next_item(state: Dict[str, Any], difficulty: float) -> WritingItem:
    target = pick_target_tag(state, ALLOWED_TAGS)
    return make_open_question_item(target)


# =========================
# UI
# =========================
# Objective: Render shared progress bar style.
def draw_progress_bar(surface, rect: pygame.Rect, frac_0_1: float):
    pygame.draw.rect(surface, (30, 30, 40), rect, border_radius=10)
    inner = rect.inflate(-6, -6)
    fill_w = int(inner.width * clamp(frac_0_1, 0.0, 1.0))
    fill_rect = pygame.Rect(inner.left, inner.top, fill_w, inner.height)
    pygame.draw.rect(surface, (80, 220, 120), fill_rect, border_radius=8)
    pygame.draw.rect(surface, (70, 70, 90), rect, width=2, border_radius=10)


# Objective: Draw one centered text line.
def render_center(screen, font, text, y, color):
    surf = font.render(to_nfc(text), True, color)
    rect = surf.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surf, rect)


# Objective: Wrap text by width while respecting explicit line breaks.
def wrap_text(font, text: str, max_width: int) -> List[str]:
    text = to_nfc(text)
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if font.size(trial)[0] <= max_width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


# Objective: Draw wrapped text inside a bounded rectangle.
def render_wrapped_block(
    screen,
    font,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    color,
    line_gap: int = 8,
):
    lines = wrap_text(font, text, w - 20)
    line_h = font.get_linesize() + line_gap
    max_lines = max(1, h // line_h)
    lines = lines[:max_lines]

    y_cur = y + 10
    for line in lines:
        surf = font.render(to_nfc(line), True, color)
        rect = surf.get_rect(midtop=(x + w // 2, y_cur))
        screen.blit(surf, rect)
        y_cur += line_h


# Objective: Draw typed answer with per-word red highlights at error positions.
def render_answer_block(
    screen,
    font,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    default_color,
    error_word_idxs: Set[int],
    line_gap: int = 8,
):
    words = text.split()
    if not words:
        words = [""]

    left = x + 10
    right = x + w - 10
    y_cur = y + 10
    line_h = font.get_linesize() + line_gap
    max_lines = max(1, h // line_h)

    x_cur = left
    rendered_lines = 0
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        token_w = font.size(token)[0]
        if x_cur + token_w > right:
            rendered_lines += 1
            if rendered_lines >= max_lines:
                break
            x_cur = left
            y_cur += line_h

        color = (240, 90, 90) if i in error_word_idxs else default_color
        surf = font.render(to_nfc(token), True, color)
        screen.blit(surf, (x_cur, y_cur))
        x_cur += token_w


# Objective: Prefer fonts that render umlauts/ß correctly.
def pick_unicode_font(size: int):
    for name in ("DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size)


# Objective: Run one adaptive fullscreen German training session.
def main():
    events = load_recent_events()
    state = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("German Question Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pick_unicode_font(int(base * 0.95))
    font_input = pick_unicode_font(int(base * 0.82))
    font_hint = pick_unicode_font(int(base * 0.34))
    font_small = pick_unicode_font(int(base * 0.30))

    session_id = f"german_{int(now_ts())}"
    session_base_difficulty = latest_logged_difficulty(events, APP_ID, float(state["difficulty"]))

    q_index = 0
    solved_count = 0
    completed = False

    user_text = ""
    feedback: Optional[str] = None
    feedback_since = 0.0
    attempts_for_item = 0
    item_start = now_ts()
    item_solved = False
    error_word_idxs: Set[int] = set()
    error_msg = ""

    # Objective: Increase challenge slightly as session advances.
    def current_session_difficulty() -> float:
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        return clamp(session_base_difficulty + (RAMP_MAX_BONUS * prog), 0.0, 1.0)

    item = pick_next_item(state, current_session_difficulty())

    append_event(
        {
            "type": "session_start",
            "app": APP_ID,
            "session_id": session_id,
            "questions_target": SESSION_QUESTIONS,
            "difficulty_start": float(current_session_difficulty()),
        }
    )

    # Objective: Persist each attempt with diagnostic metadata.
    def log_attempt(correct: bool, typed: str, rt: float):
        append_event(
            {
                "type": "attempt",
                "app": APP_ID,
                "session_id": session_id,
                "q_index": q_index + 1,
                "kind": item.kind,
                "prompt": item.prompt,
                "target": "free_composition",
                "typed": typed,
                "correct": bool(correct),
                "attempt": attempts_for_item,
                "rt": float(rt),
                "difficulty": float(current_session_difficulty()),
                "tags": item.tags,
                "error_message": error_msg,
            }
        )

    # Objective: Reset round state and pick next question.
    def advance_item():
        nonlocal item, user_text, feedback, attempts_for_item, item_start, item_solved, error_word_idxs, error_msg
        item = pick_next_item(state, current_session_difficulty())
        user_text = ""
        feedback = None
        attempts_for_item = 0
        item_start = now_ts()
        item_solved = False
        error_word_idxs = set()
        error_msg = ""

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

                    attempts_for_item += 1
                    rt = now_ts() - item_start

                    ok, err_idxs, msg = evaluate_free_answer(item, user_text)
                    error_word_idxs = err_idxs
                    error_msg = msg

                    if ok:
                        feedback = "correct"
                        feedback_since = now_ts()
                        log_attempt(True, user_text, rt)

                        if not item_solved:
                            item_solved = True
                            solved_count += 1
                            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1
                            update_tag_stats(state, item.tags, correct=True, rt=rt, tag_window=TAG_WINDOW)
                            update_overall_difficulty(
                                state,
                                default_acc=DEFAULT_ACC,
                                default_rt=DEFAULT_RT,
                                rt_good=3.0,
                                rt_bad=14.0,
                                smooth_old=0.90,
                                smooth_new=0.10,
                            )
                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, item.tags, correct=False, rt=rt, tag_window=TAG_WINDOW)
                        update_overall_difficulty(
                            state,
                            default_acc=DEFAULT_ACC,
                            default_rt=DEFAULT_RT,
                            rt_good=3.0,
                            rt_bad=14.0,
                            smooth_old=0.90,
                            smooth_new=0.10,
                        )

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if len(user_text) < MAX_INPUT_CHARS:
                        if event.unicode and event.unicode.isprintable() and event.unicode not in ("\r", "\n"):
                            user_text = to_nfc(user_text + event.unicode)

        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                q_index += 1
                if q_index >= SESSION_QUESTIONS:
                    completed = True
                    append_event(
                        {
                            "type": "session_end",
                            "app": APP_ID,
                            "session_id": session_id,
                            "questions_done": q_index,
                            "correct_solved": solved_count,
                            "difficulty_end": float(state["difficulty"]),
                        }
                    )
                else:
                    advance_item()

        screen.fill((10, 10, 14))

        if completed:
            render_center(screen, font_task, "SESSION COMPLETE", h // 2, (80, 220, 120))
            render_center(screen, font_hint, f"Solved: {solved_count}/{SESSION_QUESTIONS}", int(h * 0.60), (160, 160, 170))
            render_center(screen, font_hint, "ESC", int(h * 0.66), (160, 160, 170))
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)
        else:
            prompt_rect = pygame.Rect(int(w * 0.08), int(h * 0.15), int(w * 0.84), int(h * 0.28))
            input_rect = pygame.Rect(int(w * 0.08), int(h * 0.48), int(w * 0.84), int(h * 0.28))
            feedback_y = input_rect.bottom + int(h * 0.03)

            pygame.draw.rect(screen, (18, 18, 26), prompt_rect, border_radius=12)
            pygame.draw.rect(screen, (45, 45, 62), prompt_rect, width=2, border_radius=12)
            pygame.draw.rect(screen, (18, 18, 26), input_rect, border_radius=12)
            pygame.draw.rect(screen, (45, 45, 62), input_rect, width=2, border_radius=12)

            render_center(screen, font_hint, item.instruction, int(h * 0.10), (190, 190, 205))
            render_wrapped_block(
                screen,
                font_task,
                item.prompt + "\n" + item.example,
                prompt_rect.x,
                prompt_rect.y,
                prompt_rect.width,
                prompt_rect.height,
                (240, 240, 240),
            )

            input_color = (230, 230, 230)
            if feedback == "correct":
                input_color = (80, 220, 120)

            shown_input = user_text if user_text else " "
            render_answer_block(
                screen,
                font_input,
                shown_input,
                input_rect.x,
                input_rect.y,
                input_rect.width,
                input_rect.height,
                input_color,
                error_word_idxs if feedback == "wrong" else set(),
            )

            if feedback == "wrong":
                msg = error_msg if error_msg else "Bitte pruefen"
                render_center(screen, font_hint, msg, feedback_y, (240, 90, 90))
            elif feedback == "correct":
                render_center(screen, font_hint, "Correct", feedback_y, (80, 220, 120))

            frac = q_index / SESSION_QUESTIONS
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, frac)

            hint = "ESC  ENTER  BACKSPACE"
            screen.blit(font_hint.render(hint, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.86)))

            qtxt = f"Question {q_index + 1}/{SESSION_QUESTIONS}"
            screen.blit(font_small.render(qtxt, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.06)))
            dtxt = f"Difficulty {current_session_difficulty():.2f}"
            screen.blit(font_small.render(dtxt, True, (120, 120, 140)), (int(w * 0.10), int(h * 0.09)))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
