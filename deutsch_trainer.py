import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
MAX_INPUT_CHARS = 120
NORMALIZE_MULTI_SPACES = True

APP_ID = "german"

# Focus boosts inferred from handwritten text:
# - noun capitalization
# - consonant doubling / cluster spelling
# - verb endings
# - punctuation
# - umlaut / ß handling
FOCUS_BOOSTS = {
    "noun_cap": 0.35,
    "verb_end": 0.30,
    "double_consonant": 0.35,
    "cluster_sch_ch": 0.25,
    "punct": 0.20,
    "umlaut_esz": 0.25,
}


# =========================
# Helpers
# =========================
# Objective: Set prior defaults for unseen German skill tags.
DEFAULT_ACC = 0.58
DEFAULT_RT = 10.0


# Objective: Normalize composed/decomposed unicode into a stable form.
def to_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Objective: Normalize learner input/target text before strict comparison.
def normalize_text(s: str) -> str:
    s = to_nfc(s).strip()
    if NORMALIZE_MULTI_SPACES:
        s = " ".join(s.split())
    return s


# Objective: Perform normalized exact-match checking.
def is_match(typed: str, target: str) -> bool:
    return normalize_text(typed) == normalize_text(target)


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
# Objective: Represent one German exercise item and expected answer.
@dataclass
class WritingItem:
    instruction: str
    prompt: str
    target: str
    tags: List[str]
    kind: str


NOUNS = [
    "der Bagger", "der Hund", "der Zug", "der Baum", "der Vogel", "der Frosch", "der Drache", "der Schnee",
    "die Katze", "die Schule", "die Straße", "die Tasche", "die Lampe", "die Maus", "die Wolke", "die Brücke",
    "das Kind", "das Haus", "das Brot", "das Fenster", "das Pferd", "das Messer", "das Bett", "das Wasser",
]

NOUNS_PL = [
    "die Kinder", "die Hunde", "die Züge", "die Bäume", "die Vögel", "die Frösche", "die Straßen", "die Häuser",
    "die Fenster", "die Mäuse", "die Brücken", "die Wolken", "die Taschen", "die Betten",
]

VERBS_SG = [
    "läuft", "schläft", "schreibt", "liest", "springt", "lacht", "weint", "baut", "sammelt", "klettert", "zeichnet",
]

VERBS_PL = [
    "laufen", "schlafen", "schreiben", "lesen", "springen", "lachen", "weinen", "bauen", "sammeln", "klettern", "zeichnen",
]

ADJECTIVES = [
    "kalt", "warm", "groß", "klein", "leise", "laut", "schnell", "langsam", "fröhlich", "müde", "dunkel", "hell",
]

OBJECTS = [
    "einen Ball", "eine Tasche", "ein Brot", "ein Bild", "einen großen Stein", "ein kleines Haus", "eine warme Suppe",
]

PREP_PHRASES = [
    "im Garten", "in der Schule", "auf der Straße", "im Haus", "am Abend", "in der Nacht", "am Morgen",
]

CONNECTORS = ["und", "aber", "dann", "danach", "plötzlich", "später"]

WORD_DRILLS = [
    ("Schule", ["noun_cap", "cluster_sch_ch"]),
    ("Schiene", ["noun_cap", "cluster_sch_ch", "ie_ei"]),
    ("Straße", ["noun_cap", "umlaut_esz"]),
    ("müde", ["umlaut_esz"]),
    ("später", ["umlaut_esz"]),
    ("rennen", ["double_consonant", "verb_end"]),
    ("kommen", ["double_consonant", "verb_end"]),
    ("schwimmen", ["double_consonant", "cluster_sch_ch", "verb_end"]),
    ("machen", ["cluster_sch_ch", "verb_end"]),
    ("lesen", ["verb_end"]),
    ("Freunde", ["noun_cap"]),
    ("Mädchen", ["noun_cap", "umlaut_esz"]),
    ("Brücke", ["noun_cap", "double_consonant", "umlaut_esz"]),
]

ALLOWED_TAGS = [
    "noun_cap",
    "verb_end",
    "double_consonant",
    "cluster_sch_ch",
    "ie_ei",
    "umlaut_esz",
    "punct",
    "sentence_flow",
]


# Objective: Select the next weakest orthography/grammar tag.
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


# Objective: Infer orthography/grammar tags directly from sentence text.
def add_tags_from_text(sentence: str) -> List[str]:
    tags: List[str] = ["sentence_flow", "punct"]

    words = sentence.replace(".", "").split()
    for i, word in enumerate(words):
        clean = word.strip(".,!?;:")
        if i > 0 and clean and clean[0].isupper():
            tags.append("noun_cap")

        low = clean.lower()
        if any(x in low for x in ("sch", "ch")):
            tags.append("cluster_sch_ch")
        if "ie" in low or "ei" in low:
            tags.append("ie_ei")
        if any(x in low for x in ("ä", "ö", "ü", "ß")):
            tags.append("umlaut_esz")
        if "nn" in low or "mm" in low or "tt" in low:
            tags.append("double_consonant")
        if low.endswith("en") or low.endswith("t"):
            tags.append("verb_end")

    return list(sorted(set(tags)))


# Objective: Build focused single-word spelling tasks.
def make_word_item(target_tag: str) -> WritingItem:
    candidates = [(w, t) for (w, t) in WORD_DRILLS if target_tag in t]
    if not candidates:
        candidates = WORD_DRILLS[:]
    word, tags = random.choice(candidates)
    merged = list(sorted(set(tags + [target_tag])))
    return WritingItem(
        instruction="Schreibe das Wort richtig:",
        prompt=word,
        target=word,
        tags=merged,
        kind="copy_word",
    )


# Objective: Generate broad sentence variants for copy/comprehension tasks.
def make_sentence() -> str:
    pattern = random.randint(1, 5)

    if pattern == 1:
        subj = random.choice(NOUNS)
        verb = random.choice(VERBS_SG)
        prep = random.choice(PREP_PHRASES)
        return capitalize_sentence_start(f"{subj} {verb} {prep}.")

    if pattern == 2:
        subj = random.choice(NOUNS_PL)
        verb = random.choice(VERBS_PL)
        obj = random.choice(OBJECTS)
        return capitalize_sentence_start(f"{subj} {verb} {obj}.")

    if pattern == 3:
        subj = random.choice(NOUNS)
        verb = random.choice(VERBS_SG)
        adj = random.choice(ADJECTIVES)
        return capitalize_sentence_start(f"{subj} ist {adj}.")

    if pattern == 4:
        first = capitalize_sentence_start(f"{random.choice(NOUNS)} {random.choice(VERBS_SG)}.")
        second = f"{random.choice(CONNECTORS).capitalize()} {random.choice(NOUNS_PL)} {random.choice(VERBS_PL)}."
        return f"{first} {second}"

    subj = random.choice(NOUNS_PL)
    verb = random.choice(VERBS_PL)
    prep = random.choice(PREP_PHRASES)
    obj = random.choice(OBJECTS)
    return capitalize_sentence_start(f"{subj} {verb} {prep} und tragen {obj}.")


# Objective: Guarantee sentence-initial capitalization.
def capitalize_sentence_start(s: str) -> str:
    s = to_nfc(s)
    if not s:
        return s
    return s[0].upper() + s[1:]


# Objective: Build sentence-copy tasks aligned to target tags.
def make_sentence_item(target_tag: str) -> WritingItem:
    for _ in range(140):
        sentence = make_sentence()
        tags = add_tags_from_text(sentence)
        if target_tag in tags:
            return WritingItem(
                instruction="Schreibe den Satz korrekt ab:",
                prompt=sentence,
                target=sentence,
                tags=tags,
                kind="copy_sentence",
            )
        if random.random() < 0.10:
            return WritingItem(
                instruction="Schreibe den Satz korrekt ab:",
                prompt=sentence,
                target=sentence,
                tags=tags,
                kind="copy_sentence",
            )

    sentence = make_sentence()
    tags = add_tags_from_text(sentence)
    return WritingItem(
        instruction="Schreibe den Satz korrekt ab:",
        prompt=sentence,
        target=sentence,
        tags=tags,
        kind="copy_sentence",
    )


# Objective: Build unambiguous one-word gap-fill tasks.
def make_fill_blank_item(target_tag: str) -> WritingItem:
    # Curated templates: exactly one missing word with an unambiguous expected answer.
    templates = [
        # noun capitalization
        ("Im Satz fehlt das Nomen. Setze es ein:", "Die ____ spielt im Garten.", "Katze", ["noun_cap"]),
        ("Im Satz fehlt das Nomen. Setze es ein:", "Der ____ fährt schnell.", "Zug", ["noun_cap", "ie_ei"]),
        # verb ending / grammar
        ("Setze die richtige Verbform ein:", "Die Kinder ____ im Park.", "spielen", ["verb_end"]),
        ("Setze die richtige Verbform ein:", "Der Hund ____ im Haus.", "schläft", ["verb_end", "umlaut_esz"]),
        # clusters sch/ch
        ("Setze das fehlende Wort ein:", "Die ____ ist lang.", "Schiene", ["cluster_sch_ch", "noun_cap", "ie_ei"]),
        ("Setze das fehlende Wort ein:", "Wir ____ heute ein Bild.", "zeichnen", ["cluster_sch_ch", "verb_end"]),
        # doubled consonants
        ("Setze das fehlende Wort ein:", "Wir ____ zum Bus.", "rennen", ["double_consonant", "verb_end"]),
        ("Setze das fehlende Wort ein:", "Die Kinder ____ im Wasser.", "schwimmen", ["double_consonant", "cluster_sch_ch", "verb_end"]),
        # umlaut/ß
        ("Setze das fehlende Wort ein:", "Die Straße ist ____ .", "groß", ["umlaut_esz"]),
        ("Setze das fehlende Wort ein:", "Das Mädchen ist ____ .", "müde", ["umlaut_esz"]),
    ]

    targeted = [tpl for tpl in templates if target_tag in tpl[3]]
    pool = targeted if targeted else templates
    instruction, prompt, answer, base_tags = random.choice(pool)
    tags = list(sorted(set(base_tags + [target_tag, "sentence_flow", "punct"])))
    return WritingItem(
        instruction=instruction,
        prompt=prompt,
        target=answer,
        tags=tags,
        kind="fill_blank",
    )


# Objective: Build short comprehension questions from generated text.
def make_question_item(target_tag: str) -> WritingItem:
    subj = random.choice(NOUNS)
    verb = random.choice(VERBS_SG)
    place = random.choice(PREP_PHRASES)
    obj = random.choice(OBJECTS)
    sentence = capitalize_sentence_start(f"{subj} {verb} {place} und findet {obj}.")

    q_type = random.choice(["wer", "wo", "was"])
    if q_type == "wer":
        question = "Frage: Wer handelt im Satz?"
        answer = subj
    elif q_type == "wo":
        question = "Frage: Wo passiert es?"
        answer = place
    else:
        question = "Frage: Was wird gefunden?"
        answer = obj

    prompt = f"{sentence}\n{question}"
    tags = add_tags_from_text(sentence)
    tags = list(sorted(set(tags + [target_tag, "sentence_flow"])))
    return WritingItem(
        instruction="Lies und beantworte die Frage:",
        prompt=prompt,
        target=answer,
        tags=tags,
        kind="question",
    )


# Objective: Mix exercise modes while keeping weakness targeting.
def pick_next_item(state: Dict[str, Any], difficulty: float) -> WritingItem:
    target = pick_target_tag(state, ALLOWED_TAGS)
    p_copy_word = clamp(0.22 - 0.08 * difficulty, 0.10, 0.22)
    p_fill_blank = clamp(0.38 + 0.08 * difficulty, 0.38, 0.50)
    p_question = clamp(0.28 + 0.12 * difficulty, 0.28, 0.44)

    r = random.random()
    if r < p_copy_word:
        return make_word_item(target)
    if r < p_copy_word + p_fill_blank:
        return make_fill_blank_item(target)
    if r < p_copy_word + p_fill_blank + p_question:
        return make_question_item(target)
    return make_sentence_item(target)


# =========================
# UI
# =========================
# Objective: Render a shared progress bar style for German mode.
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


# Objective: Wrap text by visual width while respecting explicit newlines.
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


# Objective: Draw wrapped multi-line text inside a bounded rectangle.
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


# Objective: Pick a font with robust umlaut/ß rendering.
def pick_unicode_font(size: int):
    # Prefer fonts that reliably render German umlauts/ß as single glyphs.
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
    pygame.display.set_caption("German Writing Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pick_unicode_font(int(base * 1.2))
    font_input = pick_unicode_font(int(base * 1.0))
    font_hint = pick_unicode_font(int(base * 0.40))
    font_small = pick_unicode_font(int(base * 0.32))

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

    def current_session_difficulty() -> float:
        prog = q_index / max(1, (SESSION_QUESTIONS - 1))
        return clamp(session_base_difficulty + (RAMP_MAX_BONUS * prog), 0.0, 1.0)

    item = pick_next_item(state, current_session_difficulty())

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
            "kind": item.kind,
            "prompt": item.prompt,
            "target": item.target,
            "typed": typed,
            "correct": bool(correct),
            "attempt": attempts_for_item,
            "rt": float(rt),
            "difficulty": float(current_session_difficulty()),
            "tags": item.tags,
        })

    def advance_item():
        nonlocal item, user_text, feedback, attempts_for_item, item_start, item_solved
        item = pick_next_item(state, current_session_difficulty())
        user_text = ""
        feedback = None
        attempts_for_item = 0
        item_start = now_ts()
        item_solved = False

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

                    ok = is_match(user_text, item.target)
                    if ok:
                        feedback = "correct"
                        feedback_since = now_ts()
                        log_attempt(True, user_text, rt)

                        if not item_solved:
                            item_solved = True
                            solved_count += 1
                            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1
                            update_tag_stats(state, item.tags, correct=True, rt=rt, tag_window=TAG_WINDOW)
                            update_overall_difficulty(state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT, rt_good=3.0, rt_bad=14.0, smooth_old=0.90, smooth_new=0.10)
                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, item.tags, correct=False, rt=rt, tag_window=TAG_WINDOW)
                        update_overall_difficulty(state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT, rt_good=3.0, rt_bad=14.0, smooth_old=0.90, smooth_new=0.10)

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
                    append_event({
                        "type": "session_end",
                        "app": APP_ID,
                        "session_id": session_id,
                        "questions_done": q_index,
                        "correct_solved": solved_count,
                        "difficulty_end": float(state["difficulty"]),
                    })
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
            prompt_rect = pygame.Rect(int(w * 0.08), int(h * 0.16), int(w * 0.84), int(h * 0.30))
            input_rect = pygame.Rect(int(w * 0.08), int(h * 0.52), int(w * 0.84), int(h * 0.24))
            feedback_y = input_rect.bottom + int(h * 0.03)

            pygame.draw.rect(screen, (18, 18, 26), prompt_rect, border_radius=12)
            pygame.draw.rect(screen, (45, 45, 62), prompt_rect, width=2, border_radius=12)
            pygame.draw.rect(screen, (18, 18, 26), input_rect, border_radius=12)
            pygame.draw.rect(screen, (45, 45, 62), input_rect, width=2, border_radius=12)

            render_center(screen, font_hint, item.instruction, int(h * 0.12), (190, 190, 205))

            render_wrapped_block(
                screen,
                font_task,
                item.prompt,
                prompt_rect.x,
                prompt_rect.y,
                prompt_rect.width,
                prompt_rect.height,
                (240, 240, 240),
            )

            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text else " "
            render_wrapped_block(
                screen,
                font_input,
                shown_input,
                input_rect.x,
                input_rect.y,
                input_rect.width,
                input_rect.height,
                input_color,
            )

            if feedback == "wrong":
                render_center(screen, font_hint, "Try again", feedback_y, (240, 90, 90))
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
