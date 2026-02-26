import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

from trainer_data import append_event, load_recent_events, now_ts

# =========================
# Config
# =========================
FPS = 60
SESSION_QUESTIONS = 40
CORRECT_PAUSE_SECONDS = 0.6

WARMUP_DIFFICULTY_OFFSET = 0.06
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
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_text(s: str) -> str:
    s = s.strip()
    if NORMALIZE_MULTI_SPACES:
        s = " ".join(s.split())
    return s


def is_match(typed: str, target: str) -> bool:
    return normalize_text(typed) == normalize_text(target)


def update_tag_stats(state: Dict[str, Any], tags: List[str], correct: bool, rt: float) -> None:
    for tag in tags:
        t = state["tags"].setdefault(tag, {"attempts": []})
        t["attempts"].append({"correct": bool(correct), "rt": float(rt), "ts": now_ts()})
        if len(t["attempts"]) > TAG_WINDOW:
            t["attempts"] = t["attempts"][-TAG_WINDOW:]


def tag_metrics(state: Dict[str, Any], tag: str) -> Tuple[float, float, int]:
    t = state["tags"].get(tag, {}).get("attempts", [])
    n = len(t)
    if n == 0:
        return (0.58, 10.0, 0)
    acc = sum(1 for a in t if a.get("correct")) / n
    avg_rt = sum(a.get("rt", 10.0) for a in t) / n
    return (acc, avg_rt, n)


def update_overall_difficulty(state: Dict[str, Any]) -> None:
    tags = list(state["tags"].keys())
    if not tags:
        return

    scores = []
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        rt_norm = clamp((avg_rt - 3.0) / 11.0, 0.0, 1.0)
        score = (acc * 0.75) + ((1.0 - rt_norm) * 0.25)
        scores.append((score, n ** 0.5))

    total_w = sum(w for _, w in scores) or 1.0
    overall_score = sum(s * w for s, w in scores) / total_w
    target = clamp((overall_score - 0.50) / 0.50, 0.0, 1.0)
    state["difficulty"] = clamp(0.90 * float(state["difficulty"]) + 0.10 * target, 0.0, 1.0)


def build_state_from_log(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    state: Dict[str, Any] = {"difficulty": 0.18, "total_questions_seen": 0, "tags": {}}
    for ev in events:
        if ev.get("app") != APP_ID or ev.get("type") != "attempt":
            continue
        tags = ev.get("tags") or []
        correct = bool(ev.get("correct", False))
        rt = float(ev.get("rt", 10.0))
        update_tag_stats(state, tags, correct, rt)
        if correct:
            state["total_questions_seen"] = int(state["total_questions_seen"]) + 1
    update_overall_difficulty(state)
    return state


# =========================
# Content model
# =========================
@dataclass
class WritingItem:
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


def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    weights = []
    for tag in allowed_tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        rt_norm = clamp((avg_rt - 3.0) / 11.0, 0.0, 1.0)
        weakness = (1.0 - acc) * 0.7 + rt_norm * 0.3
        explore = 0.18 if n == 0 else 0.0
        boost = FOCUS_BOOSTS.get(tag, 0.0)
        w = 0.20 + weakness + explore + boost
        weights.append((tag, w))

    total = sum(w for _, w in weights)
    r = random.random() * total
    upto = 0.0
    for tag, w in weights:
        upto += w
        if upto >= r:
            return tag
    return weights[-1][0]


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


def make_word_item(target_tag: str) -> WritingItem:
    candidates = [(w, t) for (w, t) in WORD_DRILLS if target_tag in t]
    if not candidates:
        candidates = WORD_DRILLS[:]
    word, tags = random.choice(candidates)
    merged = list(sorted(set(tags + [target_tag])))
    return WritingItem(prompt=word, target=word, tags=merged, kind="word")


def make_sentence() -> str:
    pattern = random.randint(1, 5)

    if pattern == 1:
        subj = random.choice(NOUNS)
        verb = random.choice(VERBS_SG)
        prep = random.choice(PREP_PHRASES)
        return f"{subj} {verb} {prep}."

    if pattern == 2:
        subj = random.choice(NOUNS_PL)
        verb = random.choice(VERBS_PL)
        obj = random.choice(OBJECTS)
        return f"{subj} {verb} {obj}."

    if pattern == 3:
        subj = random.choice(NOUNS)
        verb = random.choice(VERBS_SG)
        adj = random.choice(ADJECTIVES)
        return f"{subj} ist {adj}."

    if pattern == 4:
        first = f"{random.choice(NOUNS)} {random.choice(VERBS_SG)}."
        second = f"{random.choice(CONNECTORS).capitalize()} {random.choice(NOUNS_PL)} {random.choice(VERBS_PL)}."
        return f"{first} {second}"

    subj = random.choice(NOUNS_PL)
    verb = random.choice(VERBS_PL)
    prep = random.choice(PREP_PHRASES)
    obj = random.choice(OBJECTS)
    return f"{subj} {verb} {prep} und tragen {obj}."


def make_sentence_item(target_tag: str) -> WritingItem:
    for _ in range(140):
        sentence = make_sentence()
        tags = add_tags_from_text(sentence)
        if target_tag in tags:
            return WritingItem(prompt=sentence, target=sentence, tags=tags, kind="sentence")
        if random.random() < 0.10:
            return WritingItem(prompt=sentence, target=sentence, tags=tags, kind="sentence")

    sentence = make_sentence()
    tags = add_tags_from_text(sentence)
    return WritingItem(prompt=sentence, target=sentence, tags=tags, kind="sentence")


def pick_next_item(state: Dict[str, Any], difficulty: float) -> WritingItem:
    target = pick_target_tag(state, ALLOWED_TAGS)

    # More sentence practice as difficulty grows, but keep word drills for orthography repair.
    p_word = clamp(0.55 - 0.35 * difficulty, 0.18, 0.55)
    if random.random() < p_word:
        return make_word_item(target)
    return make_sentence_item(target)


# =========================
# UI
# =========================
def draw_progress_bar(surface, rect: pygame.Rect, frac_0_1: float):
    pygame.draw.rect(surface, (30, 30, 40), rect, border_radius=10)
    inner = rect.inflate(-6, -6)
    fill_w = int(inner.width * clamp(frac_0_1, 0.0, 1.0))
    fill_rect = pygame.Rect(inner.left, inner.top, fill_w, inner.height)
    pygame.draw.rect(surface, (80, 220, 120), fill_rect, border_radius=8)
    pygame.draw.rect(surface, (70, 70, 90), rect, width=2, border_radius=10)


def render_center(screen, font, text, y, color):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surf, rect)


def main():
    events = load_recent_events()
    state = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("German Writing Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pygame.font.SysFont(None, int(base * 1.2))
    font_input = pygame.font.SysFont(None, int(base * 1.0))
    font_hint = pygame.font.SysFont(None, int(base * 0.40))
    font_small = pygame.font.SysFont(None, int(base * 0.32))

    session_id = f"german_{int(now_ts())}"
    session_base_difficulty = clamp(float(state["difficulty"]) - WARMUP_DIFFICULTY_OFFSET, 0.0, 1.0)

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
                            update_tag_stats(state, item.tags, correct=True, rt=rt)
                            update_overall_difficulty(state)
                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, item.tags, correct=False, rt=rt)
                        update_overall_difficulty(state)

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if len(user_text) < MAX_INPUT_CHARS:
                        if event.unicode and event.unicode.isprintable() and event.unicode not in ("\r", "\n"):
                            user_text += event.unicode

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
            render_center(screen, font_task, item.prompt, int(h * 0.38), (240, 240, 240))

            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text else " "
            render_center(screen, font_input, shown_input, int(h * 0.53), input_color)

            if feedback == "wrong":
                render_center(screen, font_hint, "Try again", int(h * 0.63), (240, 90, 90))
            elif feedback == "correct":
                render_center(screen, font_hint, "Correct", int(h * 0.63), (80, 220, 120))

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
