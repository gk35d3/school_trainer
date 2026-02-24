import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import pygame

# =========================
# Config (UX like mathetrainer_v2.py)
# =========================
FPS = 60
SESSION_SECONDS = 15 * 60  # 15 minutes
CORRECT_PAUSE_SECONDS = 0.6

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(APP_DIR, "state_writing.json")
SESSIONS_DIR = os.path.join(APP_DIR, "sessions_writing")

TAG_WINDOW = 80  # keep last N attempts per tag in state

# Difficulty ramp inside a session (0..1) -> +0..0.20
RAMP_MAX_BONUS = 0.20

# Warm-up start slightly easier
WARMUP_DIFFICULTY_OFFSET = 0.06

# Strict scoring:
# - exact match including punctuation, capitalization, umlauts, ß
# - BUT normalize multiple spaces to single (kids sometimes double-space)
NORMALIZE_MULTI_SPACES = True

# Limit how many chars kid can type before we block (avoid huge inputs)
MAX_INPUT_CHARS = 80

# =========================
# Persistence helpers
# =========================
def ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def now_ts() -> float:
    return time.time()

def load_state() -> Dict[str, Any]:
    default = {
        "difficulty": 0.18,  # overall 0..1
        "total_items_seen": 0,
        "tags": {
            # tag: {"attempts":[{"correct":bool,"rt":float,"ts":float}]}
        }
    }
    if not os.path.exists(STATE_PATH):
        return default
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
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
        json.dump(state, f, indent=2, ensure_ascii=False)
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
    Compute overall difficulty gently from tag performance.
    """
    tags = list(state["tags"].keys())
    if not tags:
        return

    scores = []
    for tag in tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        # Normalize RT: 3s good, 14s bad
        rt_norm = clamp((avg_rt - 3.0) / (14.0 - 3.0), 0.0, 1.0)
        score = (acc * 0.75) + ((1.0 - rt_norm) * 0.25)
        weight = (n ** 0.5)
        scores.append((score, weight))

    total_w = sum(w for _, w in scores) or 1.0
    overall_score = sum(s * w for s, w in scores) / total_w

    # Map overall_score to difficulty gently
    target = clamp((overall_score - 0.50) / 0.50, 0.0, 1.0)
    state["difficulty"] = clamp(0.90 * float(state["difficulty"]) + 0.10 * target, 0.0, 1.0)

# =========================
# Strict compare
# =========================
def normalize_text(s: str) -> str:
    s = s.strip()
    if NORMALIZE_MULTI_SPACES:
        s = " ".join(s.split())
    return s

def is_strict_match(typed: str, target: str) -> bool:
    return normalize_text(typed) == normalize_text(target)

# =========================
# Content model
# =========================
@dataclass
class WritingItem:
    prompt: str          # what to show (minimal)
    target: str          # what kid must type
    tags: List[str]      # skill tags
    kind: str            # "word" or "sentence"

# A compact, open-ended generator: we combine small word banks into many sentences.
NOUNS = [
    ("der Igel", "m"), ("der Frosch", "m"), ("der Hund", "m"), ("der Bär", "m"),
    ("die Katze", "f"), ("die Maus", "f"), ("die Ente", "f"), ("die Sonne", "f"),
    ("das Kind", "n"), ("das Brot", "n"), ("das Eis", "n"), ("das Boot", "n"),
]
PLURALS = [
    "die Kinder", "die Hasen", "die Frösche", "die Hunde", "die Katzen", "die Enten",
]
VERBS = [
    ("schläft", "sleep"), ("läuft", "move"), ("springt", "move"), ("lacht", "sound"),
    ("lernt", "school"), ("malt", "school"), ("spielt", "play"),
]
VERBS_PL = [
    ("schlafen", "sleep"), ("laufen", "move"), ("springen", "move"), ("lachen", "sound"),
    ("lernen", "school"), ("malen", "school"), ("spielen", "play"),
]
ADJ = [
    ("groß", "esz"), ("klein", "cap"), ("kalt", "punct"), ("warm", "punct"),
    ("dünn", "umlaut"), ("dick", "punct"),
]
OBJECTS = [
    ("einen Ball", "acc"), ("einen Drachen", "acc"), ("einen großen Schneemann", "acc_esz_nn"),
    ("ein Brot", "acc"), ("ein Eis", "acc"),
]

# Words for focused word-level drills (still “open”, not only Schneemann)
WORD_DRILLS = [
    ("groß", ["esz"]),
    ("großen", ["esz"]),
    ("dünn", ["umlaut"]),
    ("schläft", ["umlaut_cap"]),
    ("schlafen", ["umlaut"]),
    ("Kinder", ["cap"]),
    ("Schneemann", ["nn", "cap"]),
]

# =========================
# Difficulty -> how complex items can be
# =========================
def difficulty_to_profile(d: float) -> Dict[str, Any]:
    # As d grows:
    # - more plural sentences
    # - more adjectives
    # - more object phrases (articles)
    # - more umlaut/ß words
    return {
        "p_word_item": clamp(0.55 - 0.35 * d, 0.15, 0.55),
        "p_plural_sentence": clamp(0.20 + 0.35 * d, 0.20, 0.65),
        "p_object": clamp(0.25 + 0.45 * d, 0.25, 0.80),
        "p_adjective": clamp(0.20 + 0.55 * d, 0.20, 0.85),
        "p_umlaut_esz": clamp(0.15 + 0.55 * d, 0.15, 0.80),
    }

# =========================
# Tag selection (like math trainer)
# =========================
ALLOWED_TAGS_BASE = [
    "punct", "cap", "esz", "umlaut", "nn", "acc",
    "mixed_sentence"
]

def pick_target_tag(state: Dict[str, Any], allowed_tags: List[str]) -> str:
    weights = []
    for tag in allowed_tags:
        acc, avg_rt, n = tag_metrics(state, tag)
        rt_norm = clamp((avg_rt - 3.0) / 11.0, 0.0, 1.0)
        weakness = (1.0 - acc) * 0.7 + rt_norm * 0.3
        explore = 0.18 if n == 0 else 0.0
        w = 0.20 + weakness + explore
        weights.append((tag, w))

    total = sum(w for _, w in weights)
    r = random.random() * total
    upto = 0.0
    for tag, w in weights:
        upto += w
        if upto >= r:
            return tag
    return weights[-1][0]

# =========================
# Item generation
# =========================
def make_word_item(target_tag: str) -> WritingItem:
    # choose drill word that matches tag if possible
    candidates = []
    for w, tags in WORD_DRILLS:
        if target_tag in tags or (target_tag == "umlaut" and "umlaut" in tags) or (target_tag == "esz" and "esz" in tags):
            candidates.append((w, tags))
    if not candidates:
        candidates = WORD_DRILLS[:]
    w, tags = random.choice(candidates)
    # Minimal prompt: just show the target? (Yes, but that's okay for word copying at this level)
    return WritingItem(prompt=w, target=w, tags=list(set(tags + [target_tag])), kind="word")

def sentence_from_parts(d: float) -> Tuple[str, List[str]]:
    prof = difficulty_to_profile(d)

    tags: List[str] = []

    # plural or singular
    if random.random() < prof["p_plural_sentence"]:
        subj = random.choice(PLURALS)  # already includes correct article + cap at start (die Kinder)
        verb, vtag = random.choice(VERBS_PL)
        tags += ["mixed_sentence", "cap"]
        if vtag == "sleep":
            tags += ["umlaut"]
        sentence = f"{subj} {verb}."
        tags += ["punct"]
    else:
        subj, gender = random.choice(NOUNS)
        verb, vtag = random.choice(VERBS)
        tags += ["mixed_sentence", "cap", "punct"]
        if vtag == "sleep":
            tags += ["umlaut"]
        sentence = f"{subj} {verb}."

    # optional adjective clause: "Es ist kalt."
    if random.random() < 0.30 + 0.20 * d:
        a, atag = random.choice(ADJ)
        # choose "Es ist ..." sentence
        sentence = f"Es ist {a}."
        tags += ["punct"]
        if atag == "esz":
            tags += ["esz"]
        if atag == "umlaut":
            tags += ["umlaut"]

    # optional object phrase to force accusative/article
    if random.random() < prof["p_object"]:
        obj, objtag = random.choice(OBJECTS)
        tags += ["acc"]
        # create "Wir bauen ..." or "Die Kinder bauen ..."
        starter = "Die Kinder" if random.random() < 0.55 else "Wir"
        tags += ["cap"]
        # add ß / nn tags if object contains them
        if "großen" in obj or "groß" in obj:
            tags += ["esz"]
        if "Schneemann" in obj:
            tags += ["nn", "cap"]
        sentence = f"{starter} bauen {obj}."
        tags += ["punct"]

    # optional capitalization challenge: ensure first letter capital already (it is)
    return sentence, list(sorted(set(tags)))

def make_sentence_item(state: Dict[str, Any], d: float, target_tag: str) -> WritingItem:
    # generate until we include target_tag often enough (but keep variety)
    for _ in range(120):
        s, tags = sentence_from_parts(d)
        if target_tag in tags or target_tag == "mixed_sentence":
            return WritingItem(prompt=s, target=s, tags=tags, kind="sentence")
        # allow some variety
        if random.random() < 0.12:
            return WritingItem(prompt=s, target=s, tags=tags, kind="sentence")

    s, tags = sentence_from_parts(d)
    return WritingItem(prompt=s, target=s, tags=tags, kind="sentence")

def pick_next_item(state: Dict[str, Any], d: float) -> WritingItem:
    prof = difficulty_to_profile(d)
    allowed_tags = ALLOWED_TAGS_BASE[:]
    target_tag = pick_target_tag(state, allowed_tags)

    if random.random() < prof["p_word_item"]:
        # word copying / spelling
        return make_word_item(target_tag)
    else:
        return make_sentence_item(state, d, target_tag)

# =========================
# Audio (simple beeps)
# =========================
def beep_ok():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1200, 120)
        else:
            print("\a", end="")
    except Exception:
        pass

def beep_bad():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(350, 180)
        else:
            print("\a", end="")
    except Exception:
        pass

# =========================
# UI helpers (like mathetrainer)
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

def format_mmss(seconds: int) -> str:
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"

# =========================
# Main app
# =========================
def main():
    ensure_dirs()
    state = load_state()

    session_name = time.strftime("write_%Y%m%d_%H%M%S")
    attempts_path = os.path.join(SESSIONS_DIR, session_name + "_attempts.jsonl")
    items_path = os.path.join(SESSIONS_DIR, session_name + "_items.jsonl")

    attempts_f = open(attempts_path, "a", encoding="utf-8")
    items_f = open(items_path, "a", encoding="utf-8")

    pygame.init()
    pygame.display.set_caption("Writing Trainer (15 min)")

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    clock = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_task = pygame.font.SysFont(None, int(base * 1.4))
    font_input = pygame.font.SysFont(None, int(base * 1.0))
    font_hint = pygame.font.SysFont(None, int(base * 0.40))
    font_small = pygame.font.SysFont(None, int(base * 0.32))

    session_start = now_ts()
    session_end = session_start + SESSION_SECONDS

    # Warmup: slightly easier
    session_base_difficulty = clamp(float(state["difficulty"]) - WARMUP_DIFFICULTY_OFFSET, 0.0, 1.0)

    # counters
    solved_count = 0
    attempts_total = 0
    completed = False

    # current item state
    user_text = ""
    feedback: Optional[str] = None  # "correct"/"wrong"
    feedback_since = 0.0
    attempts_for_item = 0
    item_start = now_ts()
    item_solved = False

    # session difficulty with ramp
    def session_progress() -> float:
        return clamp((now_ts() - session_start) / SESSION_SECONDS, 0.0, 1.0)

    def current_session_difficulty() -> float:
        ramp = RAMP_MAX_BONUS * session_progress()
        return clamp(session_base_difficulty + ramp, 0.0, 1.0)

    item = pick_next_item(state, current_session_difficulty())

    def log_attempt(correct: bool, typed: str, rt: float):
        rec = {
            "t": now_ts(),
            "kind": item.kind,
            "prompt": item.prompt,
            "target": item.target,
            "typed": typed,
            "correct": bool(correct),
            "attempt": attempts_for_item,
            "rt": float(rt),
            "session_difficulty": float(current_session_difficulty()),
            "overall_difficulty": float(state["difficulty"]),
            "tags": item.tags,
            "strict": True
        }
        attempts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        attempts_f.flush()

    def log_item_summary(final_correct: bool, total_attempts: int, total_time: float):
        rec = {
            "t": now_ts(),
            "kind": item.kind,
            "prompt": item.prompt,
            "target": item.target,
            "final_correct": bool(final_correct),
            "attempts": int(total_attempts),
            "time_total": float(total_time),
            "session_difficulty": float(current_session_difficulty()),
            "tags": item.tags
        }
        items_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        items_f.flush()

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

        # session time
        remaining = int(session_end - now_ts())
        if remaining <= 0 and not completed:
            completed = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if completed:
                    continue

                # during correct pause: ignore typing
                if feedback == "correct":
                    continue

                if event.key == pygame.K_RETURN:
                    if user_text.strip() == "":
                        continue

                    attempts_for_item += 1
                    attempts_total += 1
                    rt = now_ts() - item_start

                    ok = is_strict_match(user_text, item.target)

                    if ok:
                        feedback = "correct"
                        feedback_since = now_ts()
                        log_attempt(True, user_text, rt)

                        if not item_solved:
                            item_solved = True
                            solved_count += 1
                            update_tag_stats(state, item.tags, correct=True, rt=rt)
                            state["total_items_seen"] = int(state.get("total_items_seen", 0)) + 1
                            update_overall_difficulty(state)
                            save_state(state)

                        beep_ok()
                    else:
                        feedback = "wrong"
                        feedback_since = now_ts()
                        log_attempt(False, user_text, rt)
                        update_tag_stats(state, item.tags, correct=False, rt=rt)
                        update_overall_difficulty(state)
                        save_state(state)
                        beep_bad()

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    # accept printable chars; keep it minimal and safe
                    if len(user_text) < MAX_INPUT_CHARS:
                        if event.unicode and event.unicode.isprintable():
                            # avoid newlines
                            if event.unicode not in ("\r", "\n"):
                                user_text += event.unicode

        # After correct pause: advance
        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                total_time = now_ts() - item_start
                log_item_summary(final_correct=True, total_attempts=attempts_for_item, total_time=total_time)
                advance_item()

        # ----------------------------
        # DRAW (minimal words)
        # ----------------------------
        screen.fill((10, 10, 14))

        if completed:
            # Session complete
            render_center(screen, font_task, "DONE", h // 2, (80, 220, 120))
            render_center(screen, font_hint, "ESC", int(h * 0.60), (160, 160, 170))

            # show tiny summary (minimal)
            summary = f"{solved_count} correct"
            render_center(screen, font_hint, summary, int(h * 0.67), (160, 160, 170))

            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, 1.0)

        else:
            # Top tiny timer
            timer_text = format_mmss(max(0, remaining))
            screen.blit(font_small.render(timer_text, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.06)))

            # Minimal instruction: show ONLY the target line (prompt) + input line
            # Prompt
            render_center(screen, font_task, item.prompt, int(h * 0.40), (240, 240, 240))

            # Input color
            if feedback == "correct":
                input_color = (80, 220, 120)
            elif feedback == "wrong":
                input_color = (240, 90, 90)
            else:
                input_color = (230, 230, 230)

            shown_input = user_text if user_text != "" else " "
            render_center(screen, font_input, shown_input, int(h * 0.55), input_color)

            # feedback word (minimal)
            if feedback == "wrong":
                render_center(screen, font_hint, "Try again", int(h * 0.65), (240, 90, 90))
            elif feedback == "correct":
                render_center(screen, font_hint, "Correct", int(h * 0.65), (80, 220, 120))

            # Bottom progress bar: time progress
            frac = session_progress()
            bar_rect = pygame.Rect(int(w * 0.10), int(h * 0.90), int(w * 0.80), int(h * 0.05))
            draw_progress_bar(screen, bar_rect, frac)

            # ultra-small footer
            hint = "ESC  ENTER  BACKSPACE"
            screen.blit(font_hint.render(hint, True, (160, 160, 170)), (int(w * 0.10), int(h * 0.86)))

        pygame.display.flip()

    attempts_f.close()
    items_f.close()
    pygame.quit()


if __name__ == "__main__":
    main()