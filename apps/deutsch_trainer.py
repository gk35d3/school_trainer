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
    update_overall_difficulty,
    update_tag_stats,
    weighted_pick_tag,
)
from core.trainer_data import append_event, load_recent_events, now_ts

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
        "subject_keywords": ["bienen"],
        "subject_number": "plural",
        "subject_gender": "f",
        "tags": ["noun_cap", "verb_end", "sentence_flow", "punct"],
        "example": "Bienen fliegen und sammeln Nektar.",
        "level": 1,
    },
    {
        "question": "Was macht ein Hund im Park?",
        "subject_keywords": ["hund"],
        "subject_number": "singular",
        "subject_gender": "m",
        "tags": ["noun_cap", "verb_end", "double_consonant", "punct"],
        "example": "Ein Hund läuft und spielt im Park.",
        "level": 1,
    },
    {
        "question": "Was machen Kinder in der Schule?",
        "subject_keywords": ["kinder"],
        "subject_number": "plural",
        "subject_gender": "n",
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "punct"],
        "example": "Kinder lernen, lesen und schreiben in der Schule.",
        "level": 2,
    },
    {
        "question": "Wie ist die Straße nach dem Regen?",
        "subject_keywords": ["straße"],
        "subject_number": "singular",
        "subject_gender": "f",
        "tags": ["noun_cap", "umlaut_esz", "punct", "sentence_flow"],
        "example": "Die Straße ist nass und glatt.",
        "level": 2,
    },
    {
        "question": "Was machen Vögel am Morgen?",
        "subject_keywords": ["vögel"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "punct"],
        "example": "Vögel fliegen und singen am Morgen.",
        "level": 2,
    },
    {
        "question": "Was macht die Katze in der Nacht?",
        "subject_keywords": ["katze"],
        "subject_number": "singular",
        "subject_gender": "f",
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "punct"],
        "example": "Die Katze schleicht und jagt in der Nacht.",
        "level": 2,
    },
    {
        "question": "Was machen Feuerwehrleute bei einem Einsatz?",
        "subject_keywords": ["feuerwehrleute"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "double_consonant", "verb_end", "sentence_flow", "punct"],
        "example": "Feuerwehrleute fahren schnell, löschen das Feuer und retten Menschen.",
        "level": 3,
    },
    {
        "question": "Was machen Forscher im Labor?",
        "subject_keywords": ["forscher"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "sentence_flow", "punct"],
        "example": "Forscher untersuchen Proben, messen Werte und notieren Ergebnisse.",
        "level": 3,
    },
    {
        "question": "Was machen Bauern auf dem Feld?",
        "subject_keywords": ["bauern"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "sentence_flow", "punct"],
        "example": "Bauern pflanzen Gemüse, gießen die Felder und ernten später.",
        "level": 3,
    },
    {
        "question": "Was machen Musiker vor einem Auftritt?",
        "subject_keywords": ["musiker"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "sentence_flow", "punct"],
        "example": "Musiker üben zusammen, stimmen ihre Instrumente und hören aufeinander.",
        "level": 4,
    },
    {
        "question": "Was machen Astronauten in einer Raumstation?",
        "subject_keywords": ["astronauten"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "cluster_sch_ch", "verb_end", "sentence_flow", "punct"],
        "example": "Astronauten forschen, reparieren Geräte und beobachten die Erde.",
        "level": 4,
    },
    {
        "question": "Wie helfen Nachbarn einander nach einem Sturm?",
        "subject_keywords": ["nachbarn"],
        "subject_number": "plural",
        "subject_gender": "m",
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "sentence_flow", "punct"],
        "example": "Nachbarn helfen einander, räumen Wege frei und teilen Essen.",
        "level": 5,
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

# Objective: Keep spelling checks broad (not restricted to fixed answer verbs).
COMMON_VERBS = [
    "sein", "haben", "werden", "machen", "gehen", "kommen", "sehen", "hören", "finden", "denken",
    "geben", "nehmen", "bringen", "holen", "lassen", "bleiben", "laufen", "rennen", "springen", "fliegen",
    "singen", "spielen", "lernen", "lesen", "schreiben", "rechnen", "arbeiten", "helfen", "retten", "löschen",
    "bauen", "suchen", "fragen", "antworten", "zeigen", "sagen", "erzählen", "erklären", "verstehen", "üben",
    "stimmen", "hören", "messen", "testen", "forschen", "beobachten", "notieren", "pflanzen", "gießen", "ernten",
    "organisieren", "teilen", "räumen", "reparieren", "tragen", "tanzen", "schwimmen", "klettern", "jagen", "schleichen",
]

COMMON_FUNCTION_WORDS = {
    "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "den", "dem", "des",
    "und", "oder", "aber", "denn", "weil", "dann", "danach", "später", "heute",
    "im", "in", "am", "auf", "bei", "mit", "nach", "vor", "zu", "vom", "zum",
    "ist", "sind", "war", "waren", "hat", "haben", "wird", "werden",
    "er", "sie", "es", "ihn", "ihm", "ihr", "ihre", "sein", "seine", "seinen", "seinem",
    "wir", "ihr", "euch", "uns", "ich", "du", "nicht", "sehr", "schnell", "langsam",
    "nass", "glatt", "rutschig", "frei", "zusammen", "einander",
}


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
    subject_keywords: List[str]
    subject_number: str
    subject_gender: str
    tags: List[str]
    kind: str
    example: str
    min_words: int
    min_verbs: int
    level: int


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
def make_open_question_item(target_tag: str, difficulty: float) -> WritingItem:
    # Difficulty bands map to broader content + stricter writing requirements.
    if difficulty < 0.20:
        min_level = 1
        max_level = 1
        min_words = 3
        min_verbs = 1
    elif difficulty < 0.40:
        min_level = 1
        max_level = 2
        min_words = 6
        min_verbs = 1
    elif difficulty < 0.60:
        min_level = 2
        max_level = 3
        min_words = 8
        min_verbs = 2
    elif difficulty < 0.80:
        min_level = 3
        max_level = 4
        min_words = 10
        min_verbs = 2
    else:
        min_level = 4
        max_level = 5
        min_words = 12
        min_verbs = 3

    candidates = [tpl for tpl in QUESTION_TEMPLATES if min_level <= tpl.get("level", 1) <= max_level]
    targeted = [tpl for tpl in candidates if target_tag in tpl["tags"]]
    tpl = random.choice(targeted if targeted else candidates)
    instruction = "Schreibe einen ganzen Antwortsatz (freie Form):"

    return WritingItem(
        instruction=instruction,
        prompt=f"Frage: {tpl['question']}",
        subject_keywords=tpl["subject_keywords"],
        subject_number=str(tpl["subject_number"]),
        subject_gender=str(tpl["subject_gender"]),
        tags=list(sorted(set(tpl["tags"] + [target_tag]))),
        kind="open_question",
        example=f"Beispiel: {tpl['example']}",
        min_words=min_words,
        min_verbs=min_verbs,
        level=int(tpl.get("level", 1)),
    )


# Objective: Build a broad, internal German lexicon for strict spelling validation.
def build_allowed_words() -> Set[str]:
    allowed: Set[str] = set(COMMON_FUNCTION_WORDS)
    for tpl in QUESTION_TEMPLATES:
        for field in ("question", "example"):
            for w in extract_words(str(tpl[field]).lower()):
                allowed.add(w)
        for s in tpl["subject_keywords"]:
            allowed.add(s.lower())
    for v in COMMON_VERBS:
        vl = v.lower()
        allowed.add(vl)
        if vl.endswith("en") and len(vl) > 3:
            stem = vl[:-2]
            allowed.add(stem + "t")    # er/sie/es form
            allowed.add(stem + "st")   # du form
            allowed.add(stem + "e")    # ich form
    # Irregular/high-frequency forms.
    allowed.update({
        "bin", "bist", "ist", "sind", "seid", "war", "waren",
        "habe", "hast", "hat", "habt", "hatte", "hatten",
        "werde", "wirst", "wird", "wurden",
        "gehe", "gehst", "geht", "ging", "gingen",
        "komme", "kommst", "kommt", "kam", "kamen",
        "laufe", "läufst", "läuft", "lief", "liefen",
        "renne", "rennst", "rennt", "rannte", "rannten",
        "fliege", "fliegst", "fliegt", "flog", "flogen",
        "lese", "liest", "las", "lasen",
        "schreibe", "schreibst", "schreibt", "schrieb", "schrieben",
        "fahre", "fährst", "fährt", "fuhr", "fuhren",
        "trage", "trägst", "trägt", "trug", "trugen",
        "helfe", "hilfst", "hilft", "half", "halfen",
    })
    return allowed


# Objective: Build a normalized set of known finite and infinitive verb forms.
def build_known_verb_forms() -> Set[str]:
    forms: Set[str] = set()
    for v in COMMON_VERBS:
        vl = v.lower()
        forms.add(vl)
        if vl.endswith("en") and len(vl) > 3:
            stem = vl[:-2]
            forms.add(stem + "t")
            forms.add(stem + "st")
            forms.add(stem + "e")
    forms.update({
        "bin", "bist", "ist", "sind", "seid", "war", "waren",
        "habe", "hast", "hat", "habt", "hatte", "hatten",
        "werde", "wirst", "wird", "wurden",
        "gehe", "gehst", "geht", "ging", "gingen",
        "komme", "kommst", "kommt", "kam", "kamen",
        "laufe", "läufst", "läuft", "lief", "liefen",
        "renne", "rennst", "rennt", "rannte", "rannten",
        "fliege", "fliegst", "fliegt", "flog", "flogen",
        "lese", "liest", "las", "lasen",
        "schreibe", "schreibst", "schreibt", "schrieb", "schrieben",
        "fahre", "fährst", "fährt", "fuhr", "fuhren",
        "trage", "trägst", "trägt", "trug", "trugen",
        "helfe", "hilfst", "hilft", "half", "halfen",
    })
    return forms


ALLOWED_WORDS = build_allowed_words()
KNOWN_VERB_FORMS = build_known_verb_forms()


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
    if len(words) < item.min_words:
        error_word_idxs.update(range(len(words)))
        issues.append(f"Antwort ist zu kurz (mindestens {item.min_words} Woerter)")

    # Content anchor: subject noun from the question must appear.
    subject_l = [s.lower() for s in item.subject_keywords]
    if not any(s in words_lower for s in subject_l):
        best_idx = -1
        best_dist = 99
        best_target = ""
        for i, w in enumerate(words_lower):
            for s in subject_l:
                d = levenshtein(w, s)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    best_target = s
        if best_idx >= 0 and best_dist <= 2:
            error_word_idxs.add(best_idx)
            issues.append(f"Subjekt-Rechtschreibung prüfen: '{words[best_idx]}' (nahe bei '{best_target}')")
        else:
            error_word_idxs.update(range(len(words)))
            issues.append(f"Subjekt fehlt: {item.subject_keywords[0]}")

    # Grammar: ensure enough finite-looking verbs for the current level.
    verb_like = [w for w in words_lower if w in KNOWN_VERB_FORMS]
    if len(verb_like) < item.min_verbs:
        error_word_idxs.update(range(len(words)))
        issues.append(f"Zu wenige Verben (mindestens {item.min_verbs})")

    # Grammar: pronoun consistency with subject gender/number.
    pronoun_idxs = [i for i, w in enumerate(words_lower) if w in {"er", "sie", "es"}]
    allowed_pronouns = {"sie"} if item.subject_number == "plural" else {
        "m": {"er"},
        "f": {"sie"},
        "n": {"es"},
    }.get(item.subject_gender, {"er", "sie", "es"})
    for i in pronoun_idxs:
        if words_lower[i] not in allowed_pronouns:
            error_word_idxs.add(i)
            issues.append("Pronomen passt nicht zu Genus/Anzahl")
            break

    # Grammar: singular/plural agreement for the first detected verb form.
    subject_forms = set(subject_l)
    verb_idxs = [i for i, w in enumerate(words_lower) if w in KNOWN_VERB_FORMS and w not in subject_forms]
    if verb_idxs:
        v = words_lower[verb_idxs[0]]
        if item.subject_number == "plural" and not v.endswith("en"):
            error_word_idxs.add(verb_idxs[0])
            issues.append("Verbform passt nicht zum Plural")
        if item.subject_number == "singular" and v.endswith("en"):
            error_word_idxs.add(verb_idxs[0])
            issues.append("Verbform passt nicht zum Singular")

    # Spelling: every word must be valid or a very-close typo.
    for i, w in enumerate(words_lower):
        if w in ALLOWED_WORDS:
            continue
        best = min((levenshtein(w, aw), aw) for aw in ALLOWED_WORDS)
        if best[0] <= 1:
            error_word_idxs.add(i)
            issues.append(f"Rechtschreibung: '{words[i]}' -> '{best[1]}'")
        else:
            error_word_idxs.add(i)
            issues.append(f"Unbekannt/fehlerhaft: '{words[i]}'")
        if len(issues) >= 3:
            break

    ok = len(issues) == 0
    message = " | ".join(issues[:2]) if issues else ""
    return (ok, error_word_idxs, message)


# Objective: Keep only open-question mode as requested.
def pick_next_item(state: Dict[str, Any], difficulty: float) -> WritingItem:
    target = pick_target_tag(state, ALLOWED_TAGS)
    return make_open_question_item(target, difficulty)


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


# Objective: Pick the largest font size that fits wrapped text into a box.
def fit_font_for_box(text: str, base_size: int, min_size: int, box_w: int, box_h: int, line_gap: int = 8):
    for size in range(base_size, min_size - 1, -1):
        f = pick_unicode_font(size)
        lines = wrap_text(f, text, box_w - 20)
        line_h = f.get_linesize() + line_gap
        if len(lines) * line_h <= box_h:
            return f
    return pick_unicode_font(min_size)


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
    task_font_base = int(base * 0.95)
    input_font_base = int(base * 0.82)
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
            done_font = pick_unicode_font(task_font_base)
            render_center(screen, done_font, "SESSION COMPLETE", h // 2, (80, 220, 120))
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
            prompt_text = item.prompt + "\n" + item.example
            task_font = fit_font_for_box(prompt_text, task_font_base, 14, prompt_rect.width, prompt_rect.height, line_gap=6)
            render_wrapped_block(
                screen,
                task_font,
                prompt_text,
                prompt_rect.x,
                prompt_rect.y,
                prompt_rect.width,
                prompt_rect.height,
                (240, 240, 240),
                line_gap=6,
            )

            input_color = (230, 230, 230)
            if feedback == "correct":
                input_color = (80, 220, 120)

            shown_input = user_text if user_text else " "
            input_font = fit_font_for_box(shown_input, input_font_base, 12, input_rect.width, input_rect.height, line_gap=6)
            render_answer_block(
                screen,
                input_font,
                shown_input,
                input_rect.x,
                input_rect.y,
                input_rect.width,
                input_rect.height,
                input_color,
                error_word_idxs if feedback == "wrong" else set(),
                line_gap=6,
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
