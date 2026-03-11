import re
import unicodedata
from dataclasses import dataclass, field
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
FPS = 60
SESSION_SENTENCES = 10       # exactly 10 sentences per session
CORRECT_PAUSE_SECONDS = 1.2  # brief pause before moving on after correct
MIN_WORDS = 5                # every sentence must have at least 5 words
MAX_INPUT_CHARS = 200
NORMALIZE_MULTI_SPACES = True

APP_ID = "satz_trainer"

DEFAULT_ACC = 0.58
DEFAULT_RT = 12.0

FOCUS_BOOSTS = {
    "noun_cap": 0.42,
    "punct": 0.38,
    "umlaut_esz": 0.38,
    "sentence_flow": 0.32,
    "verb_end": 0.32,
}

# =========================
# Boy-friendly topics for age 7
# =========================
# Each topic has: a German prompt, an example answer, topic tags,
# and a set of "character names" that appear in the prompt or example
# (names must never be flagged as spelling errors).

TOPICS: List[Dict[str, Any]] = [
    # --- Dinosaurs ---
    {
        "topic": "Dinosaurier",
        "prompt": "Was macht ein Tyrannosaurus Rex?",
        "example": "Der Tyrannosaurus Rex jagt andere Dinosaurier und frisst sie.",
        "subject_keywords": ["tyrannosaurus", "dinosaurier", "rex", "t-rex"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Dinosaurier",
        "prompt": "Wie sieht ein Triceratops aus?",
        "example": "Der Triceratops hat drei Hörner und einen großen Schild am Kopf.",
        "subject_keywords": ["triceratops"],
        "names": [],
        "tags": ["noun_cap", "umlaut_esz", "punct"],
    },
    {
        "topic": "Dinosaurier",
        "prompt": "Warum sind die Dinosaurier ausgestorben?",
        "example": "Die Dinosaurier starben aus, weil ein Meteor die Erde traf.",
        "subject_keywords": ["dinosaurier"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    {
        "topic": "Dinosaurier",
        "prompt": "Was frisst ein Brachiosaurus?",
        "example": "Der Brachiosaurus frisst Blätter von hohen Bäumen.",
        "subject_keywords": ["brachiosaurus"],
        "names": [],
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "punct"],
    },
    # --- Space / Weltraum ---
    {
        "topic": "Weltraum",
        "prompt": "Was macht ein Astronaut im Weltall?",
        "example": "Ein Astronaut schwebt im Weltall und repariert die Raumstation.",
        "subject_keywords": ["astronaut"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Weltraum",
        "prompt": "Wie sieht die Sonne aus?",
        "example": "Die Sonne ist ein riesiger, heißer Feuerball aus Gas.",
        "subject_keywords": ["sonne"],
        "names": [],
        "tags": ["noun_cap", "umlaut_esz", "punct"],
    },
    {
        "topic": "Weltraum",
        "prompt": "Was passiert bei einer Mondlandung?",
        "example": "Bei einer Mondlandung setzt das Raumschiff sanft auf dem Mond auf.",
        "subject_keywords": ["mond", "mondlandung", "raumschiff"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    {
        "topic": "Weltraum",
        "prompt": "Was ist ein Schwarzes Loch?",
        "example": "Ein Schwarzes Loch zieht alles an, was ihm zu nahe kommt.",
        "subject_keywords": ["schwarzes", "loch"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    # --- Minecraft ---
    {
        "topic": "Minecraft",
        "prompt": "Was baut Steve in Minecraft?",
        "example": "Steve baut ein großes Haus aus Holz und Stein.",
        "subject_keywords": ["steve", "haus"],
        "names": ["Steve"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Minecraft",
        "prompt": "Wie besiegt man einen Creeper in Minecraft?",
        "example": "Man muss schnell weglaufen, damit der Creeper nicht explodiert.",
        "subject_keywords": ["creeper"],
        "names": ["Creeper"],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    {
        "topic": "Minecraft",
        "prompt": "Was macht Alex in Minecraft?",
        "example": "Alex sammelt Erze, baut Werkzeuge und kämpft gegen Monster.",
        "subject_keywords": ["alex"],
        "names": ["Alex"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Minecraft",
        "prompt": "Warum braucht man in Minecraft eine Fackel?",
        "example": "Man braucht eine Fackel, damit es in der Höhle hell ist.",
        "subject_keywords": ["fackel"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    # --- Superheroes / Superhelden ---
    {
        "topic": "Superhelden",
        "prompt": "Was kann Spider-Man besonders gut?",
        "example": "Spider-Man kann an Wänden klettern und Netze schießen.",
        "subject_keywords": ["spider-man", "spiderman"],
        "names": ["Spider-Man"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Superhelden",
        "prompt": "Wie hilft Superman den Menschen?",
        "example": "Superman fliegt sehr schnell und rettet Menschen in Gefahr.",
        "subject_keywords": ["superman"],
        "names": ["Superman"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    # --- Fußball ---
    {
        "topic": "Fußball",
        "prompt": "Was macht ein Torwart beim Fußball?",
        "example": "Der Torwart hält den Ball und schützt das Tor.",
        "subject_keywords": ["torwart"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Fußball",
        "prompt": "Wie schießt man ein Tor beim Fußball?",
        "example": "Man schießt den Ball mit dem Fuß ins Netz des gegnerischen Tores.",
        "subject_keywords": ["ball", "tor"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow"],
    },
    # --- Piraten ---
    {
        "topic": "Piraten",
        "prompt": "Was suchen Piraten auf dem Meer?",
        "example": "Piraten suchen auf dem Meer nach Schätzen und Gold.",
        "subject_keywords": ["piraten"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Piraten",
        "prompt": "Was macht der Kapitän Edgar auf seinem Schiff?",
        "example": "Kapitän Edgar steuert das Schiff und gibt den Matrosen Befehle.",
        "subject_keywords": ["kapitän", "edgar", "schiff"],
        "names": ["Edgar"],
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "punct"],
    },
    # --- Ritter / Knights ---
    {
        "topic": "Ritter",
        "prompt": "Was macht ein Ritter in einer Burg?",
        "example": "Ein Ritter trainiert mit dem Schwert und verteidigt die Burg.",
        "subject_keywords": ["ritter"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Ritter",
        "prompt": "Wie kämpft Ritter Edgar gegen den Drachen?",
        "example": "Ritter Edgar greift mutig an und besiegt den Drachen mit seinem Schwert.",
        "subject_keywords": ["ritter", "drachen", "edgar"],
        "names": ["Edgar"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    # --- Raketen / Rockets ---
    {
        "topic": "Raketen",
        "prompt": "Wie startet eine Rakete?",
        "example": "Eine Rakete startet mit einem lauten Knall und fliegt in den Himmel.",
        "subject_keywords": ["rakete"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    # --- Roboter ---
    {
        "topic": "Roboter",
        "prompt": "Was kann ein Roboter alles machen?",
        "example": "Ein Roboter kann sprechen, rechnen und Aufgaben für Menschen erledigen.",
        "subject_keywords": ["roboter"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct", "sentence_flow"],
    },
    {
        "topic": "Roboter",
        "prompt": "Wie hilft Roboter Max den Kindern?",
        "example": "Roboter Max erklärt den Kindern Aufgaben und spielt Spiele mit ihnen.",
        "subject_keywords": ["roboter", "max"],
        "names": ["Max"],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    # --- Vulkane ---
    {
        "topic": "Vulkane",
        "prompt": "Was passiert, wenn ein Vulkan ausbricht?",
        "example": "Wenn ein Vulkan ausbricht, fließt heiße Lava den Berg hinunter.",
        "subject_keywords": ["vulkan", "lava"],
        "names": [],
        "tags": ["noun_cap", "punct", "sentence_flow", "umlaut_esz"],
    },
    # --- Tiere / Animals ---
    {
        "topic": "Tiere",
        "prompt": "Was macht ein Hai im Ozean?",
        "example": "Der Hai schwimmt schnell durch den Ozean und jagt Fische.",
        "subject_keywords": ["hai"],
        "names": [],
        "tags": ["noun_cap", "verb_end", "punct"],
    },
    {
        "topic": "Tiere",
        "prompt": "Wie lebt ein Löwe in der Savanne?",
        "example": "Ein Löwe lebt in der Savanne, jagt Zebras und schläft viel.",
        "subject_keywords": ["löwe"],
        "names": [],
        "tags": ["noun_cap", "umlaut_esz", "verb_end", "punct"],
    },
]

ALLOWED_TAGS = ["noun_cap", "punct", "umlaut_esz", "sentence_flow", "verb_end"]

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüßA-Z\-]+")
GERMAN_VOWELS = set("aeiouyäöü")

# =========================
# Vocabulary
# =========================
COMMON_VERBS = [
    "sein", "haben", "werden", "machen", "gehen", "kommen", "sehen", "hören", "finden", "denken",
    "geben", "nehmen", "bringen", "holen", "lassen", "bleiben", "laufen", "rennen", "springen",
    "fliegen", "singen", "spielen", "lernen", "lesen", "schreiben", "rechnen", "arbeiten",
    "helfen", "retten", "bauen", "suchen", "fragen", "antworten", "zeigen", "sagen", "erzählen",
    "erklären", "verstehen", "üben", "hören", "messen", "testen", "forschen", "beobachten",
    "notieren", "pflanzen", "gießen", "ernten", "organisieren", "teilen", "räumen", "reparieren",
    "tragen", "tanzen", "schwimmen", "klettern", "jagen", "schleichen", "essen", "fressen",
    "kämpfen", "schießen", "bauen", "fliegen", "landen", "starten", "explodieren", "sammeln",
    "erkunden", "graben", "schlagen", "verteidigen", "angreifen", "besiegen", "retten", "steuern",
    "schweben", "leuchten", "brennen", "fließen", "ausbrechen", "zieht", "treffen", "klettern",
    "laufen", "weglaufen", "schützen", "halten", "steigen", "sinken", "tauchen",
]

COMMON_FUNCTION_WORDS = {
    "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "den", "dem", "des",
    "und", "oder", "aber", "denn", "weil", "damit", "wenn", "als", "dann", "danach",
    "später", "heute", "morgen", "im", "in", "am", "auf", "bei", "mit", "nach", "vor",
    "zu", "vom", "zum", "über", "unter", "durch", "für", "ohne", "gegen", "zwischen",
    "aus", "an", "von", "ist", "sind", "war", "waren", "hat", "haben", "wird", "werden",
    "er", "sie", "es", "ihn", "ihm", "ihr", "ihre", "sein", "seine", "seinen", "seinem",
    "wir", "ihr", "euch", "uns", "ich", "du", "nicht", "sehr", "schnell", "langsam",
    "groß", "klein", "stark", "mutig", "riesig", "heiß", "laut", "sanft", "hell", "dunkel",
    "alle", "viele", "andere", "andere", "immer", "oft", "nie", "auch", "noch", "schon",
    "man", "was", "wie", "wo", "wer", "warum", "damit", "neben", "hinter", "nah", "nahe",
    "hoch", "hinunter", "hinauf", "hinein", "heraus", "weiter", "alles", "nichts", "etwas",
    "drei", "zwei", "einen", "zwei", "drei", "vier", "fünf", "sechs",
}

# Names that appear across all topics – never flag as errors
GLOBAL_NAMES: Set[str] = {
    "edgar", "steve", "alex", "max", "lena", "tim", "felix", "ben", "leon", "noah",
    "superman", "spiderman", "creeper", "enderman",
}


# =========================
# Helpers
# =========================
def to_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def normalize_text(s: str) -> str:
    s = to_nfc(s).strip()
    if NORMALIZE_MULTI_SPACES:
        s = " ".join(s.split())
    return s


def extract_words(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(to_nfc(text))]


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


def build_state_from_log(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return build_state_from_events(
        events,
        app_id=APP_ID,
        initial_difficulty=0.18,
        default_acc=DEFAULT_ACC,
        default_rt=DEFAULT_RT,
        tag_window=100,
        rt_good=3.0,
        rt_bad=16.0,
        smooth_old=0.90,
        smooth_new=0.10,
        total_seen_key="total_sentences_seen",
    )


# =========================
# Vocabulary sets (built once)
# =========================
def build_allowed_words(session_names: Set[str]) -> Set[str]:
    allowed: Set[str] = set(COMMON_FUNCTION_WORDS)
    allowed.update(GLOBAL_NAMES)
    allowed.update(n.lower() for n in session_names)
    for tpl in TOPICS:
        for field_name in ("prompt", "example"):
            for w in extract_words(str(tpl[field_name]).lower()):
                allowed.add(w)
        for s in tpl["subject_keywords"]:
            allowed.add(s.lower())
        for nm in tpl.get("names", []):
            allowed.add(nm.lower())
    for v in COMMON_VERBS:
        vl = v.lower()
        allowed.add(vl)
        if vl.endswith("en") and len(vl) > 3:
            stem = vl[:-2]
            allowed.add(stem + "t")
            allowed.add(stem + "st")
            allowed.add(stem + "e")
            allowed.add(stem + "en")
    allowed.update({
        "bin", "bist", "ist", "sind", "seid", "war", "waren",
        "habe", "hast", "hat", "habt", "hatte", "hatten",
        "werde", "wirst", "wird", "wurden",
        "gehe", "gehst", "geht", "ging", "gingen",
        "komme", "kommst", "kommt", "kam", "kamen",
        "laufe", "läufst", "läuft", "lief", "liefen",
        "fliege", "fliegst", "fliegt", "flog", "flogen",
        "helfe", "hilfst", "hilft", "half", "halfen",
        "kämpfe", "kämpfst", "kämpft",
        "greife", "greifst", "greift",
        "schieße", "schießt",
        "esse", "isst", "aß",
        "fresse", "frisst", "fraß",
        "zieht", "zog", "zogen",
        "baut", "baut", "gebaut",
        "sammle", "sammelst", "sammelt",
        "steuert", "erklärt", "schützt",
        "trifft", "traf",
    })
    return allowed


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
        "gehe", "gehst", "geht", "läuft", "fährt", "fliegt",
        "kommt", "hilft", "kämpft", "greift", "schießt", "zieht",
        "isst", "frisst", "trifft", "baut", "sammelt", "steuert",
        "erklärt", "schützt", "rettet", "besiegt",
    })
    return forms


KNOWN_VERB_FORMS = build_known_verb_forms()
IRREGULAR_PLURAL_VERBS = {"sind", "waren", "haben", "werden", "wurden", "hatten"}
IRREGULAR_SINGULAR_VERBS = {"ist", "war", "hat", "wird", "hatte"}


# =========================
# German word check
# =========================
def looks_like_german_word(word: str) -> bool:
    w = word.lower()
    # Strip hyphens (compound words, Spider-Man etc.)
    w = w.replace("-", "")
    if not (2 <= len(w) <= 30):
        return False
    if not any(ch in GERMAN_VOWELS for ch in w):
        return False
    if re.search(r"(.)\1\1\1", w):
        return False
    if len(w) <= 7:
        return True
    common_endings = (
        "en", "er", "e", "n", "t", "st", "ung", "heit", "keit", "lich",
        "isch", "ig", "chen", "lein", "bar", "sam", "schaft", "tion",
        "iert", "ieren", "ismus", "isten",
    )
    return w.endswith(common_endings)


def is_likely_typo(word: str, candidate: str, dist: int) -> bool:
    if dist != 1:
        return False
    lw, lc = len(word), len(candidate)
    if lw <= 4 or lc <= 4:
        if lw == lc + 1 and word[:-1] == candidate and word[-1] == candidate[-1]:
            return True
        if lc == lw + 1 and candidate[:-1] == word and candidate[-1] == word[-1]:
            return True
        return False
    if abs(lw - lc) > 1:
        return False
    return word[:2] == candidate[:2] and word[-2:] == candidate[-2:]


# =========================
# Noun capitalisation check
# =========================
# Objective: Check that apparent nouns in the sentence are capitalised.
# A word is treated as a noun candidate if it is NOT a known lowercase function
# word, not a verb form, not the first word of the sentence, and is currently
# lowercased in the typed text.
def find_uncapitalised_nouns(words: List[str], allowed_lower_words: Set[str]) -> List[int]:
    """Return word indices that look like nouns but are not capitalised."""
    bad: List[int] = []
    for i, w in enumerate(words):
        if i == 0:
            continue  # first word capitalized check handled separately
        wl = w.lower()
        if wl in allowed_lower_words:
            continue  # known lowercase word
        if wl in KNOWN_VERB_FORMS:
            continue
        if not w[0].islower():
            continue  # already capitalised – fine
        # Heuristic: if it's likely a proper noun or German noun (upper-case in the
        # example), flag it. We do this by checking if the original typed version is
        # lowercase and the word passes the German word test.
        if looks_like_german_word(wl) and len(wl) >= 3:
            bad.append(i)
    return bad


# =========================
# Evaluation
# =========================
def evaluate_sentence(
    typed: str,
    topic: Dict[str, Any],
    allowed_words: Set[str],
    session_names: Set[str],
) -> Tuple[bool, Set[int], str]:
    """
    Returns (ok, error_word_indices, error_message).
    Checks: capitalisation at start, end punctuation, min 5 words,
    noun capitalisation, basic spelling.
    Names (per-topic + global + session) are never flagged.
    Error messages always use capitalised noun labels.
    """
    text = normalize_text(typed)
    if not text:
        return (False, set(), "Bitte schreibe einen Satz.")

    words = extract_words(text)
    words_lower = [w.lower() for w in words]
    error_idxs: Set[int] = set()
    issues: List[str] = []

    # Collect all accepted names (lowercase)
    all_names_lower: Set[str] = GLOBAL_NAMES | {n.lower() for n in session_names}
    all_names_lower.update(n.lower() for n in topic.get("names", []))
    all_names_lower.update(s.lower() for s in topic.get("subject_keywords", []))

    # 1. Sentence starts with capital letter
    first_alpha = next((c for c in text if c.isalpha()), "")
    if first_alpha and not first_alpha.isupper():
        if words:
            error_idxs.add(0)
        issues.append("Satz muss mit einem großen Buchstaben beginnen")

    # 2. Ends with punctuation
    if not text.rstrip().endswith((".", "!", "?")):
        if words:
            error_idxs.add(len(words) - 1)
        issues.append("Am Ende fehlt ein Satzzeichen (. ! ?)")

    # 3. Minimum 5 words
    if len(words) < MIN_WORDS:
        issues.append(f"Der Satz braucht mindestens {MIN_WORDS} Wörter (du hast {len(words)})")
        # Return early – not worth checking more on a very short input
        return (False, error_idxs, " | ".join(issues[:2]))

    # 4. Noun capitalisation – check all non-first words
    #    Build a set of words that are legitimately lowercase
    lower_ok: Set[str] = set(COMMON_FUNCTION_WORDS) | KNOWN_VERB_FORMS | all_names_lower
    for i, w in enumerate(words):
        if i == 0:
            continue
        wl = w.lower()
        if wl in lower_ok:
            continue
        if wl in all_names_lower:
            continue
        # If the word is written in lowercase but looks like a noun, flag it
        if w[0].islower() and looks_like_german_word(wl) and len(wl) >= 3:
            error_idxs.add(i)
            # Capitalise the noun label in the error message (fix the reference bug)
            noun_display = wl[0].upper() + wl[1:]
            issues.append(f'Nomen groß schreiben: "{noun_display}"')

    # 5. Spelling check for each word
    for i, w in enumerate(words_lower):
        if i in error_idxs:
            continue  # already flagged
        if w in allowed_words:
            continue
        if w in all_names_lower:
            continue
        # Stripped form for hyphenated compounds
        w_stripped = w.replace("-", "")
        if w_stripped in allowed_words or w_stripped in all_names_lower:
            continue
        if looks_like_german_word(w):
            continue
        # Find nearest known word
        best_dist, best_word = min((levenshtein(w, aw), aw) for aw in allowed_words)
        if is_likely_typo(w, best_word, best_dist):
            error_idxs.add(i)
            issues.append(f'Rechtschreibung: "{words[i]}" -> "{best_word}"')
        else:
            error_idxs.add(i)
            issues.append(f'Unbekanntes Wort: "{words[i]}"')
        if len(issues) >= 3:
            break

    ok = len(issues) == 0
    msg = " | ".join(issues[:2]) if issues else ""
    return (ok, error_idxs, msg)


# =========================
# Session topic picker
# =========================
def pick_topics_for_session() -> List[Dict[str, Any]]:
    """Pick SESSION_SENTENCES topics without repeating the same topic consecutively."""
    import random
    chosen: List[Dict[str, Any]] = []
    last_topic = None
    pool = list(TOPICS)
    random.shuffle(pool)
    while len(chosen) < SESSION_SENTENCES:
        candidates = [t for t in pool if t["topic"] != last_topic]
        if not candidates:
            candidates = pool
        t = random.choice(candidates)
        chosen.append(t)
        last_topic = t["topic"]
        # Cycle pool to avoid running out
        if len(chosen) % len(pool) == 0:
            random.shuffle(pool)
    return chosen


# =========================
# UI helpers
# =========================
# --- Colours (white background theme) ---
COL_BG          = (250, 250, 250)
COL_TEXT        = (15,  15,  20)
COL_SUBTLE      = (90,  90, 100)
COL_CARD_BG     = (235, 237, 242)
COL_CARD_BORDER = (180, 182, 195)
COL_CORRECT     = (20, 140,  60)
COL_WRONG       = (195,  30,  30)
COL_ERROR_WORD  = (200,  30,  30)
COL_GOLD        = (160, 110,  20)
COL_BAR_TRACK   = (205, 207, 215)
COL_BAR_FILL    = (30,  110, 200)
COL_TOPIC_TAG   = (30,  100, 190)
COL_SENTENCE_OK = (20,  130,  55)
COL_SENTENCE_PENDING = (140, 140, 155)


FONT_CACHE: Dict[int, pygame.font.Font] = {}


def pick_unicode_font(size: int) -> pygame.font.Font:
    if size in FONT_CACHE:
        return FONT_CACHE[size]
    for name in ("DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            FONT_CACHE[size] = f
            return f
    fb = pygame.font.SysFont(None, size)
    FONT_CACHE[size] = fb
    return fb


def wrap_text(font: pygame.font.Font, text: str, max_w: int) -> List[str]:
    text = to_nfc(text)
    lines: List[str] = []
    for para in text.split("\n"):
        ws = para.split()
        if not ws:
            lines.append("")
            continue
        line = ws[0]
        for word in ws[1:]:
            trial = f"{line} {word}"
            if font.size(trial)[0] <= max_w:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def render_center(screen: pygame.Surface, font: pygame.font.Font,
                  text: str, y: int, color: Tuple) -> None:
    surf = font.render(to_nfc(text), True, color)
    rect = surf.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surf, rect)


def render_left(screen: pygame.Surface, font: pygame.font.Font,
                text: str, x: int, y: int, color: Tuple) -> None:
    screen.blit(font.render(to_nfc(text), True, color), (x, y))


def render_wrapped(screen: pygame.Surface, font: pygame.font.Font,
                   text: str, x: int, y: int, w: int, color: Tuple,
                   line_gap: int = 5) -> int:
    """Render wrapped text, return bottom y."""
    lines = wrap_text(font, text, w)
    line_h = font.get_linesize() + line_gap
    y_cur = y
    for ln in lines:
        surf = font.render(to_nfc(ln), True, color)
        screen.blit(surf, (x, y_cur))
        y_cur += line_h
    return y_cur


def render_answer_highlighted(
    screen: pygame.Surface, font: pygame.font.Font,
    text: str, x: int, y: int, w: int,
    default_color: Tuple, error_idxs: Set[int],
    line_gap: int = 5,
) -> int:
    """Render typed answer with error words in red. Returns bottom y."""
    words = text.split()
    if not words:
        words = [""]
    right = x + w
    x_cur, y_cur = x, y
    line_h = font.get_linesize() + line_gap
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        tw = font.size(token)[0]
        if x_cur + tw > right and x_cur != x:
            x_cur = x
            y_cur += line_h
        color = COL_ERROR_WORD if i in error_idxs else default_color
        screen.blit(font.render(to_nfc(token), True, color), (x_cur, y_cur))
        x_cur += tw
    return y_cur + line_h


def draw_card(screen: pygame.Surface, rect: pygame.Rect,
              bg: Tuple = COL_CARD_BG, border: Tuple = COL_CARD_BORDER,
              radius: int = 12) -> None:
    pygame.draw.rect(screen, bg, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=radius)


def draw_progress_bar(screen: pygame.Surface, rect: pygame.Rect, frac: float) -> None:
    pygame.draw.rect(screen, COL_BAR_TRACK, rect, border_radius=8)
    inner = rect.inflate(-4, -4)
    fw = int(inner.width * clamp(frac, 0.0, 1.0))
    pygame.draw.rect(screen, COL_BAR_FILL,
                     pygame.Rect(inner.left, inner.top, fw, inner.height),
                     border_radius=6)
    pygame.draw.rect(screen, COL_CARD_BORDER, rect, width=1, border_radius=8)


def fit_font(text: str, base: int, min_sz: int,
             box_w: int, box_h: int, gap: int = 5) -> pygame.font.Font:
    for sz in range(base, min_sz - 1, -1):
        f = pick_unicode_font(sz)
        lines = wrap_text(f, text, box_w - 20)
        if len(lines) * (f.get_linesize() + gap) <= box_h - 16:
            return f
    return pick_unicode_font(min_sz)



# =========================
# End screen
# =========================
def run_end_screen(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    topic_name: str,
    session_sentences: List[Dict[str, Any]],
    fonts: Dict[str, pygame.font.Font],
) -> None:
    """
    Show topic at top-center, then all 10 sentences as a paragraph, left-aligned,
    font sized to fit the full page without overflow.
    """
    w, h = screen.get_size()
    font_h1   = fonts["h1"]
    font_small = fonts["small"]

    pad_x = int(w * 0.10)
    pad_top = int(h * 0.10)
    content_w = w - 2 * pad_x

    sentences = [s["typed"] for s in session_sentences]
    full_text = "  ".join(sentences)   # two spaces between sentences

    # Find the largest font size that fits the paragraph in the available height
    avail_h = h - pad_top - int(h * 0.08)  # leave room for topic header + bottom margin
    topic_h = font_h1.get_linesize() + int(h * 0.04)
    text_h = avail_h - topic_h

    best_font = fonts["small"]
    for sz in range(int(min(w, h) // 5), 10, -1):
        f = pick_unicode_font(sz)
        lines = wrap_text(f, full_text, content_w)
        needed = len(lines) * (f.get_linesize() + 6)
        if needed <= text_h:
            best_font = f
            break

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                running = False

        screen.fill(COL_BG)

        # Topic at top-center
        ts = font_h1.render(to_nfc(topic_name), True, COL_TEXT)
        screen.blit(ts, ts.get_rect(center=(w // 2, int(h * 0.06))))

        # Paragraph of sentences, left-aligned
        lines = wrap_text(best_font, full_text, content_w)
        lh = best_font.get_linesize() + 6
        y = pad_top + topic_h
        for ln in lines:
            screen.blit(best_font.render(to_nfc(ln), True, COL_TEXT), (pad_x, y))
            y += lh

        pygame.display.flip()


# =========================
# Main
# =========================
def main() -> None:
    import random

    events = load_recent_events()
    state  = build_state_from_log(events)

    pygame.init()
    pygame.display.set_caption("Satz-Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    pg_clock = pygame.time.Clock()

    base       = max(22, min(w, h) // 12)
    font_h1    = pick_unicode_font(int(base * 1.1))
    font_body  = pick_unicode_font(int(base * 0.85))
    font_hint  = pick_unicode_font(int(base * 0.40))
    font_small = pick_unicode_font(int(base * 0.34))

    fonts = {"h1": font_h1, "body": font_body, "hint": font_hint, "small": font_small}

    session_id         = f"satz_{int(now_ts())}"
    session_start_ts   = now_ts()
    session_base_diff  = latest_logged_difficulty(events, APP_ID, float(state["difficulty"]))

    session_topics = pick_topics_for_session()
    session_names: Set[str] = set()
    for t in session_topics:
        for nm in t.get("names", []):
            session_names.add(nm)
    allowed_words = build_allowed_words(session_names)

    # Derive a single topic name for the session (most frequent topic or first)
    from collections import Counter
    topic_counts = Counter(t["topic"] for t in session_topics)
    session_topic_name = topic_counts.most_common(1)[0][0]

    q_index        = 0
    solved_count   = 0
    completed      = False

    user_text             = ""
    feedback: Optional[str] = None
    feedback_since        = 0.0
    attempts_for_item     = 0
    item_start            = now_ts()
    error_idxs: Set[int]  = set()
    error_msg             = ""

    session_sentences: List[Dict[str, Any]] = []

    def current_topic() -> Dict[str, Any]:
        return session_topics[min(q_index, SESSION_SENTENCES - 1)]

    def current_difficulty() -> float:
        prog = q_index / max(1, SESSION_SENTENCES - 1)
        return clamp(session_base_diff + 0.15 * prog, 0.0, 1.0)

    def save_sentence(typed: str, ok: bool) -> None:
        topic = current_topic()
        append_event({
            "type": "sentence",
            "app": APP_ID,
            "session_id": session_id,
            "sentence_index": q_index + 1,
            "topic": topic["topic"],
            "prompt": topic["prompt"],
            "typed": typed,
            "ok": ok,
            "ts": float(now_ts()),
        })
        session_sentences.append({
            "topic": topic["topic"],
            "prompt": topic["prompt"],
            "typed": typed,
            "ok": ok,
        })

    def advance() -> None:
        nonlocal q_index, completed, user_text, feedback
        nonlocal attempts_for_item, item_start, error_idxs, error_msg
        q_index += 1
        if q_index >= SESSION_SENTENCES:
            completed = True
            append_event({
                "type": "session_end",
                "app": APP_ID,
                "session_id": session_id,
                "sentences_done": q_index,
                "correct_solved": solved_count,
                "elapsed_sec": float(now_ts() - session_start_ts),
                "difficulty_end": float(current_difficulty()),
            })
            return
        user_text = ""
        feedback = None
        attempts_for_item = 0
        item_start = now_ts()
        error_idxs = set()
        error_msg = ""

    append_event({
        "type": "session_start",
        "app": APP_ID,
        "session_id": session_id,
        "sentences_target": SESSION_SENTENCES,
        "difficulty_start": float(current_difficulty()),
    })

    pad  = int(w * 0.08)
    cw   = w - 2 * pad

    # ======================
    # SCREEN 1 – Intro
    # Show topic centered, instruction, "ENTER to start"
    # ======================
    intro = True
    while intro:
        pg_clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_RETURN:
                    intro = False

        screen.fill(COL_BG)

        # Topic name large, centered
        ts = font_h1.render(to_nfc(session_topic_name), True, COL_TEXT)
        screen.blit(ts, ts.get_rect(center=(w // 2, int(h * 0.35))))

        # Instruction
        inst1 = font_body.render(to_nfc("Schreibe 10 Saetze auf Deutsch."), True, COL_TEXT)
        screen.blit(inst1, inst1.get_rect(center=(w // 2, int(h * 0.52))))

        inst2 = font_hint.render(to_nfc("Jeder Satz braucht mindestens 5 Woerter."), True, COL_SUBTLE)
        screen.blit(inst2, inst2.get_rect(center=(w // 2, int(h * 0.60))))

        start_s = font_hint.render(to_nfc("ENTER druecken zum Starten"), True, COL_SUBTLE)
        screen.blit(start_s, start_s.get_rect(center=(w // 2, int(h * 0.74))))

        pygame.display.flip()

    item_start = now_ts()  # reset timer after intro

    # ======================
    # SCREEN 2 – Writing loop
    # Topic top-center, "Satz N von 10" top-left,
    # typed text large left-aligned in the middle of the page
    # Error message below in red when wrong
    # ======================
    running = True
    while running and not completed:
        pg_clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if feedback == "correct":
                    # swallow keys during pause
                    continue

                if event.key == pygame.K_RETURN:
                    if not user_text.strip():
                        continue
                    attempts_for_item += 1
                    rt = now_ts() - item_start
                    ok, idxs, msg = evaluate_sentence(
                        user_text, current_topic(), allowed_words, session_names)
                    error_idxs = idxs
                    error_msg  = msg

                    if ok:
                        feedback = "correct"
                        feedback_since = now_ts()
                        solved_count += 1
                        state["total_sentences_seen"] = (
                            int(state.get("total_sentences_seen", 0)) + 1)
                        update_tag_stats(state, current_topic()["tags"],
                                         correct=True, rt=rt, tag_window=100)
                        update_overall_difficulty(
                            state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT,
                            rt_good=3.0, rt_bad=16.0,
                            smooth_old=0.90, smooth_new=0.10)
                        save_sentence(user_text, True)
                        append_event({
                            "type": "attempt", "app": APP_ID,
                            "session_id": session_id,
                            "q_index": q_index + 1,
                            "topic": current_topic()["topic"],
                            "typed": user_text, "correct": True,
                            "attempt": attempts_for_item, "rt": float(rt),
                        })
                    else:
                        feedback = "wrong"
                        update_tag_stats(state, current_topic()["tags"],
                                         correct=False, rt=rt, tag_window=100)
                        update_overall_difficulty(
                            state, default_acc=DEFAULT_ACC, default_rt=DEFAULT_RT,
                            rt_good=3.0, rt_bad=16.0,
                            smooth_old=0.90, smooth_new=0.10)
                        append_event({
                            "type": "attempt", "app": APP_ID,
                            "session_id": session_id,
                            "q_index": q_index + 1,
                            "topic": current_topic()["topic"],
                            "typed": user_text, "correct": False,
                            "attempt": attempts_for_item, "rt": float(rt),
                            "error_msg": error_msg,
                        })

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                    if feedback == "wrong":
                        feedback = None
                        error_idxs = set()
                        error_msg = ""
                else:
                    if (len(user_text) < MAX_INPUT_CHARS
                            and event.unicode
                            and event.unicode.isprintable()
                            and event.unicode not in ("\r", "\n")):
                        user_text = to_nfc(user_text + event.unicode)
                    if feedback == "wrong":
                        feedback = None
                        error_idxs = set()
                        error_msg = ""

        # Auto-advance after correct pause
        if feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                advance()
                if completed:
                    break

        # --- Draw writing screen ---
        screen.fill(COL_BG)

        # Topic at top-center
        t_surf = font_h1.render(to_nfc(session_topic_name), True, COL_TEXT)
        screen.blit(t_surf, t_surf.get_rect(center=(w // 2, int(h * 0.06))))

        # Sentence counter top-left
        cnt_text = f"Satz {min(q_index + 1, SESSION_SENTENCES)} von {SESSION_SENTENCES}"
        screen.blit(font_hint.render(to_nfc(cnt_text), True, COL_SUBTLE),
                    (pad, int(h * 0.06) - font_hint.get_linesize() // 2))

        # Typed text – large, left-aligned, vertically centered in remaining space
        shown = user_text + ("_" if feedback != "correct" else "")
        if not shown.strip():
            shown = "_"

        # Pick font size: try large, shrink to fit width
        txt_color = COL_CORRECT if feedback == "correct" else \
                    COL_WRONG   if feedback == "wrong"   else COL_TEXT

        # Use a reasonably large font – fit to content width
        txt_font = fit_font(shown, int(base * 2.2), int(base * 0.7),
                            cw, int(h * 0.55), gap=8)

        if feedback == "wrong" and error_idxs:
            render_answer_highlighted(
                screen, txt_font, user_text,
                pad, int(h * 0.28), cw, COL_TEXT, error_idxs, line_gap=10)
        else:
            render_wrapped(screen, txt_font, shown,
                           pad, int(h * 0.28), cw, txt_color, line_gap=10)

        # Error message below
        if feedback == "wrong" and error_msg:
            parts = error_msg.split(" | ")
            ey = int(h * 0.78)
            for p in parts[:2]:
                err_s = font_hint.render(to_nfc(p), True, COL_WRONG)
                screen.blit(err_s, (pad, ey))
                ey += font_hint.get_linesize() + 4

        pygame.display.flip()

    if not running:
        pygame.quit()
        return

    # ======================
    # SCREEN 3 – End: topic + all sentences as paragraph
    # ======================
    run_end_screen(screen, pg_clock, session_topic_name, session_sentences, fonts)
    pygame.quit()


if __name__ == "__main__":
    main()