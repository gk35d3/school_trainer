"""
Deutsch Trainer v4
==================
Flow per word:
  1. Word/sentence appears on screen
  2. Child reads it silently, then HOLDS SPACE to record → releases to check
  3. Whisper transcribes → lenient fuzzy match → Green / Red
  4. Parent can override with Y (correct) or N (wrong) after wrong result
  5. Child writes the answer on paper (no typing in app during session)
  6. At end of session: spelling review screen shows all answers so parent
     can check the paper

New in v4:
  - SPACE hold-to-record (no auto-listen, no ENTER flow)
  - End-of-session spelling review screen
  - Steeper / faster difficulty ramp (L2 unlocks at 0.15, L3 at 0.40)
  - Lenient voice matching: Levenshtein threshold 3 + phonetic normalisation
  - All v3 features: faster-whisper, spaced repetition w/ decay, phoneme tags,
    blocked warmup, cross-session trophy screen, JSONL session logs

Install:
    pip install pygame faster-whisper sounddevice numpy
"""

import json
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import sounddevice as sd

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    print("[WARN] faster-whisper not installed. Voice recognition disabled.")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FPS                      = 60
SESSION_DURATION_SECONDS = 18 * 60
WARMUP_DURATION_SECONDS  = 4  * 60   # first 4 min: drill weakest phoneme tag
CORRECT_PAUSE_SECONDS    = 0.7
WARMUP_DIFFICULTY_OFFSET = 0.05      # start slightly easier
RAMP_MAX_BONUS           = 0.40      # steeper ramp: +0.40 over the session
TAG_WINDOW               = 80
SR_DECAY_HALFLIFE        = 7 * 24 * 3600   # 7-day spaced repetition

WHISPER_MODEL_SIZE = "small"         # "base" = faster but less accurate
SAMPLE_RATE        = 16_000
LEV_THRESHOLD      = 3               # lenient: allow up to 3 char edits
SENTENCE_MATCH_MISS= 2               # allow up to N words missing in sentence

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
STATE_PATH   = os.path.join(APP_DIR, "deutsch_state.json")
SESSIONS_DIR = os.path.join(APP_DIR, "deutsch_sessions")

# ══════════════════════════════════════════════════════════════════════
# COLOURS
# ══════════════════════════════════════════════════════════════════════
BG       = (10,  10,  14)
WHITE    = (240, 240, 240)
GREEN    = (80,  220, 120)
RED      = (240,  90,  90)
YELLOW   = (255, 210,  60)
ORANGE   = (255, 160,  40)
BLUE     = (80,  160, 240)
PURPLE   = (180, 100, 240)
DIM      = (160, 160, 170)
VDIM     = (100, 100, 120)
BAR_BG   = (30,   30,  40)
BAR_BORD = (70,   70,  90)

# ══════════════════════════════════════════════════════════════════════
# TASK DATABASE
# Every task has: prompt, answer, mode ("voice"|"voice_parent"), tags, emoji_text
# mode "voice"        → Whisper auto-check
# mode "voice_parent" → Whisper tries first, then parent Y/N
#
# Difficulty gates (MUCH steeper than v3):
#   0.00 → l1 only
#   0.15 → l2 unlocks  (previously 0.35)
#   0.40 → l3 + sentences  (previously 0.65)
# ══════════════════════════════════════════════════════════════════════
TASKS: List[Dict] = [

    # ── L1 · simple words ───────────────────────────────────────────
    {"prompt":"Mama",       "answer":"Mama",       "mode":"voice","tags":["l1","ma"],           "emoji":"👩"},
    {"prompt":"Papa",       "answer":"Papa",       "mode":"voice","tags":["l1","pa"],           "emoji":"👨"},
    {"prompt":"Hund",       "answer":"Hund",       "mode":"voice","tags":["l1","nd"],           "emoji":"🐶"},
    {"prompt":"Katze",      "answer":"Katze",      "mode":"voice","tags":["l1","tz"],           "emoji":"🐱"},
    {"prompt":"Haus",       "answer":"Haus",       "mode":"voice","tags":["l1","au"],           "emoji":"🏠"},
    {"prompt":"Ball",       "answer":"Ball",       "mode":"voice","tags":["l1","ll"],           "emoji":"⚽"},
    {"prompt":"Baum",       "answer":"Baum",       "mode":"voice","tags":["l1","au"],           "emoji":"🌳"},
    {"prompt":"Buch",       "answer":"Buch",       "mode":"voice","tags":["l1","ch"],           "emoji":"📖"},
    {"prompt":"Auto",       "answer":"Auto",       "mode":"voice","tags":["l1","au"],           "emoji":"🚗"},
    {"prompt":"Milch",      "answer":"Milch",      "mode":"voice","tags":["l1","ch"],           "emoji":"🥛"},
    {"prompt":"Brot",       "answer":"Brot",       "mode":"voice","tags":["l1","br"],           "emoji":"🍞"},
    {"prompt":"Apfel",      "answer":"Apfel",      "mode":"voice","tags":["l1","pf"],           "emoji":"🍎"},
    {"prompt":"Fisch",      "answer":"Fisch",      "mode":"voice","tags":["l1","sch"],          "emoji":"🐟"},
    {"prompt":"Mond",       "answer":"Mond",       "mode":"voice","tags":["l1","nd"],           "emoji":"🌙"},
    {"prompt":"Sonne",      "answer":"Sonne",      "mode":"voice","tags":["l1","nn"],           "emoji":"☀️"},
    {"prompt":"Nase",       "answer":"Nase",       "mode":"voice","tags":["l1","na"],           "emoji":"👃"},
    {"prompt":"Hand",       "answer":"Hand",       "mode":"voice","tags":["l1","nd"],           "emoji":"✋"},
    {"prompt":"Maus",       "answer":"Maus",       "mode":"voice","tags":["l1","au"],           "emoji":"🐭"},
    {"prompt":"Tisch",      "answer":"Tisch",      "mode":"voice","tags":["l1","sch"],          "emoji":"🪑"},
    {"prompt":"Stern",      "answer":"Stern",      "mode":"voice","tags":["l1","st"],           "emoji":"⭐"},
    {"prompt":"Wald",       "answer":"Wald",       "mode":"voice","tags":["l1","ld"],           "emoji":"🌲"},
    {"prompt":"Berg",       "answer":"Berg",       "mode":"voice","tags":["l1","rg"],           "emoji":"⛰️"},
    {"prompt":"Kind",       "answer":"Kind",       "mode":"voice","tags":["l1","nd"],           "emoji":"🧒"},

    # ── L2 · tricky phonics ──────────────────────────────────────────
    {"prompt":"Eisen",       "answer":"Eisen",       "mode":"voice","tags":["l2","ei"],          "emoji":"🔩"},
    {"prompt":"Feuer",       "answer":"Feuer",       "mode":"voice","tags":["l2","eu"],          "emoji":"🔥"},
    {"prompt":"Biene",       "answer":"Biene",       "mode":"voice","tags":["l2","ie"],          "emoji":"🐝"},
    {"prompt":"Leiter",      "answer":"Leiter",      "mode":"voice","tags":["l2","ei"],          "emoji":"🪜"},
    {"prompt":"Stiefel",     "answer":"Stiefel",     "mode":"voice","tags":["l2","ie","st"],     "emoji":"🥾"},
    {"prompt":"Schule",      "answer":"Schule",      "mode":"voice","tags":["l2","sch"],         "emoji":"🏫"},
    {"prompt":"Schaukel",    "answer":"Schaukel",    "mode":"voice","tags":["l2","sch","au"],    "emoji":"🎠"},
    {"prompt":"Fleisch",     "answer":"Fleisch",     "mode":"voice","tags":["l2","ei","sch"],    "emoji":"🥩"},
    {"prompt":"Freund",      "answer":"Freund",      "mode":"voice","tags":["l2","eu"],          "emoji":"🤝"},
    {"prompt":"Scheune",     "answer":"Scheune",     "mode":"voice","tags":["l2","eu","sch"],    "emoji":"🏚️"},
    {"prompt":"Spielplatz",  "answer":"Spielplatz",  "mode":"voice","tags":["l2","ie","sp"],     "emoji":"🛝"},
    {"prompt":"Schnee",      "answer":"Schnee",      "mode":"voice","tags":["l2","sch","ee"],    "emoji":"❄️"},
    {"prompt":"Blume",       "answer":"Blume",       "mode":"voice","tags":["l2","bl"],          "emoji":"🌸"},
    {"prompt":"Pflanze",     "answer":"Pflanze",     "mode":"voice","tags":["l2","pf"],          "emoji":"🌿"},
    {"prompt":"Straße",      "answer":"Straße",      "mode":"voice","tags":["l2","st"],          "emoji":"🛣️"},
    {"prompt":"Schmetterling","answer":"Schmetterling","mode":"voice","tags":["l2","sch"],        "emoji":"🦋"},
    {"prompt":"Erdbeere",    "answer":"Erdbeere",    "mode":"voice","tags":["l2","ee"],          "emoji":"🍓"},
    {"prompt":"Regenbogen",  "answer":"Regenbogen",  "mode":"voice","tags":["l2"],              "emoji":"🌈"},
    {"prompt":"Geschenk",    "answer":"Geschenk",    "mode":"voice","tags":["l2","sch"],         "emoji":"🎁"},
    {"prompt":"Eichhörnchen","answer":"Eichhörnchen","mode":"voice","tags":["l2","ei","ch"],     "emoji":"🐿️"},
    {"prompt":"Häuschen",    "answer":"Häuschen",    "mode":"voice","tags":["l2","au","sch","ae"],"emoji":"🏡"},
    {"prompt":"Mäuse",       "answer":"Mäuse",       "mode":"voice","tags":["l2","eu","ae"],     "emoji":"🐭"},
    {"prompt":"Frühstück",   "answer":"Frühstück",   "mode":"voice","tags":["l2","ue","sch"],    "emoji":"🥐"},
    {"prompt":"Geburtstag",  "answer":"Geburtstag",  "mode":"voice","tags":["l2","st"],          "emoji":"🎂"},
    {"prompt":"Flugzeug",    "answer":"Flugzeug",    "mode":"voice","tags":["l2","eu"],          "emoji":"✈️"},
    {"prompt":"Schornstein", "answer":"Schornstein", "mode":"voice","tags":["l2","sch","st","ei"],"emoji":"🏭"},
    {"prompt":"Krankenwagen","answer":"Krankenwagen","mode":"voice","tags":["l2"],              "emoji":"🚑"},

    # ── L3 · simple sentences (parent-confirmed) ─────────────────────
    {"prompt":"Die Sonne scheint.",           "answer":"Die Sonne scheint.",           "mode":"voice_parent","tags":["l3","satz","sch"],        "emoji":"☀️"},
    {"prompt":"Der Hund bellt laut.",         "answer":"Der Hund bellt laut.",         "mode":"voice_parent","tags":["l3","satz"],              "emoji":"🐶"},
    {"prompt":"Ich mag Eis.",                 "answer":"Ich mag Eis.",                 "mode":"voice_parent","tags":["l3","satz"],              "emoji":"🍦"},
    {"prompt":"Das Auto ist rot.",            "answer":"Das Auto ist rot.",            "mode":"voice_parent","tags":["l3","satz","au"],         "emoji":"🚗"},
    {"prompt":"Die Katze schläft.",           "answer":"Die Katze schläft.",           "mode":"voice_parent","tags":["l3","satz","sch"],        "emoji":"🐱"},
    {"prompt":"Ich spiele im Garten.",        "answer":"Ich spiele im Garten.",        "mode":"voice_parent","tags":["l3","satz","ie","sp"],    "emoji":"🌿"},
    {"prompt":"Der Schnee ist weiß.",         "answer":"Der Schnee ist weiß.",         "mode":"voice_parent","tags":["l3","satz","sch","ei"],   "emoji":"❄️"},
    {"prompt":"Wir fahren mit dem Auto.",     "answer":"Wir fahren mit dem Auto.",     "mode":"voice_parent","tags":["l3","satz","au"],         "emoji":"🚗"},
    {"prompt":"Die Biene fliegt zur Blume.",  "answer":"Die Biene fliegt zur Blume.",  "mode":"voice_parent","tags":["l3","satz","ie","bl"],    "emoji":"🐝"},
    {"prompt":"Mein Freund heißt Leo.",       "answer":"Mein Freund heißt Leo.",       "mode":"voice_parent","tags":["l3","satz","eu","ei"],    "emoji":"👦"},
    {"prompt":"Der Stern leuchtet hell.",     "answer":"Der Stern leuchtet hell.",     "mode":"voice_parent","tags":["l3","satz","st","eu"],    "emoji":"⭐"},
    {"prompt":"Das Eichhörnchen klettert.",   "answer":"Das Eichhörnchen klettert.",   "mode":"voice_parent","tags":["l3","satz","ei","ch"],    "emoji":"🐿️"},
    {"prompt":"Heute scheint die Sonne schön.","answer":"Heute scheint die Sonne schön.","mode":"voice_parent","tags":["l3","satz","sch","eu"], "emoji":"☀️"},
    {"prompt":"Im Frühling blühen Blumen.",   "answer":"Im Frühling blühen Blumen.",   "mode":"voice_parent","tags":["l3","satz","ue","bl"],    "emoji":"🌸"},
]

# ── Difficulty gates: STEEPER than v3 ─────────────────────────────
DIFFICULTY_GATES = [
    (0.00, {"l1","au","ch","nd","nn","tz","pf","ll","br","ma","pa","na","st","sch","ld","rg"}),
    (0.15, {"l1","l2","au","ch","nd","nn","tz","pf","ll","br","ma","pa","na",
            "st","sch","ld","rg","ei","ie","eu","ee","bl","sp","ae","ue"}),
    (0.40, {"l1","l2","l3","satz","au","ch","nd","nn","tz","pf","ll","br","ma","pa","na",
            "st","sch","ld","rg","ei","ie","eu","ee","bl","sp","ae","ue"}),
]

def tasks_for_difficulty(difficulty: float) -> List[Dict]:
    ok = set()
    for threshold, tags in DIFFICULTY_GATES:
        if difficulty >= threshold:
            ok = tags
    return [t for t in TASKS if any(tag in ok for tag in t["tags"])]

# ══════════════════════════════════════════════════════════════════════
# LEVENSHTEIN
# ══════════════════════════════════════════════════════════════════════
def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
        prev = curr
    return prev[-1]

# ══════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def now_ts() -> float:
    return time.time()

def load_state() -> Dict[str, Any]:
    default = {
        "difficulty": 0.10,
        "total_questions_seen": 0,
        "total_stars": 0,
        "sessions_count": 0,
        "tags": {}
    }
    if not os.path.exists(STATE_PATH):
        return default
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            if k not in data: data[k] = v
        return data
    except Exception:
        return default

def save_state(state: Dict[str, Any]):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)

# ══════════════════════════════════════════════════════════════════════
# SPACED REPETITION + TAG METRICS
# ══════════════════════════════════════════════════════════════════════
def decayed_weakness(attempts: List[Dict], now: float) -> float:
    if not attempts: return 0.55   # unknown → treat as moderately weak
    score = weight_sum = 0.0
    for a in attempts:
        age = max(0, now - a.get("ts", now))
        w = math.exp(-age / SR_DECAY_HALFLIFE)
        score      += w * (1.0 if a.get("correct") else 0.0)
        weight_sum += w
    if weight_sum == 0: return 0.55
    return 1.0 - (score / weight_sum)

def update_tag_stats(state: Dict, tags: List[str], correct: bool, rt: float):
    for tag in tags:
        t = state["tags"].setdefault(tag, {"attempts": []})
        t["attempts"].append({"correct": bool(correct), "rt": float(rt), "ts": now_ts()})
        if len(t["attempts"]) > TAG_WINDOW:
            t["attempts"] = t["attempts"][-TAG_WINDOW:]

def tag_weakness(state: Dict, tag: str) -> float:
    return decayed_weakness(state["tags"].get(tag, {}).get("attempts", []), now_ts())

def weakest_tag(state: Dict, tag_pool: List[str]) -> Optional[str]:
    if not tag_pool: return None
    return max(tag_pool, key=lambda t: tag_weakness(state, t))

def update_overall_difficulty(state: Dict):
    tags = list(state["tags"].keys())
    if not tags: return
    scores = []
    for tag in tags:
        attempts = state["tags"][tag].get("attempts", [])
        n = len(attempts)
        w = max(1.0, n ** 0.5)
        strength = 1.0 - decayed_weakness(attempts, now_ts())
        scores.append((strength, w))
    total_w = sum(w for _, w in scores) or 1.0
    overall = sum(s * w for s, w in scores) / total_w
    target = clamp((overall - 0.50) / 0.50, 0.0, 1.0)
    state["difficulty"] = clamp(0.88 * float(state["difficulty"]) + 0.12 * target, 0.0, 1.0)

# ══════════════════════════════════════════════════════════════════════
# TASK SELECTION
# ══════════════════════════════════════════════════════════════════════
def pick_task(state: Dict, difficulty: float,
              force_tag: Optional[str] = None,
              exclude_recent: Optional[List[str]] = None) -> Dict:
    pool = tasks_for_difficulty(difficulty)
    if not pool: pool = TASKS

    if force_tag:
        forced = [t for t in pool if force_tag in t["tags"]]
        if forced: pool = forced

    if exclude_recent:
        filtered = [t for t in pool if t["prompt"] not in exclude_recent]
        if len(filtered) >= 3: pool = filtered

    now = now_ts()
    weighted = []
    for task in pool:
        w = 0.15
        for tag in task["tags"]:
            w += decayed_weakness(state["tags"].get(tag, {}).get("attempts", []), now) \
                 / max(1, len(task["tags"]))
        weighted.append((task, w))

    total = sum(w for _, w in weighted)
    r = random.random() * total
    upto = 0.0
    for task, w in weighted:
        upto += w
        if upto >= r: return task
    return weighted[-1][0]

# ══════════════════════════════════════════════════════════════════════
# ANSWER CHECKING (LENIENT)
# ══════════════════════════════════════════════════════════════════════
def normalize(s: str) -> str:
    s = s.strip().lower()
    # Accept typed umlaut substitutions
    for a, b in [("ae", "ä"), ("oe", "ö"), ("ue", "ü"), ("ss", "ß")]:
        s = s.replace(a, b)
    # Strip trailing punctuation
    s = s.rstrip(".!?,")
    return s

def phonetic_normalize(s: str) -> str:
    """Extra normalisation to handle common Whisper German mistakes."""
    s = normalize(s)
    # Whisper sometimes returns 'ei' as 'ai', 'ie' as 'i', 'sch' as 'sh'/'sh' etc.
    replacements = [
        ("ai", "ei"), ("sh", "sch"), ("tsch", "sch"),
        ("ph", "f"),  ("ck", "k"),   ("tz", "z"),
        ("th", "t"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s

def is_correct_voice(transcript: str, answer: str) -> bool:
    """Lenient matching: exact → phonetic → Levenshtein → sentence-word overlap."""
    t = normalize(transcript)
    a = normalize(answer)
    if t == a: return True

    tp = phonetic_normalize(transcript)
    ap = phonetic_normalize(answer)
    if tp == ap: return True

    # Single word: lenient Levenshtein
    if " " not in a:
        if levenshtein(tp, ap) <= LEV_THRESHOLD: return True
        # Also allow if transcript contains the answer as a substring
        if ap in tp or tp in ap: return True
        return False

    # Sentence: word-level overlap — allow up to SENTENCE_MATCH_MISS words missing
    words_a = a.split()
    words_t = set(tp.split())
    matches = sum(1 for wa in words_a
                  if any(levenshtein(wa, wt) <= 1 for wt in words_t))
    return matches >= len(words_a) - SENTENCE_MATCH_MISS

# ══════════════════════════════════════════════════════════════════════
# AUDIO RECORDER  (streams while SPACE held, stops on release)
# ══════════════════════════════════════════════════════════════════════
class AudioRecorder:
    def __init__(self):
        self._buf: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self.recording = False

    def start(self):
        self._buf.clear()
        self.recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._cb, blocksize=1024)
        self._stream.start()

    def _cb(self, indata, frames, time_info, status):
        if self.recording:
            with self._lock:
                self._buf.append(indata.copy())

    def stop(self) -> np.ndarray:
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if self._buf:
                return np.concatenate(self._buf, axis=0).flatten()
        return np.zeros(SAMPLE_RATE, dtype="float32")

# ══════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIBER  (loads in background thread)
# ══════════════════════════════════════════════════════════════════════
class Transcriber:
    def __init__(self):
        self._model = None
        self._ready = False
        self._loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if WHISPER_OK:
            try:
                # Use local_files_only=True so it never attempts a download.
                # The model must already be in the faster-whisper cache
                # (typically ~/.cache/huggingface/hub or ~/.cache/whisper).
                self._model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
                self._ready = True
            except Exception as e:
                print(f"[Whisper] load failed: {e}")
        self._loading = False

    @property
    def ready(self): return self._ready
    @property
    def loading(self): return self._loading

    def transcribe(self, audio: np.ndarray) -> str:
        if not self._ready or self._model is None: return ""
        try:
            segs, _ = self._model.transcribe(
                audio, language="de", beam_size=4,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200})
            return " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print(f"[Whisper] error: {e}")
            return ""

# ══════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════
def draw_bar(surface, rect, frac, color=(80, 220, 120)):
    frac = clamp(frac, 0.0, 1.0)
    pygame.draw.rect(surface, BAR_BG, rect, border_radius=10)
    inner = rect.inflate(-6, -6)
    fw = int(inner.width * frac)
    if fw > 0:
        pygame.draw.rect(surface, color,
                         pygame.Rect(inner.left, inner.top, fw, inner.height),
                         border_radius=8)
    pygame.draw.rect(surface, BAR_BORD, rect, width=2, border_radius=10)

def draw_timer_bar(surface, rect, frac):
    r = int(80 + (1.0 - frac) * 160)
    g = int(220 * frac)
    draw_bar(surface, rect, frac, (r, g, 80))

def rc(surface, font, text, color, cx, cy):
    s = font.render(text, True, color)
    surface.blit(s, s.get_rect(center=(cx, cy)))

def rl(surface, font, text, color, x, y):
    surface.blit(font.render(text, True, color), (x, y))

def rr(surface, font, text, color, rx, y):
    s = font.render(text, True, color)
    surface.blit(s, (rx - s.get_width(), y))

LOBE_BIG   = ["WOW! SUPER! 🚀", "MEGA TOLL! 🎉", "DU BIST SUPER! ⭐", "FANTASTISCH! 🏆", "RICHTIG! 🌟"]
LOBE_SMALL = ["Super! ⭐", "Toll! 🌟", "Prima! 🎉", "Klasse!", "Ja! ✨", "Top! 💪", "Perfekt! 🥇"]
ENCOURAGE  = ["Fast! Nochmal 💪", "Versuch es nochmal!", "Du schaffst das! 🤗", "Nochmal! 🎯"]

STAR_THRESHOLDS = [10, 25, 50, 100, 200, 400]

def stars_earned(total_q: int) -> int:
    return sum(1 for t in STAR_THRESHOLDS if total_q >= t)

# ══════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════
def welcome_screen(screen, fonts, state, w, h) -> bool:
    font_big, font_mid, font_small = fonts
    total_q  = state.get("total_questions_seen", 0)
    sessions = state.get("sessions_count", 0)
    earned   = stars_earned(total_q)
    total_s  = len(STAR_THRESHOLDS)
    next_at  = next((t for t in STAR_THRESHOLDS if t > total_q), None)
    clock    = pygame.time.Clock()

    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                if ev.key == pygame.K_RETURN: return True

        screen.fill(BG)
        rc(screen, font_big, "Deutsch Trainer", WHITE, w//2, int(h*0.20))

        star_str = ("★ " * earned).strip() if earned else "noch keine Sterne"
        rc(screen, font_mid, star_str, YELLOW, w//2, int(h*0.42))

        if next_at:
            prev_at = max((t for t in STAR_THRESHOLDS if t <= total_q), default=0)
            frac = (total_q - prev_at) / max(1, next_at - prev_at)
            draw_bar(screen, pygame.Rect(int(w*0.25), int(h*0.54), int(w*0.50), int(h*0.04)), frac, YELLOW)

        rc(screen, font_mid, "ENTER", GREEN, w//2, int(h*0.75))
        pygame.display.flip()

# ══════════════════════════════════════════════════════════════════════
# SPELLING REVIEW SCREEN  (end-of-session)
# Shows all answers so parent can check what child wrote on paper
# ══════════════════════════════════════════════════════════════════════
def spelling_review_screen(screen, fonts, session_log: List[Dict], w, h):
    """
    session_log: list of {"prompt": str, "answer": str, "correct": bool}
    Parent scrolls through, checks child's paper spelling.
    """
    font_big, font_mid, font_small = fonts
    clock  = pygame.time.Clock()
    scroll = 0
    LINE_H = int(min(w, h) * 0.065)
    VISIBLE = max(4, int(h * 0.65 / LINE_H))

    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_RETURN): return
                if ev.key == pygame.K_DOWN:
                    scroll = min(scroll + 1, max(0, len(session_log) - VISIBLE))
                if ev.key == pygame.K_UP:
                    scroll = max(scroll - 1, 0)

        screen.fill(BG)
        rc(screen, font_big, "Rechtschreibung prüfen", WHITE, w//2, int(h*0.07))
        rc(screen, font_small, "Vergleiche mit dem Blatt  •  ↑↓ scrollen  •  ENTER fertig", DIM, w//2, int(h*0.14))

        y = int(h * 0.20)
        for i, entry in enumerate(session_log[scroll:scroll + VISIBLE]):
            num    = scroll + i + 1
            answer = entry["answer"]
            cor    = entry.get("correct", False)
            dot_color = GREEN if cor else RED
            dot = "●"

            # Number
            rl(screen, font_small, f"{num}.", VDIM, int(w*0.06), y + LINE_H//4)
            # Dot (voice result)
            rc(screen, font_mid, dot, dot_color, int(w*0.12), y + LINE_H//2)
            # Answer in large text
            rc(screen, font_mid, answer, WHITE, w//2, y + LINE_H//2)
            y += LINE_H

        # Scroll hint
        if len(session_log) > VISIBLE:
            shown_end = min(scroll + VISIBLE, len(session_log))
            rc(screen, font_small,
               f"{scroll+1}-{shown_end} von {len(session_log)}",
               VDIM, w//2, int(h*0.92))

        rc(screen, font_small, "ENTER zum Beenden", DIM, w//2, int(h*0.96))
        pygame.display.flip()

# ══════════════════════════════════════════════════════════════════════
# SESSION COMPLETE SCREEN
# ══════════════════════════════════════════════════════════════════════
def complete_screen(screen, fonts, state, q_done, elapsed_s, w, h, session_stars) -> bool:
    """Returns True if user wants spelling review."""
    font_big, font_mid, font_small = fonts
    total_q = state.get("total_questions_seen", 0)
    clock   = pygame.time.Clock()

    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                if ev.key == pygame.K_RETURN: return True   # → spelling review
                if ev.key == pygame.K_SPACE:  return False  # skip review

        screen.fill(BG)
        rc(screen, font_big, "Super gemacht! 🎉", GREEN,  w//2, int(h*0.15))
        rc(screen, font_mid, f"{q_done} Woerter  •  {int(elapsed_s/60)} Minuten", DIM, w//2, int(h*0.28))

        if session_stars > 0:
            msg = f"+{session_stars} neuer Stern!" if session_stars == 1 else f"+{session_stars} neue Sterne!"
            rc(screen, font_mid, msg, YELLOW, w//2, int(h*0.39))

        earned  = stars_earned(total_q)
        total_s = len(STAR_THRESHOLDS)
        star_str = ("★ " * earned) + ("☆ " * (total_s - earned))
        rc(screen, font_mid, star_str.strip(), YELLOW, w//2, int(h*0.52))

        rc(screen, font_mid,  "ENTER  →  Rechtschreibung prüfen", GREEN, w//2, int(h*0.68))
        rc(screen, font_small, "LEERTASTE  →  Fertig", DIM,   w//2, int(h*0.76))
        pygame.display.flip()

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    ensure_dirs()
    state = load_state()

    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    attempts_f   = open(os.path.join(SESSIONS_DIR, session_name + "_attempts.jsonl"),  "a", encoding="utf-8")
    questions_f  = open(os.path.join(SESSIONS_DIR, session_name + "_questions.jsonl"), "a", encoding="utf-8")

    pygame.init()
    pygame.display.set_caption("Deutsch Trainer v4")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h   = screen.get_size()
    clock  = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    # DejaVu Sans has full German umlaut + special char support
    _FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    def load_font(size):
        for p in _FONT_PATHS:
            if os.path.exists(p):
                return pygame.font.Font(p, size)
        return pygame.font.SysFont("dejavusans,liberationsans,freesans,sans", size)
    font_big   = load_font(int(base * 1.9))
    font_mid   = load_font(int(base * 0.85))
    font_input = load_font(int(base * 1.15))
    font_hint  = load_font(int(base * 0.44))
    font_small = load_font(int(base * 0.34))
    fonts      = (font_big, font_mid, font_small)

    # ── Welcome ──────────────────────────────────────────────────────
    if not welcome_screen(screen, fonts, state, w, h):
        pygame.quit(); return

    # ── Load Whisper ──────────────────────────────────────────────────
    recorder    = AudioRecorder()
    transcriber = Transcriber()

    if WHISPER_OK:
        while transcriber.loading:
            screen.fill(BG)
            rc(screen, font_mid, "Lade Sprach-Erkennung...", DIM, w//2, h//2)
            rc(screen, font_small, "(nur beim ersten Start)", VDIM, w//2, int(h*0.58))
            pygame.display.flip()
            clock.tick(FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return

    # ── Session setup ─────────────────────────────────────────────────
    session_start     = now_ts()
    stars_at_start    = stars_earned(state.get("total_questions_seen", 0))
    session_base_diff = clamp(float(state["difficulty"]) - WARMUP_DIFFICULTY_OFFSET, 0.0, 1.0)

    all_phoneme_tags = list({tag for t in TASKS for tag in t["tags"]
                              if tag not in ("l1","l2","l3","satz")})
    warmup_tag = weakest_tag(state, all_phoneme_tags)

    # Per-question state
    q_index        = 0
    completed      = False
    user_text      = ""       # not used for input, just visual
    feedback: Optional[str] = None   # "correct"|"wrong"|"parent_wait"
    feedback_since = 0.0
    feedback_lob   = ""
    attempts_q     = 0
    problem_start  = now_ts()
    problem_solved = False

    # Voice state — single mutable dict so the background thread can write
    # back reliably without nonlocal chains across nested closures
    vs = {
        "transcript":   "",
        "done":         False,
        "rec_active":   False,
        "space_held":   False,
        "transcribing": False,   # True while Whisper thread is running
    }

    # Recent prompts to avoid immediate repeats
    recent_prompts: List[str] = []

    # Session log for spelling review
    session_log: List[Dict] = []

    def elapsed(): return now_ts() - session_start
    def progress(): return clamp(elapsed() / SESSION_DURATION_SECONDS, 0.0, 1.0)
    def in_warmup(): return elapsed() < WARMUP_DURATION_SECONDS

    def cur_diff():
        return clamp(session_base_diff + RAMP_MAX_BONUS * progress(), 0.0, 1.0)

    def next_task():
        ft = warmup_tag if in_warmup() else None
        return pick_task(state, cur_diff(), force_tag=ft, exclude_recent=recent_prompts[-6:])

    task = next_task()

    def start_rec():
        if not (WHISPER_OK and transcriber.ready): return
        vs["done"]       = False
        vs["transcript"] = ""
        vs["rec_active"] = True
        recorder.start()

    def stop_rec_and_transcribe():
        audio = recorder.stop()
        vs["rec_active"]   = False
        vs["transcribing"] = True
        def _run():
            vs["transcript"]   = transcriber.transcribe(audio)
            vs["transcribing"] = False
            vs["done"]         = True   # main loop picks this up next frame
        threading.Thread(target=_run, daemon=True).start()

    def reset_question():
        nonlocal user_text, feedback, attempts_q, problem_start, problem_solved
        user_text = ""; feedback = None; attempts_q = 0
        problem_start = now_ts(); problem_solved = False
        vs["transcript"] = ""; vs["done"] = False
        vs["rec_active"] = False; vs["space_held"] = False; vs["transcribing"] = False

    def log_attempt(correct, typed, rt):
        rec = {
            "t": now_ts(), "q": q_index + 1,
            "prompt": task["prompt"], "answer": task["answer"],
            "typed": typed, "correct": bool(correct),
            "attempt": attempts_q, "rt": float(rt), "tags": task["tags"]
        }
        attempts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        attempts_f.flush()

    def log_summary(final_correct, total_time):
        rec = {
            "t": now_ts(), "q": q_index + 1,
            "prompt": task["prompt"], "answer": task["answer"],
            "final_correct": bool(final_correct),
            "attempts": int(attempts_q), "time_total": float(total_time),
            "difficulty": float(cur_diff()), "tags": task["tags"],
            "warmup": in_warmup()
        }
        questions_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        questions_f.flush()

    def mark_correct(rt):
        nonlocal feedback, feedback_since, feedback_lob, problem_solved
        feedback = "correct"
        feedback_since = now_ts()
        feedback_lob = random.choice(LOBE_BIG if attempts_q <= 1 else LOBE_SMALL)
        log_attempt(True, vs["transcript"], rt)
        if not problem_solved:
            problem_solved = True
            update_tag_stats(state, task["tags"], True, rt)
            state["total_questions_seen"] += 1
            update_overall_difficulty(state)
            save_state(state)
        # Add to session log
        session_log.append({"prompt": task["prompt"], "answer": task["answer"], "correct": True})

    def mark_wrong(rt, typed):
        nonlocal feedback, feedback_since
        feedback = "wrong"
        feedback_since = now_ts()
        log_attempt(False, typed, rt)
        update_tag_stats(state, task["tags"], False, rt)
        update_overall_difficulty(state)
        save_state(state)

    # ── Main loop ─────────────────────────────────────────────────────
    running = True
    while running:
        clock.tick(FPS)

        # Poll voice done
        if vs["done"] and not vs["rec_active"]:
            rt = now_ts() - problem_start
            attempts_q += 1
            result_ok = is_correct_voice(vs["transcript"], task["answer"])

            if result_ok:
                mark_correct(rt)
            else:
                if task["mode"] == "voice_parent":
                    feedback = "parent_wait"
                    feedback_since = now_ts()
                    log_attempt(False, vs["transcript"], rt)
                else:
                    mark_wrong(rt, vs["transcript"])
            vs["done"] = False   # consume

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False; break
                if not completed and feedback != "correct":
                    # SPACE: start recording (always available unless parent_wait)
                    if ev.key == pygame.K_SPACE and not vs["space_held"]:
                        if not vs["rec_active"] and feedback not in ("parent_wait",):
                            vs["space_held"] = True
                            vs["done"] = False
                            vs["transcript"] = ""
                            start_rec()
                    # Parent override Y/N
                    if feedback == "parent_wait":
                        rt = now_ts() - problem_start
                        if ev.key == pygame.K_y:
                            feedback = None
                            mark_correct(rt)
                        elif ev.key == pygame.K_n:
                            feedback = None
                            update_tag_stats(state, task["tags"], False, rt)
                            update_overall_difficulty(state)
                            save_state(state)
                            attempts_q += 1
                            vs["transcript"] = ""
                            vs["done"] = False
                            vs["space_held"] = False

            elif ev.type == pygame.KEYUP:
                # SPACE released → stop mic → transcribe
                # This must be elif, NOT a nested if inside KEYDOWN,
                # so it fires independently every frame
                if ev.key == pygame.K_SPACE:
                    if vs["space_held"]:
                        vs["space_held"] = False
                        if vs["rec_active"]:
                            stop_rec_and_transcribe()

        # ── Advance after correct pause ───────────────────────────────
        if not completed and feedback == "correct":
            if now_ts() - feedback_since >= CORRECT_PAUSE_SECONDS:
                log_summary(True, now_ts() - problem_start)
                q_index += 1
                recent_prompts.append(task["prompt"])

                if progress() >= 1.0:
                    completed = True
                    state["sessions_count"] = state.get("sessions_count", 0) + 1
                    save_state(state)
                else:
                    task = next_task()
                    reset_question()

        # Check time
        if not completed and elapsed() >= SESSION_DURATION_SECONDS:
            completed = True
            state["sessions_count"] = state.get("sessions_count", 0) + 1
            save_state(state)

        # ══════════════════════════════════════════════════════════════
        # DRAW
        # ══════════════════════════════════════════════════════════════
        screen.fill(BG)

        if completed:
            stars_gained = stars_earned(state.get("total_questions_seen", 0)) - stars_at_start
            want_review = complete_screen(screen, fonts, state, q_index,
                                          elapsed(), w, h, stars_gained)
            if want_review and session_log:
                spelling_review_screen(screen, fonts, session_log, w, h)
            running = False
            continue

        # ── Top info bar ──────────────────────────────────────────────
        wu_label = f"WARM-UP [{warmup_tag}]" if in_warmup() and warmup_tag else ""
        if wu_label:
            rl(screen, font_small, wu_label, ORANGE, int(w*0.10), int(h*0.05))

        mode_txt = {"voice":"LESEN  –  sprechen", "voice_parent":"LESEN  –  vorlesen"}
        rl(screen, font_small, mode_txt.get(task["mode"], ""), VDIM, int(w*0.10), int(h*0.08))
        rr(screen, font_small, f"#{q_index+1}", DIM, int(w*0.90), int(h*0.05))
        rr(screen, font_small, f"Level {cur_diff():.2f}", VDIM, int(w*0.90), int(h*0.08))

        # ── Emoji ─────────────────────────────────────────────────────
        rc(screen, font_mid, task.get("emoji", ""), WHITE, int(w*0.18), int(h*0.30))

        # ── Main prompt ───────────────────────────────────────────────
        rc(screen, font_big, task["prompt"], WHITE, w//2, int(h*0.32))

        # ── Voice / recording area ────────────────────────────────────
        if vs["rec_active"]:
            # Big pulsing red circle + text — unmistakable recording state
            pulse = 0.5 + 0.5 * math.sin(now_ts() * 6)
            r_col = (int(180 + 75 * pulse), int(30 + 30 * pulse), 30)
            rc(screen, font_big,  "● REC", r_col, w//2, int(h*0.48))
            rc(screen, font_mid,  "Loslassen zum Prüfen", DIM,   w//2, int(h*0.60))

        elif not vs["rec_active"] and not vs["done"] and not vs["transcribing"]:
            # Idle — waiting for SPACE
            rc(screen, font_mid,  "LEERTASTE halten",   BLUE,  w//2, int(h*0.48))
            rc(screen, font_hint, "loslassen = Prufen", VDIM,  w//2, int(h*0.57))

        elif vs["transcribing"]:
            # Actively running Whisper
            dots = "." * (int(now_ts() * 4) % 5)
            rc(screen, font_big,  f"Pruefe{dots}", YELLOW, w//2, int(h*0.48))

        elif vs["done"] and feedback not in ("correct",):
            # Show transcription result
            disp = f'"{vs["transcript"]}"' if vs["transcript"] else "(nichts gehoert)"
            col  = GREEN if feedback == "correct" else (RED if feedback == "wrong" else BLUE)
            rc(screen, font_input, disp, col, w//2, int(h*0.48))

        # ── Feedback messages ─────────────────────────────────────────
        if feedback == "correct":
            rc(screen, font_mid, feedback_lob, GREEN, w//2, int(h*0.63))
            rc(screen, font_hint, "Jetzt auf Papier aufschreiben!", YELLOW, w//2, int(h*0.72))

        elif feedback == "wrong":
            rc(screen, font_mid, random.choice(ENCOURAGE) if attempts_q <= 1 else "Nochmal! 🎯",
               RED, w//2, int(h*0.63))
            if attempts_q >= 2:
                rc(screen, font_hint, f"Antwort: {task['answer']}", YELLOW, w//2, int(h*0.72))
                # After showing answer, also add to session log so it appears in review
                already = any(e["prompt"] == task["prompt"] for e in session_log)
                if not already:
                    session_log.append({"prompt": task["prompt"], "answer": task["answer"], "correct": False})

        elif feedback == "parent_wait":
            heard = f'Gehört: "{vs["transcript"]}"' if vs["transcript"] else "Nichts gehört"
            rc(screen, font_hint, heard, YELLOW, w//2, int(h*0.50))
            rc(screen, font_mid,  "Richtig gelesen?", WHITE, w//2, int(h*0.60))
            rc(screen, font_mid,  "Y = Ja          N = Nochmal", DIM, w//2, int(h*0.68))

        else:
            # Idle hint
            if task["mode"] == "voice_parent":
                rc(screen, font_hint, "Kind liest laut vor  –  dann LEERTASTE", VDIM, w//2, int(h*0.63))

        # ── Timer bar ─────────────────────────────────────────────────
        tbar = pygame.Rect(int(w*0.10), int(h*0.84), int(w*0.80), int(h*0.025))
        draw_timer_bar(screen, tbar, 1.0 - progress())
        mins_left = max(0, int((SESSION_DURATION_SECONDS - elapsed()) / 60) + 1)
        rl(screen, font_small, f"{mins_left} Min", DIM, int(w*0.10), int(h*0.875))

        # Warmup progress bar (blue, thinner)
        if in_warmup():
            wu_frac = clamp(elapsed() / WARMUP_DURATION_SECONDS, 0.0, 1.0)
            draw_bar(screen,
                     pygame.Rect(int(w*0.10), int(h*0.91), int(w*0.80), int(h*0.012)),
                     wu_frac, BLUE)
            rl(screen, font_small, "Aufwaermen", BLUE, int(w*0.10), int(h*0.927))

        # Stars mini display
        earned  = stars_earned(state.get("total_questions_seen", 0))
        total_s = len(STAR_THRESHOLDS)
        star_str = ("★" * earned) + ("☆" * (total_s - earned))
        rr(screen, font_small, star_str, YELLOW, int(w*0.90), int(h*0.875))

        rc(screen, font_small, "ESC beenden", VDIM, w//2, int(h*0.96))
        pygame.display.flip()

    # ── Cleanup ───────────────────────────────────────────────────────
    if vs["rec_active"]:
        recorder.stop()
    attempts_f.close()
    questions_f.close()
    pygame.quit()


if __name__ == "__main__":
    main()