"""
Deutsch Trainer v5
==================
Flow per word:
  1. Word/sentence appears on screen
  2. Child reads it silently, then HOLDS SPACE to record -> releases to check
  3. Whisper transcribes -> lenient fuzzy match -> Green / Red
  4. Parent can override with Y (correct) or N (wrong) after wrong result
  5. Child writes the answer on paper (no typing in app during session)
  6. At end of session: spelling review screen shows all answers so parent
     can check the paper

New in v5:
  - Explicit state machine (idle/recording/transcribing/result_correct/
    result_wrong/parent_confirm/advance)
  - All bug fixes: spacebar fallback, no emoji rendering, dedup session_log,
    attempts_q increment fix, transcribing guard on poll
  - ESC = skip word (hold 2s to quit)
  - Visual: pulsing red REC circle, animated transcribing dots, border flash
  - Pronunciation hint after 2+ wrong attempts
  - Improved spelling review with score, prompt+answer, arrow nav
  - Parent debug info bottom-right

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

# ======================================================================
# CONFIG
# ======================================================================
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

# ESC hold duration to quit (seconds)
ESC_HOLD_TO_QUIT = 2.0

# ======================================================================
# STATE MACHINE STATES
# ======================================================================
ST_IDLE           = "idle"
ST_RECORDING      = "recording"
ST_TRANSCRIBING   = "transcribing"
ST_RESULT_CORRECT = "result_correct"
ST_RESULT_WRONG   = "result_wrong"
ST_PARENT_CONFIRM = "parent_confirm"
ST_ADVANCE        = "advance"

# ======================================================================
# COLOURS
# ======================================================================
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

# ======================================================================
# TASK DATABASE
# Every task: prompt, answer, mode, tags, label (short German text, no emoji)
# Difficulty gates:
#   0.00 -> l1 only
#   0.15 -> l2 unlocks
#   0.40 -> l3 + sentences
# ======================================================================
TASKS: List[Dict] = [

    # L1 simple words
    {"prompt":"Mama",       "answer":"Mama",       "mode":"voice","tags":["l1","ma"],           "label":"Person"},
    {"prompt":"Papa",       "answer":"Papa",       "mode":"voice","tags":["l1","pa"],           "label":"Person"},
    {"prompt":"Hund",       "answer":"Hund",       "mode":"voice","tags":["l1","nd"],           "label":"Tier"},
    {"prompt":"Katze",      "answer":"Katze",      "mode":"voice","tags":["l1","tz"],           "label":"Tier"},
    {"prompt":"Haus",       "answer":"Haus",       "mode":"voice","tags":["l1","au"],           "label":"Gebaeude"},
    {"prompt":"Ball",       "answer":"Ball",       "mode":"voice","tags":["l1","ll"],           "label":"Spielzeug"},
    {"prompt":"Baum",       "answer":"Baum",       "mode":"voice","tags":["l1","au"],           "label":"Pflanze"},
    {"prompt":"Buch",       "answer":"Buch",       "mode":"voice","tags":["l1","ch"],           "label":"Gegenstand"},
    {"prompt":"Auto",       "answer":"Auto",       "mode":"voice","tags":["l1","au"],           "label":"Fahrzeug"},
    {"prompt":"Milch",      "answer":"Milch",      "mode":"voice","tags":["l1","ch"],           "label":"Getraenk"},
    {"prompt":"Brot",       "answer":"Brot",       "mode":"voice","tags":["l1","br"],           "label":"Essen"},
    {"prompt":"Apfel",      "answer":"Apfel",      "mode":"voice","tags":["l1","pf"],           "label":"Obst"},
    {"prompt":"Fisch",      "answer":"Fisch",      "mode":"voice","tags":["l1","sch"],          "label":"Tier"},
    {"prompt":"Mond",       "answer":"Mond",       "mode":"voice","tags":["l1","nd"],           "label":"Himmel"},
    {"prompt":"Sonne",      "answer":"Sonne",      "mode":"voice","tags":["l1","nn"],           "label":"Himmel"},
    {"prompt":"Nase",       "answer":"Nase",       "mode":"voice","tags":["l1","na"],           "label":"Koerper"},
    {"prompt":"Hand",       "answer":"Hand",       "mode":"voice","tags":["l1","nd"],           "label":"Koerper"},
    {"prompt":"Maus",       "answer":"Maus",       "mode":"voice","tags":["l1","au"],           "label":"Tier"},
    {"prompt":"Tisch",      "answer":"Tisch",      "mode":"voice","tags":["l1","sch"],          "label":"Moebel"},
    {"prompt":"Stern",      "answer":"Stern",      "mode":"voice","tags":["l1","st"],           "label":"Himmel"},
    {"prompt":"Wald",       "answer":"Wald",       "mode":"voice","tags":["l1","ld"],           "label":"Natur"},
    {"prompt":"Berg",       "answer":"Berg",       "mode":"voice","tags":["l1","rg"],           "label":"Natur"},
    {"prompt":"Kind",       "answer":"Kind",       "mode":"voice","tags":["l1","nd"],           "label":"Mensch"},

    # L2 tricky phonics
    {"prompt":"Eisen",       "answer":"Eisen",       "mode":"voice","tags":["l2","ei"],          "label":"Metall"},
    {"prompt":"Feuer",       "answer":"Feuer",       "mode":"voice","tags":["l2","eu"],          "label":"Element"},
    {"prompt":"Biene",       "answer":"Biene",       "mode":"voice","tags":["l2","ie"],          "label":"Insekt"},
    {"prompt":"Leiter",      "answer":"Leiter",      "mode":"voice","tags":["l2","ei"],          "label":"Geraet"},
    {"prompt":"Stiefel",     "answer":"Stiefel",     "mode":"voice","tags":["l2","ie","st"],     "label":"Kleidung"},
    {"prompt":"Schule",      "answer":"Schule",      "mode":"voice","tags":["l2","sch"],         "label":"Gebaeude"},
    {"prompt":"Schaukel",    "answer":"Schaukel",    "mode":"voice","tags":["l2","sch","au"],    "label":"Spielplatz"},
    {"prompt":"Fleisch",     "answer":"Fleisch",     "mode":"voice","tags":["l2","ei","sch"],    "label":"Essen"},
    {"prompt":"Freund",      "answer":"Freund",      "mode":"voice","tags":["l2","eu"],          "label":"Mensch"},
    {"prompt":"Scheune",     "answer":"Scheune",     "mode":"voice","tags":["l2","eu","sch"],    "label":"Gebaeude"},
    {"prompt":"Spielplatz",  "answer":"Spielplatz",  "mode":"voice","tags":["l2","ie","sp"],     "label":"Ort"},
    {"prompt":"Schnee",      "answer":"Schnee",      "mode":"voice","tags":["l2","sch","ee"],    "label":"Wetter"},
    {"prompt":"Blume",       "answer":"Blume",       "mode":"voice","tags":["l2","bl"],          "label":"Pflanze"},
    {"prompt":"Pflanze",     "answer":"Pflanze",     "mode":"voice","tags":["l2","pf"],          "label":"Pflanze"},
    {"prompt":"Strasse",     "answer":"Strasse",     "mode":"voice","tags":["l2","st"],          "label":"Ort"},
    {"prompt":"Schmetterling","answer":"Schmetterling","mode":"voice","tags":["l2","sch"],       "label":"Insekt"},
    {"prompt":"Erdbeere",    "answer":"Erdbeere",    "mode":"voice","tags":["l2","ee"],          "label":"Obst"},
    {"prompt":"Regenbogen",  "answer":"Regenbogen",  "mode":"voice","tags":["l2"],              "label":"Wetter"},
    {"prompt":"Geschenk",    "answer":"Geschenk",    "mode":"voice","tags":["l2","sch"],         "label":"Gegenstand"},
    {"prompt":"Eichhoernchen","answer":"Eichhoernchen","mode":"voice","tags":["l2","ei","ch"],   "label":"Tier"},
    {"prompt":"Haeuschen",   "answer":"Haeuschen",   "mode":"voice","tags":["l2","au","sch","ae"],"label":"Gebaeude"},
    {"prompt":"Maeuse",      "answer":"Maeuse",      "mode":"voice","tags":["l2","eu","ae"],     "label":"Tier"},
    {"prompt":"Fruehstueck", "answer":"Fruehstueck", "mode":"voice","tags":["l2","ue","sch"],    "label":"Essen"},
    {"prompt":"Geburtstag",  "answer":"Geburtstag",  "mode":"voice","tags":["l2","st"],          "label":"Feier"},
    {"prompt":"Flugzeug",    "answer":"Flugzeug",    "mode":"voice","tags":["l2","eu"],          "label":"Fahrzeug"},
    {"prompt":"Schornstein", "answer":"Schornstein", "mode":"voice","tags":["l2","sch","st","ei"],"label":"Gebaeude"},
    {"prompt":"Krankenwagen","answer":"Krankenwagen","mode":"voice","tags":["l2"],              "label":"Fahrzeug"},

    # L3 simple sentences (parent-confirmed)
    {"prompt":"Die Sonne scheint.",           "answer":"Die Sonne scheint.",           "mode":"voice_parent","tags":["l3","satz","sch"],       "label":"Satz"},
    {"prompt":"Der Hund bellt laut.",         "answer":"Der Hund bellt laut.",         "mode":"voice_parent","tags":["l3","satz"],             "label":"Satz"},
    {"prompt":"Ich mag Eis.",                 "answer":"Ich mag Eis.",                 "mode":"voice_parent","tags":["l3","satz"],             "label":"Satz"},
    {"prompt":"Das Auto ist rot.",            "answer":"Das Auto ist rot.",            "mode":"voice_parent","tags":["l3","satz","au"],        "label":"Satz"},
    {"prompt":"Die Katze schlaeft.",          "answer":"Die Katze schlaeft.",          "mode":"voice_parent","tags":["l3","satz","sch"],       "label":"Satz"},
    {"prompt":"Ich spiele im Garten.",        "answer":"Ich spiele im Garten.",        "mode":"voice_parent","tags":["l3","satz","ie","sp"],   "label":"Satz"},
    {"prompt":"Der Schnee ist weiss.",        "answer":"Der Schnee ist weiss.",        "mode":"voice_parent","tags":["l3","satz","sch","ei"],  "label":"Satz"},
    {"prompt":"Wir fahren mit dem Auto.",     "answer":"Wir fahren mit dem Auto.",     "mode":"voice_parent","tags":["l3","satz","au"],        "label":"Satz"},
    {"prompt":"Die Biene fliegt zur Blume.",  "answer":"Die Biene fliegt zur Blume.",  "mode":"voice_parent","tags":["l3","satz","ie","bl"],   "label":"Satz"},
    {"prompt":"Mein Freund heisst Leo.",      "answer":"Mein Freund heisst Leo.",      "mode":"voice_parent","tags":["l3","satz","eu","ei"],   "label":"Satz"},
    {"prompt":"Der Stern leuchtet hell.",     "answer":"Der Stern leuchtet hell.",     "mode":"voice_parent","tags":["l3","satz","st","eu"],   "label":"Satz"},
    {"prompt":"Heute scheint die Sonne schoen.","answer":"Heute scheint die Sonne schoen.","mode":"voice_parent","tags":["l3","satz","sch","eu"],"label":"Satz"},
    {"prompt":"Im Fruehling bluehen Blumen.", "answer":"Im Fruehling bluehen Blumen.", "mode":"voice_parent","tags":["l3","satz","ue","bl"],   "label":"Satz"},
]

# Difficulty gates
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

# ======================================================================
# LEVENSHTEIN
# ======================================================================
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

# ======================================================================
# PERSISTENCE
# ======================================================================
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

# ======================================================================
# SPACED REPETITION + TAG METRICS
# ======================================================================
def decayed_weakness(attempts: List[Dict], now: float) -> float:
    if not attempts: return 0.55
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

# ======================================================================
# TASK SELECTION
# ======================================================================
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

# ======================================================================
# ANSWER CHECKING (LENIENT)
# ======================================================================
def normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.rstrip(".!?,")
    return s

def phonetic_normalize(s: str) -> str:
    s = normalize(s)
    replacements = [
        ("ai", "ei"), ("sh", "sch"), ("tsch", "sch"),
        ("ph", "f"),  ("ck", "k"),   ("tz", "z"),
        ("th", "t"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s

def is_correct_voice(transcript: str, answer: str) -> bool:
    t = normalize(transcript)
    a = normalize(answer)
    if t == a: return True
    tp = phonetic_normalize(transcript)
    ap = phonetic_normalize(answer)
    if tp == ap: return True
    if " " not in a:
        if levenshtein(tp, ap) <= LEV_THRESHOLD: return True
        if ap in tp or tp in ap: return True
        return False
    words_a = a.split()
    words_t = set(tp.split())
    matches = sum(1 for wa in words_a
                  if any(levenshtein(wa, wt) <= 1 for wt in words_t))
    return matches >= len(words_a) - SENTENCE_MATCH_MISS

# ======================================================================
# AUDIO RECORDER
# ======================================================================
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

# ======================================================================
# WHISPER TRANSCRIBER
# ======================================================================
class Transcriber:
    def __init__(self):
        self._model = None
        self._ready = False
        self._loading = True
        self._error = ""
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not WHISPER_OK:
            self._error = "faster-whisper not installed"
            self._loading = False
            return

        # Try model sizes in order of preference
        model_sizes = [WHISPER_MODEL_SIZE, "base", "tiny"]
        # De-duplicate while preserving order
        seen = set()
        unique_sizes = []
        for s in model_sizes:
            if s not in seen:
                seen.add(s)
                unique_sizes.append(s)

        for size in unique_sizes:
            # Strategy 1: try local_files_only=True first
            try:
                print(f"[DT] trying Whisper model '{size}' (local only)...")
                self._model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
                self._ready = True
                print(f"[DT] Whisper model '{size}' loaded successfully (local)")
                self._loading = False
                return
            except Exception as e:
                print(f"[DT] Whisper local load '{size}' failed: {e}")

            # Strategy 2: allow download
            try:
                print(f"[DT] trying Whisper model '{size}' (allow download)...")
                self._model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",
                )
                self._ready = True
                print(f"[DT] Whisper model '{size}' loaded successfully (downloaded)")
                self._loading = False
                return
            except Exception as e:
                print(f"[DT] Whisper download load '{size}' failed: {e}")

        self._error = "All Whisper model load attempts failed. Check terminal for details."
        print(f"[DT] ERROR: {self._error}")
        self._loading = False

    @property
    def ready(self): return self._ready
    @property
    def loading(self): return self._loading
    @property
    def error(self): return self._error

    def transcribe(self, audio: np.ndarray) -> str:
        if not self._ready or self._model is None: return ""
        try:
            segs, _ = self._model.transcribe(
                audio, language="de", beam_size=4,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200})
            return " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print(f"[DT] Whisper transcribe error: {e}")
            return ""

# ======================================================================
# FONT LOADER (call after pygame.init())
# ======================================================================
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

def load_font(size: int) -> pygame.font.Font:
    """Load a font that supports German umlauts. Call after pygame.init()."""
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return pygame.font.Font(p, size)
    return pygame.font.SysFont("dejavusans,liberationsans,freesans,sans", size)

# ======================================================================
# UI HELPERS
# ======================================================================
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
    """Render text centered at (cx, cy)."""
    s = font.render(text, True, color)
    surface.blit(s, s.get_rect(center=(cx, cy)))

def rl(surface, font, text, color, x, y):
    """Render text left-aligned at (x, y)."""
    surface.blit(font.render(text, True, color), (x, y))

def rr(surface, font, text, color, rx, y):
    """Render text right-aligned (right edge at rx, top at y)."""
    s = font.render(text, True, color)
    surface.blit(s, (rx - s.get_width(), y))

def draw_border_flash(surface, w, h, color, thickness=8, alpha_frac=1.0):
    """Draw a colored border around the screen edges."""
    c = tuple(int(ch * alpha_frac) for ch in color)
    pygame.draw.rect(surface, c, pygame.Rect(0, 0, w, thickness))
    pygame.draw.rect(surface, c, pygame.Rect(0, h - thickness, w, thickness))
    pygame.draw.rect(surface, c, pygame.Rect(0, 0, thickness, h))
    pygame.draw.rect(surface, c, pygame.Rect(w - thickness, 0, thickness, h))

LOBE_BIG   = ["WOW! SUPER!", "MEGA TOLL!", "DU BIST SUPER!", "FANTASTISCH!", "RICHTIG!"]
LOBE_SMALL = ["Super!", "Toll!", "Prima!", "Klasse!", "Ja!", "Top!", "Perfekt!"]
ENCOURAGE  = ["Fast! Nochmal!", "Versuch es nochmal!", "Du schaffst das!", "Nochmal!"]

STAR_THRESHOLDS = [10, 25, 50, 100, 200, 400]

def stars_earned(total_q: int) -> int:
    return sum(1 for t in STAR_THRESHOLDS if total_q >= t)

# ======================================================================
# WELCOME SCREEN
# ======================================================================
def welcome_screen(screen, fonts, state, w, h) -> bool:
    font_big, font_mid, font_small = fonts
    total_q  = state.get("total_questions_seen", 0)
    earned   = stars_earned(total_q)
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
        # Stars using DejaVu-safe characters
        if earned > 0:
            total_s = len(STAR_THRESHOLDS)
            star_str = "\u2605 " * earned + "\u2606 " * (total_s - earned)
            rc(screen, font_mid, star_str.strip(), YELLOW, w//2, int(h*0.42))
        else:
            rc(screen, font_mid, "noch keine Sterne", DIM, w//2, int(h*0.42))
        if next_at:
            prev_at = max((t for t in STAR_THRESHOLDS if t <= total_q), default=0)
            frac = (total_q - prev_at) / max(1, next_at - prev_at)
            draw_bar(screen, pygame.Rect(int(w*0.25), int(h*0.54), int(w*0.50), int(h*0.04)), frac, YELLOW)
        rc(screen, font_mid, "ENTER zum Starten", GREEN, w//2, int(h*0.75))
        pygame.display.flip()

# ======================================================================
# SPELLING REVIEW SCREEN (IMPROVED)
# ======================================================================
def spelling_review_screen(screen, fonts, session_log: List[Dict], w, h):
    font_big, font_mid, font_small = fonts
    clock  = pygame.time.Clock()
    scroll = 0
    LINE_H = int(min(w, h) * 0.065)
    VISIBLE = max(4, int(h * 0.55 / LINE_H))

    correct_count = sum(1 for e in session_log if e.get("correct"))
    total_count = len(session_log)

    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_RETURN): return
                if ev.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    scroll = min(scroll + 1, max(0, total_count - VISIBLE))
                if ev.key in (pygame.K_UP, pygame.K_LEFT):
                    scroll = max(scroll - 1, 0)

        screen.fill(BG)
        rc(screen, font_big, "Rechtschreibung pruefen", WHITE, w//2, int(h*0.07))

        # Score line
        score_color = GREEN if correct_count >= total_count * 0.7 else YELLOW if correct_count >= total_count * 0.4 else RED
        rc(screen, font_mid, f"Score: {correct_count}/{total_count}", score_color, w//2, int(h*0.15))

        rc(screen, font_small, "Vergleiche mit dem Blatt  |  Pfeiltasten  |  ENTER fertig", DIM, w//2, int(h*0.20))

        y = int(h * 0.25)
        for i, entry in enumerate(session_log[scroll:scroll + VISIBLE]):
            num    = scroll + i + 1
            prompt = entry.get("prompt", "")
            answer = entry["answer"]
            cor    = entry.get("correct", False)
            mark_color = GREEN if cor else RED
            # Use plain ASCII check/cross marks
            mark = "V" if cor else "X"

            # Number
            rl(screen, font_small, f"{num}.", VDIM, int(w*0.06), y + LINE_H//4)
            # Check/cross mark
            rc(screen, font_mid, mark, mark_color, int(w*0.12), y + LINE_H//2)
            # Prompt on left side
            rl(screen, font_small, prompt, DIM, int(w*0.17), y + LINE_H//4)
            # Answer large in center-right area
            rc(screen, font_mid, answer, WHITE, int(w*0.65), y + LINE_H//2)
            y += LINE_H

        if total_count > VISIBLE:
            shown_end = min(scroll + VISIBLE, total_count)
            rc(screen, font_small, f"{scroll+1}-{shown_end} von {total_count}", VDIM, w//2, int(h*0.88))

        rc(screen, font_small, "ENTER zum Beenden", DIM, w//2, int(h*0.96))
        pygame.display.flip()

# ======================================================================
# SESSION COMPLETE SCREEN
# ======================================================================
def complete_screen(screen, fonts, state, q_done, elapsed_s, w, h, session_stars) -> bool:
    font_big, font_mid, font_small = fonts
    clock   = pygame.time.Clock()

    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                if ev.key == pygame.K_RETURN: return True
                if ev.key == pygame.K_SPACE:  return False

        screen.fill(BG)
        rc(screen, font_big, "Super gemacht!", GREEN,  w//2, int(h*0.15))
        rc(screen, font_mid, f"{q_done} Woerter  -  {int(elapsed_s/60)} Minuten", DIM, w//2, int(h*0.28))
        if session_stars > 0:
            msg = f"+{session_stars} neuer Stern!" if session_stars == 1 else f"+{session_stars} neue Sterne!"
            rc(screen, font_mid, msg, YELLOW, w//2, int(h*0.39))
        total_q = state.get("total_questions_seen", 0)
        earned  = stars_earned(total_q)
        total_s = len(STAR_THRESHOLDS)
        star_str = ("\u2605 " * earned + "\u2606 " * (total_s - earned)).strip()
        rc(screen, font_mid, star_str, YELLOW, w//2, int(h*0.52))
        rc(screen, font_mid,  "ENTER -> Rechtschreibung pruefen", GREEN, w//2, int(h*0.68))
        rc(screen, font_small, "LEERTASTE -> Fertig", DIM,   w//2, int(h*0.76))
        pygame.display.flip()

# ======================================================================
# MAIN
# ======================================================================
def main():
    ensure_dirs()
    state = load_state()

    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    attempts_f   = open(os.path.join(SESSIONS_DIR, session_name + "_attempts.jsonl"),  "a", encoding="utf-8")
    questions_f  = open(os.path.join(SESSIONS_DIR, session_name + "_questions.jsonl"), "a", encoding="utf-8")

    pygame.init()
    pygame.key.set_repeat(0)   # CRITICAL: disable key repeat so KEYUP fires cleanly
    print("[DT] pygame.key.set_repeat(0) called - key repeat disabled")
    pygame.display.set_caption("Deutsch Trainer v5")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h   = screen.get_size()
    clock  = pygame.time.Clock()

    base = max(24, min(w, h) // 11)
    font_big   = load_font(int(base * 1.9))
    font_mid   = load_font(int(base * 0.85))
    font_input = load_font(int(base * 1.15))
    font_hint  = load_font(int(base * 0.44))
    font_small = load_font(int(base * 0.34))
    font_tiny  = load_font(int(base * 0.26))
    fonts      = (font_big, font_mid, font_small)

    if not welcome_screen(screen, fonts, state, w, h):
        pygame.quit(); return

    recorder    = AudioRecorder()
    transcriber = Transcriber()

    # Wait for Whisper to finish loading (or fail)
    if WHISPER_OK:
        while transcriber.loading:
            screen.fill(BG)
            rc(screen, font_mid, "Lade Sprach-Erkennung...", DIM, w//2, h//2)
            pygame.display.flip()
            clock.tick(FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return

        # If Whisper failed, show error and let user press ENTER to continue or ESC to quit
        if not transcriber.ready:
            print(f"[DT] Whisper not ready after loading. Error: {transcriber.error}")
            waiting_for_user = True
            while waiting_for_user:
                screen.fill(BG)
                rc(screen, font_mid, "Sprach-Erkennung fehlgeschlagen!", RED, w//2, int(h*0.35))
                rc(screen, font_small, transcriber.error, DIM, w//2, int(h*0.45))
                rc(screen, font_small, "Das Modell wird jetzt heruntergeladen...", YELLOW, w//2, int(h*0.55))
                rc(screen, font_small, "Bitte warten oder ESC zum Beenden", DIM, w//2, int(h*0.65))
                pygame.display.flip()
                clock.tick(FPS)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            pygame.quit(); return
                        if ev.key == pygame.K_RETURN:
                            waiting_for_user = False
                # Check if a background retry succeeded (transcriber thread may still be running
                # in rare edge cases — this is just a safety check)
                if transcriber.ready:
                    waiting_for_user = False

    session_start     = now_ts()
    stars_at_start    = stars_earned(state.get("total_questions_seen", 0))
    session_base_diff = clamp(float(state["difficulty"]) - WARMUP_DIFFICULTY_OFFSET, 0.0, 1.0)

    all_phoneme_tags = list({tag for t in TASKS for tag in t["tags"]
                              if tag not in ("l1","l2","l3","satz")})
    warmup_tag = weakest_tag(state, all_phoneme_tags)
    print(f"[DT] session start. warmup_tag={warmup_tag} difficulty={state['difficulty']:.3f}")

    q_index        = 0
    completed      = False
    attempts_q     = 0
    problem_start  = now_ts()
    problem_solved = False

    # State machine
    machine_state  = ST_IDLE
    feedback_since = 0.0
    feedback_lob   = ""
    transcript     = ""

    # Voice state
    vs = {
        "transcript":   "",
        "done":         False,
        "rec_active":   False,
        "space_held":   False,
        "transcribing": False,
    }

    recent_prompts: List[str] = []
    session_log: List[Dict] = []

    # ESC tracking for hold-to-quit
    esc_held_since: Optional[float] = None

    # Border flash effect
    border_flash_color: Optional[Tuple[int,int,int]] = None
    border_flash_start: float = 0.0
    BORDER_FLASH_DURATION = 0.4

    def elapsed(): return now_ts() - session_start
    def progress(): return clamp(elapsed() / SESSION_DURATION_SECONDS, 0.0, 1.0)
    def in_warmup(): return elapsed() < WARMUP_DURATION_SECONDS
    def cur_diff():
        return clamp(session_base_diff + RAMP_MAX_BONUS * progress(), 0.0, 1.0)
    def next_task():
        ft = warmup_tag if in_warmup() else None
        return pick_task(state, cur_diff(), force_tag=ft, exclude_recent=recent_prompts[-6:])

    task = next_task()
    print(f"[DT] first task: '{task['prompt']}' mode={task['mode']}")

    def start_rec() -> bool:
        nonlocal machine_state
        if not (WHISPER_OK and transcriber.ready):
            print("[DT] cannot record: Whisper not ready")
            return False
        vs["done"]       = False
        vs["transcript"] = ""
        vs["rec_active"] = True
        machine_state = ST_RECORDING
        recorder.start()
        print(f"[DT] state -> {ST_RECORDING}: recording started")
        return True

    def stop_rec_and_transcribe():
        nonlocal machine_state
        audio = recorder.stop()
        vs["rec_active"]   = False
        vs["transcribing"] = True
        machine_state = ST_TRANSCRIBING
        print(f"[DT] state -> {ST_TRANSCRIBING}: transcribing...")
        def _run():
            vs["transcript"]   = transcriber.transcribe(audio)
            vs["transcribing"] = False
            vs["done"]         = True
            print(f"[DT] transcription done: '{vs['transcript']}'")
        threading.Thread(target=_run, daemon=True).start()

    def reset_question():
        nonlocal machine_state, attempts_q, problem_start, problem_solved
        machine_state = ST_IDLE
        attempts_q = 0
        problem_start = now_ts()
        problem_solved = False
        vs["transcript"] = ""; vs["done"] = False
        vs["rec_active"] = False; vs["space_held"] = False; vs["transcribing"] = False
        print(f"[DT] state -> {ST_IDLE}: question reset")

    def trigger_border_flash(color):
        nonlocal border_flash_color, border_flash_start
        border_flash_color = color
        border_flash_start = now_ts()

    def log_attempt(correct, typed, rt):
        rec = {"t": now_ts(), "q": q_index + 1, "prompt": task["prompt"],
               "answer": task["answer"], "typed": typed, "correct": bool(correct),
               "attempt": attempts_q, "rt": float(rt), "tags": task["tags"]}
        attempts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        attempts_f.flush()

    def log_summary(final_correct, total_time):
        rec = {"t": now_ts(), "q": q_index + 1, "prompt": task["prompt"],
               "answer": task["answer"], "final_correct": bool(final_correct),
               "attempts": int(attempts_q), "time_total": float(total_time),
               "difficulty": float(cur_diff()), "tags": task["tags"], "warmup": in_warmup()}
        questions_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        questions_f.flush()

    def mark_correct(rt):
        nonlocal machine_state, feedback_since, feedback_lob, problem_solved, attempts_q
        # BUG 4 FIX: increment attempts_q here, not in poll
        attempts_q += 1
        machine_state = ST_RESULT_CORRECT
        feedback_since = now_ts()
        feedback_lob = random.choice(LOBE_BIG if attempts_q <= 1 else LOBE_SMALL)
        log_attempt(True, vs["transcript"], rt)
        trigger_border_flash(GREEN)
        if not problem_solved:
            problem_solved = True
            update_tag_stats(state, task["tags"], True, rt)
            state["total_questions_seen"] += 1
            update_overall_difficulty(state)
            save_state(state)
        # BUG 5 FIX: dedup before append
        if not any(e["prompt"] == task["prompt"] for e in session_log):
            session_log.append({"prompt": task["prompt"], "answer": task["answer"], "correct": True})
        print(f"[DT] state -> {ST_RESULT_CORRECT}: correct! attempts={attempts_q} lob='{feedback_lob}'")

    def mark_wrong(rt, typed):
        nonlocal machine_state, feedback_since, attempts_q
        # BUG 4 FIX: increment attempts_q here, not in poll
        attempts_q += 1
        machine_state = ST_RESULT_WRONG
        feedback_since = now_ts()
        log_attempt(False, typed, rt)
        trigger_border_flash(RED)
        update_tag_stats(state, task["tags"], False, rt)
        update_overall_difficulty(state)
        save_state(state)
        # BUG 5 FIX: dedup before append; log wrong after 2 failed attempts
        if attempts_q >= 2:
            if not any(e["prompt"] == task["prompt"] for e in session_log):
                session_log.append({"prompt": task["prompt"], "answer": task["answer"], "correct": False})
        print(f"[DT] state -> {ST_RESULT_WRONG}: wrong. attempts={attempts_q} typed='{typed}'")

    running = True
    while running:
        clock.tick(FPS)
        t_now = now_ts()

        # ---- ESC hold-to-quit tracking ----
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            if esc_held_since is None:
                esc_held_since = t_now
            elif t_now - esc_held_since >= ESC_HOLD_TO_QUIT:
                print("[DT] ESC held 2s -> quitting")
                running = False
                break
        else:
            esc_held_since = None

        # ---- Fallback: if space_held but key is no longer pressed, treat as KEYUP ----
        if vs["space_held"] and not keys[pygame.K_SPACE]:
            print("[DT] KEYUP fallback triggered (get_pressed shows SPACE released)")
            vs["space_held"] = False
            if vs["rec_active"]:
                stop_rec_and_transcribe()

        # ---- Poll voice done (BUG 3 FIX: also check not transcribing) ----
        if vs["done"] and not vs["rec_active"] and not vs["transcribing"]:
            rt = t_now - problem_start
            result_ok = is_correct_voice(vs["transcript"], task["answer"])
            print(f"[DT] voice poll: result_ok={result_ok} transcript='{vs['transcript']}' answer='{task['answer']}'")
            if result_ok:
                mark_correct(rt)
            else:
                if task["mode"] == "voice_parent":
                    machine_state = ST_PARENT_CONFIRM
                    feedback_since = t_now
                    # Don't increment attempts_q here; it's handled when parent decides
                    print(f"[DT] state -> {ST_PARENT_CONFIRM}: waiting for parent Y/N")
                else:
                    mark_wrong(rt, vs["transcript"])
            vs["done"] = False

        # ---- Event handling ----
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if ev.type == pygame.KEYDOWN:
                # ESC single press = skip word
                if ev.key == pygame.K_ESCAPE:
                    if not completed and machine_state not in (ST_RESULT_CORRECT, ST_ADVANCE):
                        # Skip this word
                        print(f"[SKIP] skipping '{task['prompt']}'")
                        # Stop any active recording
                        if vs["rec_active"]:
                            recorder.stop()
                            vs["rec_active"] = False
                        vs["space_held"] = False
                        vs["transcribing"] = False
                        vs["done"] = False
                        vs["transcript"] = ""
                        # Move to next task without logging
                        q_index += 1
                        recent_prompts.append(task["prompt"])
                        if progress() >= 1.0:
                            completed = True
                            state["sessions_count"] = state.get("sessions_count", 0) + 1
                            save_state(state)
                        else:
                            task = next_task()
                            reset_question()
                            print(f"[DT] next task after skip: '{task['prompt']}' mode={task['mode']}")
                    continue

                if not completed:
                    # SPACE down -> start recording
                    if ev.key == pygame.K_SPACE and not vs["space_held"]:
                        if machine_state in (ST_IDLE, ST_RESULT_WRONG):
                            if not vs["rec_active"] and not vs["transcribing"]:
                                vs["space_held"] = True
                                vs["done"] = False
                                vs["transcript"] = ""
                                print(f"[DT] SPACE down in state={machine_state} -> starting recording")
                                if not start_rec():
                                    vs["space_held"] = False  # reset — recording didn't actually start

                    # Parent confirm keys
                    if machine_state == ST_PARENT_CONFIRM:
                        rt = t_now - problem_start
                        if ev.key == pygame.K_y:
                            print("[DT] parent pressed Y -> marking correct")
                            mark_correct(rt)
                        elif ev.key == pygame.K_n:
                            print("[DT] parent pressed N -> marking wrong, retry")
                            mark_wrong(rt, vs["transcript"])
                            # After parent N, go to result_wrong so child can retry
                            vs["transcript"] = ""
                            vs["done"] = False
                            vs["space_held"] = False

            elif ev.type == pygame.KEYUP:
                if ev.key == pygame.K_SPACE and vs["space_held"]:
                    print("[DT] SPACE up (KEYUP event) -> stopping recording")
                    vs["space_held"] = False
                    if vs["rec_active"]:
                        stop_rec_and_transcribe()

        # ---- Advance after correct pause ----
        if not completed and machine_state == ST_RESULT_CORRECT:
            if t_now - feedback_since >= CORRECT_PAUSE_SECONDS:
                machine_state = ST_ADVANCE
                print(f"[DT] state -> {ST_ADVANCE}: advancing to next task")
                log_summary(True, t_now - problem_start)
                q_index += 1
                recent_prompts.append(task["prompt"])
                if progress() >= 1.0:
                    completed = True
                    state["sessions_count"] = state.get("sessions_count", 0) + 1
                    save_state(state)
                else:
                    task = next_task()
                    reset_question()
                    print(f"[DT] next task: '{task['prompt']}' mode={task['mode']}")

        # ---- Time-based session end ----
        if not completed and elapsed() >= SESSION_DURATION_SECONDS:
            completed = True
            state["sessions_count"] = state.get("sessions_count", 0) + 1
            save_state(state)
            print("[DT] session time expired -> completed")

        # ================================================================
        # DRAW
        # ================================================================
        screen.fill(BG)

        if completed:
            stars_gained = stars_earned(state.get("total_questions_seen", 0)) - stars_at_start
            want_review = complete_screen(screen, fonts, state, q_index,
                                          elapsed(), w, h, stars_gained)
            if want_review and session_log:
                spelling_review_screen(screen, fonts, session_log, w, h)
            running = False
            continue

        # ---- Top bar ----
        wu_label = f"WARM-UP [{warmup_tag}]" if in_warmup() and warmup_tag else ""
        if wu_label:
            rl(screen, font_small, wu_label, ORANGE, int(w*0.10), int(h*0.05))
        mode_txt = {"voice":"LESEN - sprechen", "voice_parent":"LESEN - vorlesen"}
        rl(screen, font_small, mode_txt.get(task["mode"], ""), VDIM, int(w*0.10), int(h*0.08))
        rr(screen, font_small, f"#{q_index+1}", DIM, int(w*0.90), int(h*0.05))

        # ---- Main prompt word ----
        rc(screen, font_big, task["prompt"], WHITE, w//2, int(h*0.32))
        # Category label (plain text, no emoji)
        label = task.get("label", "")
        if label:
            rc(screen, font_hint, label, VDIM, w//2, int(h*0.42))

        # ---- State-based voice area drawing ----
        if machine_state == ST_RECORDING:
            # Pulsing red circle + REC text
            pulse = math.sin(t_now * 6)
            radius = int(30 + 10 * pulse)
            center_x = w // 2
            center_y = int(h * 0.52)
            alpha = 0.6 + 0.4 * (0.5 + 0.5 * pulse)
            r_val = int(220 * alpha)
            g_val = int(40 * alpha)
            b_val = int(40 * alpha)
            pygame.draw.circle(screen, (r_val, g_val, b_val), (center_x - 80, center_y), radius)
            rc(screen, font_big, "REC", (r_val, g_val, b_val), center_x + 20, center_y)
            rc(screen, font_mid, "Loslassen zum Pruefen", DIM, w//2, int(h*0.62))

        elif machine_state == ST_TRANSCRIBING:
            # Animated dots
            num_dots = 5
            active_dot = int(t_now * 5) % num_dots
            dot_y = int(h * 0.52)
            dot_spacing = 30
            start_x = w // 2 - (num_dots - 1) * dot_spacing // 2
            for i in range(num_dots):
                dx = start_x + i * dot_spacing
                if i == active_dot:
                    pygame.draw.circle(screen, YELLOW, (dx, dot_y), 10)
                else:
                    pygame.draw.circle(screen, VDIM, (dx, dot_y), 6)
            rc(screen, font_mid, "Pruefe...", YELLOW, w//2, int(h*0.58))

        elif machine_state == ST_RESULT_CORRECT:
            rc(screen, font_mid, feedback_lob, GREEN, w//2, int(h*0.55))
            rc(screen, font_hint, "Jetzt aufschreiben!", YELLOW, w//2, int(h*0.63))

        elif machine_state == ST_RESULT_WRONG:
            # Show transcription
            disp = f'"{vs["transcript"]}"' if vs["transcript"] else "(nichts gehoert)"
            rc(screen, font_input, disp, RED, w//2, int(h*0.50))
            # Encouragement
            if attempts_q <= 1:
                rc(screen, font_mid, random.choice(ENCOURAGE), RED, w//2, int(h*0.60))
            else:
                rc(screen, font_mid, "Nochmal!", RED, w//2, int(h*0.60))
            # IMPROVEMENT 3: pronunciation hint after 2+ wrong attempts
            if attempts_q >= 2:
                rc(screen, font_hint, "Sprich nach:", DIM, w//2, int(h*0.68))
                rc(screen, font_input, task["answer"], YELLOW, w//2, int(h*0.75))
            else:
                rc(screen, font_hint, "LEERTASTE -> nochmal", BLUE, w//2, int(h*0.70))

        elif machine_state == ST_PARENT_CONFIRM:
            heard = f'Gehoert: "{vs["transcript"]}"' if vs["transcript"] else "Nichts gehoert"
            rc(screen, font_hint, heard, YELLOW, w//2, int(h*0.50))
            rc(screen, font_mid, "Richtig gelesen?", WHITE, w//2, int(h*0.58))
            rc(screen, font_mid, "Y = Ja     N = Nochmal", DIM, w//2, int(h*0.66))

        elif machine_state == ST_IDLE:
            if not (WHISPER_OK and transcriber.ready):
                # Whisper not available — show clear error
                if transcriber.loading:
                    rc(screen, font_mid, "Sprach-Erkennung wird geladen...", YELLOW, w//2, int(h*0.52))
                else:
                    rc(screen, font_mid, "Sprach-Erkennung nicht verfuegbar!", RED, w//2, int(h*0.52))
                    rc(screen, font_hint, "Siehe Terminal fuer Details", DIM, w//2, int(h*0.60))
            elif task["mode"] == "voice_parent":
                rc(screen, font_hint, "Kind liest laut vor - dann LEERTASTE", VDIM, w//2, int(h*0.52))
            else:
                rc(screen, font_mid, "LEERTASTE halten", BLUE, w//2, int(h*0.52))
                rc(screen, font_hint, "loslassen = Pruefen", VDIM, w//2, int(h*0.60))

        # ---- Timer bar ----
        tbar = pygame.Rect(int(w*0.10), int(h*0.84), int(w*0.80), int(h*0.025))
        draw_timer_bar(screen, tbar, 1.0 - progress())
        mins_left = max(0, int((SESSION_DURATION_SECONDS - elapsed()) / 60) + 1)
        rl(screen, font_small, f"{mins_left} Min", DIM, int(w*0.10), int(h*0.875))

        if in_warmup():
            wu_frac = clamp(elapsed() / WARMUP_DURATION_SECONDS, 0.0, 1.0)
            draw_bar(screen, pygame.Rect(int(w*0.10), int(h*0.91), int(w*0.80), int(h*0.012)), wu_frac, BLUE)
            rl(screen, font_small, "Aufwaermen", BLUE, int(w*0.10), int(h*0.927))

        # Stars display
        earned  = stars_earned(state.get("total_questions_seen", 0))
        total_s = len(STAR_THRESHOLDS)
        star_display = ("\u2605" * earned) + ("\u2606" * (total_s - earned))
        rr(screen, font_small, star_display, YELLOW, int(w*0.90), int(h*0.875))

        # ESC hint — show countdown if ESC is being held
        if esc_held_since is not None:
            hold_elapsed = t_now - esc_held_since
            remaining = max(0.0, ESC_HOLD_TO_QUIT - hold_elapsed)
            rc(screen, font_small, f"ESC halten: {remaining:.1f}s", RED, w//2, int(h*0.96))
        else:
            rc(screen, font_small, "ESC = ueberspringen  |  ESC halten = beenden", VDIM, w//2, int(h*0.96))

        # IMPROVEMENT 6: Parent debug info bottom-right
        debug_tag = warmup_tag or "-"
        rr(screen, font_tiny, f"diff:{cur_diff():.2f} | tag:{debug_tag} | q:{q_index+1}", VDIM, int(w*0.98), int(h*0.97))

        # ---- Border flash effect ----
        if border_flash_color is not None:
            flash_elapsed = t_now - border_flash_start
            if flash_elapsed < BORDER_FLASH_DURATION:
                alpha_frac = 1.0 - (flash_elapsed / BORDER_FLASH_DURATION)
                draw_border_flash(screen, w, h, border_flash_color, thickness=8, alpha_frac=alpha_frac)
            else:
                border_flash_color = None

        pygame.display.flip()

    # Cleanup
    if vs["rec_active"]:
        recorder.stop()
    attempts_f.close()
    questions_f.close()
    pygame.quit()
    print("[DT] session ended, pygame quit")


if __name__ == "__main__":
    main()
