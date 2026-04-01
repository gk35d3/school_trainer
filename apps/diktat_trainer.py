"""
Diktat-Trainer – German dictation for 2nd-grade students.

Flow per sentence:
  1. Audio plays automatically (macOS 'say' with German voice, pre-recorded to WAV).
  2. Student types what they heard.  Press R or Space to replay at any time.
  3. Enter submits.  Errors are categorised and explained on screen.
  4. Space / Enter → next sentence.
After 10 sentences: summary grouped by error category.
"""

import difflib
import hashlib
import os
import random
import subprocess
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import pygame

from core.trainer_data import append_event, load_recent_events, now_ts

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
FPS = 60
SESSION_SENTENCES = 10
APP_ID = "dictation"
MAX_INPUT_CHARS = 120

# macOS voice – Anna is the best built-in German neural voice.
# Alternatives: Markus, Petra, Yannick
GERMAN_VOICE = "Anna"
SAY_RATE = 120           # words per minute; slower → clearer for children

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dictation_audio")

# ─────────────────────────────────────────────────────────────
# Error taxonomy
# ─────────────────────────────────────────────────────────────
ERROR_LABELS: Dict[str, str] = {
    "großschreibung":  "Großschreibung (Nomen)",
    "kleinschreibung": "Kleinschreibung",
    "umlaut":          "Umlaut (ä, ö, ü)",
    "sz_ss":           "ß oder ss",
    "ie_ei":           "ie oder ei",
    "doppelkonsonant": "Doppelkonsonant (ll, mm …)",
    "ck":              "ck-Schreibung",
    "tz":              "tz-Schreibung",
    "fehlwort":        "Fehlendes Wort",
    "extra_wort":      "Zusätzliches Wort",
    "satzzeichen":     "Satzzeichen (Punkt)",
    "rechtschreibung": "Rechtschreibung",
}

ERROR_TIPS: Dict[str, str] = {
    "großschreibung":  "Nomen (Namenwörter) schreibt man groß.",
    "kleinschreibung": "Dieses Wort schreibt man klein.",
    "umlaut":          "Achte auf ä, ö und ü.",
    "sz_ss":           "Nach kurzem Vokal: ss. Nach langem Vokal oder Diphthong: ß.",
    "ie_ei":           "Verwechslung von ie und ei.",
    "doppelkonsonant": "Nach kurzem Vokal verdoppelt man den Mitlaut.",
    "ck":              "Nach kurzem Vokal schreibt man ck, nicht kk.",
    "tz":              "Nach kurzem Vokal schreibt man tz, nicht z.",
    "fehlwort":        "Ein Wort fehlt im Satz.",
    "extra_wort":      "Dieses Wort gehört nicht in den Satz.",
    "satzzeichen":     "Am Ende des Satzes steht ein Punkt.",
    "rechtschreibung": "Das Wort ist falsch geschrieben.",
}

# ─────────────────────────────────────────────────────────────
# Sentence bank  (120 sentences, ≤ 10 words, 2nd-grade level)
# ─────────────────────────────────────────────────────────────
SENTENCES: List[Dict] = [
    # ── sp / st ──────────────────────────────────────────────
    {"text": "Das Spiel macht großen Spaß im Garten.",        "tags": ["sp_st"]},
    {"text": "Der Spatz sitzt auf dem Ast.",                  "tags": ["sp_st"]},
    {"text": "Wir spielen zusammen auf dem Sportplatz.",      "tags": ["sp_st", "doppelkonsonant"]},
    {"text": "Der Stern leuchtet hell in der Nacht.",         "tags": ["sp_st", "stummes_h"]},
    {"text": "Der Stuhl steht neben dem Tisch.",              "tags": ["sp_st"]},
    {"text": "Der Stein liegt auf der Straße.",               "tags": ["sp_st", "sz_ss"]},
    {"text": "Er springt über die große Pfütze.",             "tags": ["sp_st", "sz_ss", "umlaut"]},
    {"text": "Das Spinnennetz glitzert im Morgentau.",        "tags": ["sp_st", "tz", "doppelkonsonant"]},
    {"text": "Der Storch kommt im Frühling zurück.",          "tags": ["sp_st", "umlaut"]},
    # ── ch ───────────────────────────────────────────────────
    {"text": "Ich lache laut über den Witz.",                 "tags": ["ch", "tz"]},
    {"text": "Das Mädchen liest ein Buch.",                   "tags": ["ch", "umlaut", "großschreibung"]},
    {"text": "Der Koch macht eine leckere Suppe.",            "tags": ["ch", "ck", "doppelkonsonant"]},
    {"text": "Das Licht brennt in der Küche.",                "tags": ["ch", "umlaut", "doppelkonsonant"]},
    {"text": "Wir sprechen leise in der Schule.",             "tags": ["ch", "sp_st"]},
    {"text": "Das Buch liegt auf dem Tisch.",                 "tags": ["ch"]},
    {"text": "Die Geschichte ist sehr spannend.",             "tags": ["ch", "sp_st", "doppelkonsonant"]},
    {"text": "Das Märchen handelt von einer Hexe.",           "tags": ["ch", "umlaut"]},
    {"text": "Die Schokolade schmeckt sehr gut.",             "tags": ["ch"]},
    # ── Herbst ───────────────────────────────────────────────
    {"text": "Die Blätter fallen bunt vom Baum.",             "tags": ["umlaut", "großschreibung"]},
    {"text": "Der Kürbis ist groß und orange.",               "tags": ["umlaut", "sz_ss", "großschreibung"]},
    {"text": "Im Herbst sind die Äpfel reif.",                "tags": ["umlaut"]},
    {"text": "Die Kastanie liegt im nassen Laub.",            "tags": ["doppelkonsonant", "großschreibung"]},
    {"text": "Wir gehen im Herbst spazieren.",                "tags": ["sp_st"]},
    {"text": "Die bunten Blätter rascheln im Wind.",          "tags": ["umlaut"]},
    {"text": "Der Nebel liegt über den Feldern.",             "tags": ["großschreibung", "stummes_h"]},
    # ── Winter ───────────────────────────────────────────────
    {"text": "Der Schneemann hat eine rote Nase.",            "tags": ["doppelkonsonant", "großschreibung"]},
    {"text": "Die Kinder rodeln den Hang hinunter.",          "tags": ["großschreibung"]},
    {"text": "Es ist kalt und das Eis glitzert.",             "tags": ["ei", "tz"]},
    {"text": "Wir trinken heiße Schokolade im Winter.",       "tags": ["sz_ss", "ch"]},
    {"text": "Die Schneeflocken fallen leise vom Himmel.",    "tags": ["doppelkonsonant", "stummes_h"]},
    {"text": "Im Winter schläft der Bär im Wald.",            "tags": ["umlaut"]},
    {"text": "Die Straße ist glatt und eisig.",               "tags": ["sp_st", "sz_ss", "ei"]},
    {"text": "Die Kinder bauen einen Schneemann.",            "tags": ["doppelkonsonant"]},
    # ── Frühling ─────────────────────────────────────────────
    {"text": "Die Blüten duften schön im Frühling.",          "tags": ["umlaut", "stummes_h"]},
    {"text": "Der Schmetterling fliegt über die Blumen.",     "tags": ["ie", "doppelkonsonant"]},
    {"text": "Es wird warm und die Vögel singen.",            "tags": ["umlaut", "ng"]},
    {"text": "Die Knospen öffnen sich in der Sonne.",         "tags": ["umlaut", "doppelkonsonant"]},
    {"text": "Der Regenwurm kriecht durch die Erde.",         "tags": ["großschreibung", "ie"]},
    {"text": "Im Frühling singen die Vögel schön.",           "tags": ["umlaut", "ng"]},
    # ── Sommer ───────────────────────────────────────────────
    {"text": "Wir schwimmen gerne im See.",                   "tags": ["doppelkonsonant", "ie"]},
    {"text": "Die Sonne scheint hell und heiß.",              "tags": ["doppelkonsonant", "sz_ss", "ei"]},
    {"text": "Wir bauen eine Sandburg am Strand.",            "tags": ["sp_st", "großschreibung"]},
    {"text": "Im Urlaub grillen wir im Garten.",              "tags": ["doppelkonsonant", "großschreibung"]},
    {"text": "Die Kinder planschen im kühlen Wasser.",        "tags": ["umlaut"]},
    {"text": "Das Eis ist süß und kalt.",                     "tags": ["ei", "umlaut", "sz_ss"]},
    {"text": "Am Strand findet sie eine Muschel.",            "tags": ["sp_st"]},
    # ── eu ───────────────────────────────────────────────────
    {"text": "Das neue Fahrrad ist blau und schön.",          "tags": ["eu", "stummes_h", "umlaut"]},
    {"text": "Die Eule sitzt auf dem alten Baum.",            "tags": ["eu", "großschreibung"]},
    {"text": "Wir freuen uns über das Geschenk.",             "tags": ["eu"]},
    {"text": "Das Feuer brennt im Kamin.",                    "tags": ["eu", "doppelkonsonant"]},
    {"text": "Das Flugzeug fliegt hoch über uns.",            "tags": ["eu", "tz", "ie"]},
    {"text": "Die Laterne leuchtet hell in der Nacht.",       "tags": ["eu", "stummes_h"]},
    # ── ei ───────────────────────────────────────────────────
    {"text": "Das Kleid ist weiß mit kleinen Punkten.",       "tags": ["ei", "sz_ss"]},
    {"text": "Wir reiten auf dem großen Pferd.",              "tags": ["ei", "sz_ss"]},
    {"text": "Der kleine Hund heißt Bello.",                  "tags": ["ei", "sz_ss"]},
    {"text": "Die Reise war lang und schön.",                 "tags": ["ei", "umlaut"]},
    {"text": "Er zeigt auf den bunten Stein.",                "tags": ["ei"]},
    {"text": "Das Eis leuchtet weiß im Sonnenlicht.",         "tags": ["ei", "sz_ss"]},
    # ── ie ───────────────────────────────────────────────────
    {"text": "Das Tier lebt im tiefen Wald.",                 "tags": ["ie"]},
    {"text": "Wir spielen Fußball auf der Wiese.",            "tags": ["ie", "sp_st", "sz_ss"]},
    {"text": "Die Fliege summt um das Essen.",                "tags": ["ie", "doppelkonsonant"]},
    {"text": "Er schreibt einen langen Brief an Oma.",        "tags": ["ie", "ei"]},
    {"text": "Vier Kinder sitzen im Kreis.",                  "tags": ["ie", "ei", "tz"]},
    {"text": "Das Spiel macht Spaß auf der Wiese.",           "tags": ["ie", "sp_st", "sz_ss"]},
    {"text": "Die Fliege sitzt auf dem Tisch.",               "tags": ["ie"]},
    # ── ng / nk ──────────────────────────────────────────────
    {"text": "Die Schlange liegt in der Sonne.",              "tags": ["ng", "doppelkonsonant"]},
    {"text": "Wir singen ein schönes Lied zusammen.",         "tags": ["ng", "ie", "umlaut"]},
    {"text": "Er trinkt Wasser aus der Trinkflasche.",        "tags": ["nk"]},
    {"text": "Im Dunkeln leuchtet die Laterne.",              "tags": ["nk", "eu"]},
    {"text": "Der Junge springt ins kühle Wasser.",           "tags": ["ng", "sp_st", "umlaut"]},
    {"text": "Ich danke dir für das Geschenk.",               "tags": ["nk", "umlaut"]},
    {"text": "Die Bank steht unter dem Baum.",                "tags": ["nk"]},
    {"text": "Der Schmetterling hat bunte Flügel.",           "tags": ["ng", "umlaut", "doppelkonsonant"]},
    # ── ck ───────────────────────────────────────────────────
    {"text": "Die Jacke hängt an der Tür.",                   "tags": ["ck", "umlaut"]},
    {"text": "Die Brücke führt über den Fluss.",              "tags": ["ck", "umlaut", "stummes_h", "doppelkonsonant"]},
    {"text": "Er trägt einen schweren Rucksack.",             "tags": ["ck", "umlaut"]},
    {"text": "Der Zucker ist süß und weiß.",                  "tags": ["ck", "umlaut", "sz_ss", "ei"]},
    {"text": "Das Glück lacht dem Mutigen.",                  "tags": ["ck", "umlaut", "ch"]},
    {"text": "Er schaut zurück auf den Weg.",                 "tags": ["ck"]},
    {"text": "Die Mücke sticht den Arm.",                     "tags": ["ck", "umlaut", "sp_st"]},
    # ── tz ───────────────────────────────────────────────────
    {"text": "Es blitzt und donnert draußen.",                "tags": ["tz", "sz_ss", "doppelkonsonant"]},
    {"text": "Die Katze schläft auf dem Sofa.",               "tags": ["tz", "umlaut"]},
    {"text": "Er trägt eine bunte Mütze.",                    "tags": ["tz", "umlaut"]},
    {"text": "Plötzlich regnete es sehr stark.",              "tags": ["tz", "umlaut", "sp_st"]},
    {"text": "Der Platz ist groß und grün.",                  "tags": ["tz", "sz_ss", "umlaut"]},
    {"text": "Der Witz macht alle zum Lachen.",               "tags": ["tz", "ch"]},
    # ── ß / ss ───────────────────────────────────────────────
    {"text": "Die Straße ist lang und gerade.",               "tags": ["sz_ss", "sp_st"]},
    {"text": "Der Fußball liegt auf der Wiese.",              "tags": ["sz_ss", "ie"]},
    {"text": "Es ist heiß im Sommer.",                        "tags": ["sz_ss", "ei"]},
    {"text": "Ich weiß die Antwort auf die Frage.",           "tags": ["sz_ss", "ei"]},
    {"text": "Der Fluss fließt durch die Stadt.",             "tags": ["sz_ss", "ie", "doppelkonsonant", "sp_st"]},
    # ── Doppelkonsonant ──────────────────────────────────────
    {"text": "Die Sonne scheint den ganzen Tag.",             "tags": ["doppelkonsonant"]},
    {"text": "Der Ball rollt ins Tor.",                       "tags": ["doppelkonsonant"]},
    {"text": "Wir essen zusammen zu Mittag.",                 "tags": ["doppelkonsonant"]},
    {"text": "Im Zimmer brennt das Licht.",                   "tags": ["doppelkonsonant"]},
    {"text": "Der Löffel liegt neben dem Teller.",            "tags": ["doppelkonsonant", "umlaut"]},
    {"text": "Die Tasse ist voll mit Tee.",                   "tags": ["doppelkonsonant", "ie"]},
    {"text": "Sie rennt schnell zur Schule.",                 "tags": ["doppelkonsonant"]},
    # ── Umlaut ───────────────────────────────────────────────
    {"text": "Die Äpfel hängen rot am Baum.",                 "tags": ["umlaut"]},
    {"text": "Das Mädchen trägt ein buntes Kleid.",           "tags": ["umlaut", "ei"]},
    {"text": "Die Bäume im Garten sind groß.",                "tags": ["umlaut", "sz_ss"]},
    {"text": "Wir zählen die Punkte zusammen.",               "tags": ["umlaut"]},
    {"text": "Der Bär schläft den ganzen Winter.",            "tags": ["umlaut"]},
    {"text": "Die Vögel singen schön im Frühling.",           "tags": ["umlaut", "ng"]},
    {"text": "Wir räumen das Zimmer auf.",                    "tags": ["umlaut"]},
    {"text": "Die Mäuse verstecken sich im Keller.",          "tags": ["umlaut", "ck", "doppelkonsonant"]},
    # ── Stummes h ────────────────────────────────────────────
    {"text": "Wir fahren mit dem Zug in die Stadt.",          "tags": ["stummes_h", "sp_st"]},
    {"text": "Er zahlt für die Bücher.",                      "tags": ["stummes_h", "umlaut"]},
    {"text": "Das Reh springt über den Bach.",                "tags": ["stummes_h", "sp_st", "ch"]},
    {"text": "Die Uhr zeigt zehn Uhr an.",                    "tags": ["stummes_h"]},
    {"text": "Der Lehrer erklärt die Aufgabe.",               "tags": ["stummes_h", "umlaut"]},
    {"text": "Wir wohnen in einem großen Haus.",              "tags": ["stummes_h", "sz_ss"]},
    {"text": "Er nimmt das Buch in die Hand.",                "tags": ["stummes_h"]},
    # ── Großschreibung focus ─────────────────────────────────
    {"text": "Der Hund bellt laut im Garten.",                "tags": ["großschreibung"]},
    {"text": "Das Buch liegt auf dem Schreibtisch.",          "tags": ["großschreibung", "sp_st"]},
    {"text": "Wir lernen in der Schule lesen.",               "tags": ["großschreibung"]},
    {"text": "Der Apfel ist rot und saftig.",                 "tags": ["großschreibung"]},
    {"text": "Die Sonne scheint durch das Fenster.",          "tags": ["großschreibung", "stummes_h"]},
    {"text": "Das Kind spielt mit dem Ball.",                 "tags": ["großschreibung"]},
    {"text": "Der Vogel singt auf dem Ast.",                  "tags": ["großschreibung"]},
    # ── v ────────────────────────────────────────────────────
    {"text": "Der Vogel singt ein schönes Lied.",             "tags": ["großschreibung", "ie", "umlaut"]},
    {"text": "Unser Vater fährt mit dem Auto.",               "tags": ["stummes_h"]},
    {"text": "Im November wird es kalt und dunkel.",          "tags": ["doppelkonsonant"]},
    {"text": "Der Vulkan ist sehr groß und heiß.",            "tags": ["sz_ss", "ei"]},
    # ── qu ───────────────────────────────────────────────────
    {"text": "Die Qualle lebt im tiefen Meer.",               "tags": ["ie"]},
    {"text": "Das Aquarium steht im Wohnzimmer.",             "tags": []},
    # ── Gemischt ─────────────────────────────────────────────
    {"text": "Die Katze jagt die Maus im Garten.",            "tags": ["großschreibung"]},
    {"text": "Er liest gerne spannende Bücher.",              "tags": ["ie", "umlaut", "sp_st", "doppelkonsonant"]},
    {"text": "Die Kinder tanzen fröhlich zur Musik.",         "tags": ["umlaut"]},
    {"text": "Wir schreiben einen Brief an Oma.",             "tags": ["ie", "ei"]},
    {"text": "Er rechnet die Aufgabe richtig.",               "tags": ["ie", "ch"]},
    {"text": "Im Herbst ist es oft neblig und nass.",         "tags": ["doppelkonsonant"]},
    {"text": "Der Schrank steht im Schlafzimmer.",            "tags": ["sp_st", "ch"]},
    {"text": "Der Elefant ist viel größer als der Hund.",     "tags": ["sz_ss", "umlaut"]},
    {"text": "Sie läuft schneller als ihr Bruder.",           "tags": ["umlaut", "doppelkonsonant"]},
    {"text": "Er schläft noch und träumt schön.",             "tags": ["umlaut", "eu", "stummes_h"]},
    {"text": "Das Mädchen malt ein buntes Bild.",             "tags": ["umlaut", "großschreibung"]},
    {"text": "Die Kinder lachen laut auf dem Spielplatz.",    "tags": ["ch", "sp_st"]},
    {"text": "Er schenkt ihr eine schöne Blume.",             "tags": ["umlaut"]},
    {"text": "Das Eichhörnchen klettert auf den Baum.",       "tags": ["ei", "ch", "umlaut"]},
]

# ─────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────

def _audio_path(text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return os.path.join(AUDIO_DIR, f"{h}.wav")


def ensure_audio(text: str) -> Optional[str]:
    """Pre-record text to a WAV file via macOS 'say'. Returns path or None."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    wav_path = _audio_path(text)
    if os.path.exists(wav_path):
        return wav_path
    aiff_path = wav_path.replace(".wav", ".aiff")
    try:
        subprocess.run(
            ["say", "-v", GERMAN_VOICE, "-r", str(SAY_RATE), "-o", aiff_path, "--", text],
            check=True, capture_output=True, timeout=20,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path],
            check=True, capture_output=True, timeout=15,
        )
    except Exception:
        # Fallback: try direct output format flag (older macOS)
        try:
            subprocess.run(
                ["say", "-v", GERMAN_VOICE, "-r", str(SAY_RATE),
                 "-o", wav_path, "--data-format=LEF32@22050", "--", text],
                check=True, capture_output=True, timeout=20,
            )
        except Exception:
            return None
    finally:
        if os.path.exists(aiff_path):
            try:
                os.remove(aiff_path)
            except OSError:
                pass
    return wav_path if os.path.exists(wav_path) else None


def play_audio(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception:
        pass


def stop_audio() -> None:
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass


def audio_playing() -> bool:
    try:
        return bool(pygame.mixer.music.get_busy())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip_punct(w: str) -> str:
    """Remove leading/trailing punctuation, keep the word body."""
    return w.strip(".,!?;:»«")


def _word_body(w: str) -> str:
    """NFC-normalised, lowercase body with punctuation stripped."""
    return _nfc(_strip_punct(w).lower())


def _classify_word_error(exp_word: str, act_word: str) -> str:
    """
    Classify the error between one expected word and one actual word.
    Both inputs are the raw (original-case) words from the sentence.

    Decision tree
    ─────────────
    1. Bodies identical (after lowercase+NFC+strip_punct)?
       → capitalisation error (groß/kleinschreibung)
    2. ß ↔ ss?   → sz_ss
    3. Umlaut dropped/swapped?   → umlaut
    4. ie ↔ ei?   → ie_ei
    5. Double consonant missing?   → doppelkonsonant
    6. ck ↔ k?   → ck
    7. tz ↔ z?   → tz
    8. Anything else   → rechtschreibung
    """
    eb = _word_body(exp_word)   # lowercase body of expected
    ab = _word_body(act_word)   # lowercase body of actual

    # ── 1. Capitalisation ──────────────────────────────────────
    # When lowercase bodies are identical the words are spelled correctly;
    # any remaining difference is purely a capitalisation error.
    # We never fall through to the spelling checks in this branch.
    if eb == ab:
        ew_s = _strip_punct(exp_word)
        aw_s = _strip_punct(act_word)
        if ew_s == aw_s:
            # Truly identical – caller should not have called this
            return "rechtschreibung"
        # Expected uppercase first letter, student wrote lowercase → Noun not capitalised
        if ew_s[:1].isupper() and aw_s[:1].islower():
            return "großschreibung"
        # Expected lowercase first letter, student capitalised → wrong capitalisation
        if ew_s[:1].islower() and aw_s[:1].isupper():
            return "kleinschreibung"
        # Any other case mismatch (e.g. "See" → "SEE", or mixed caps) → too many capitals
        return "großschreibung"

    # ── 2. ß ↔ ss ─────────────────────────────────────────────
    if eb.replace("ß", "ss") == ab or ab.replace("ß", "ss") == eb:
        return "sz_ss"

    # ── 3. Umlaut ─────────────────────────────────────────────
    def drop_uml(s: str) -> str:
        return s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    if drop_uml(eb) == drop_uml(ab):
        return "umlaut"

    # ── 4. ie / ei ────────────────────────────────────────────
    if eb.replace("ie", "ei") == ab or eb.replace("ei", "ie") == ab:
        return "ie_ei"

    # ── 5. Double consonant ───────────────────────────────────
    for cc in ("ll", "mm", "nn", "pp", "rr", "ss", "tt", "ff", "bb", "gg", "dd"):
        if eb.replace(cc, cc[0]) == ab or ab.replace(cc, cc[0]) == eb:
            return "doppelkonsonant"

    # ── 6. ck ─────────────────────────────────────────────────
    if eb.replace("ck", "k") == ab or ab.replace("ck", "k") == eb:
        return "ck"

    # ── 7. tz ─────────────────────────────────────────────────
    if eb.replace("tz", "z") == ab or ab.replace("tz", "z") == eb:
        return "tz"

    return "rechtschreibung"


def _error(cat: str, ew: str, aw: str) -> Dict:
    return {"category": cat, "expected_word": ew, "actual_word": aw,
            "tip": ERROR_TIPS.get(cat, "")}


def analyse_errors(expected: str, actual: str) -> List[Dict]:
    """
    Compare every word in `expected` against `actual` and return a list of
    error dicts.

    Strategy
    ────────
    • Align words using lowercase+stripped keys so that capitalisation
      differences do not disturb structural alignment (missing/extra words).
    • For every aligned pair – including those difflib considers 'equal' –
      compare the ORIGINAL (case-preserved) words to catch capitalisation
      errors that are invisible to the lowercase aligner.
    • Detect missing final period separately.
    """
    errors: List[Dict] = []

    exp_words = expected.strip().split()
    act_words = actual.strip().split() if actual.strip() else []

    # Lowercase+stripped keys used only for alignment, never for error detection.
    sm = difflib.SequenceMatcher(
        None,
        [_word_body(w) for w in exp_words],
        [_word_body(w) for w in act_words],
        autojunk=False,
    )

    for tag, i1, i2, j1, j2 in sm.get_opcodes():

        if tag == "equal":
            # Difflib sees these as structurally identical (same lowercase body).
            # Still compare the ORIGINAL words: capitalisation may differ.
            for k in range(i2 - i1):
                ew = exp_words[i1 + k]
                aw = act_words[j1 + k]
                # _strip_punct preserves case → detects cap differences
                if _strip_punct(ew) != _strip_punct(aw):
                    errors.append(_error(_classify_word_error(ew, aw), ew, aw))

        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                ew = exp_words[i1 + k] if (i1 + k) < i2 else ""
                aw = act_words[j1 + k] if (j1 + k) < j2 else ""
                if not ew:
                    errors.append(_error("extra_wort", "", aw))
                elif not aw:
                    errors.append(_error("fehlwort", ew, ""))
                else:
                    errors.append(_error(_classify_word_error(ew, aw), ew, aw))

        elif tag == "delete":
            for i in range(i1, i2):
                errors.append(_error("fehlwort", exp_words[i], ""))

        elif tag == "insert":
            for j in range(j1, j2):
                errors.append(_error("extra_wort", "", act_words[j]))

    # Missing final period (checked separately to avoid double-counting).
    if expected.rstrip().endswith(".") and not actual.rstrip().endswith("."):
        errors.append(_error("satzzeichen", ".", ""))

    return errors


def wrong_word_indices(expected: str, actual: str) -> Set[int]:
    """Return 0-based word indices in `actual` that differ from `expected`."""
    exp_words = expected.strip().split()
    act_words = actual.strip().split() if actual.strip() else []
    bad: Set[int] = set()
    sm = difflib.SequenceMatcher(
        None,
        [_word_body(w) for w in exp_words],
        [_word_body(w) for w in act_words],
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                # Flag if original forms differ (includes capitalisation)
                if _strip_punct(exp_words[i1 + k]) != _strip_punct(act_words[j1 + k]):
                    bad.add(j1 + k)
        else:
            for j in range(j1, j2):
                bad.add(j)
    return bad


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────

FONT_CACHE: Dict[int, pygame.font.Font] = {}


def font(size: int) -> pygame.font.Font:
    if size not in FONT_CACHE:
        for name in ("DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"):
            f = pygame.font.SysFont(name, size)
            if f:
                FONT_CACHE[size] = f
                return f
        FONT_CACHE[size] = pygame.font.SysFont(None, size)
    return FONT_CACHE[size]


def to_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def wrap_text(f: pygame.font.Font, text: str, max_w: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if f.size(test)[0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_text_block(
    screen, f, text: str, x: int, y: int, w: int,
    color=(240, 240, 240), line_gap: int = 6,
):
    lines = wrap_text(f, to_nfc(text), w - 20)
    lh = f.get_linesize() + line_gap
    cy = y + 10
    for line in lines:
        surf = f.render(line, True, color)
        screen.blit(surf, surf.get_rect(midtop=(x + w // 2, cy)))
        cy += lh


def draw_answer_with_errors(
    screen, f, text: str, x: int, y: int, w: int,
    bad_indices: Set[int], default_color=(240, 240, 240), line_gap: int = 6,
):
    words = text.split()
    if not words:
        return
    left, right = x + 10, x + w - 10
    cx, cy = left, y + 10
    lh = f.get_linesize() + line_gap
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        tw = f.size(to_nfc(token))[0]
        if cx + tw > right:
            cx, cy = left, cy + lh
        col = (240, 80, 80) if i in bad_indices else default_color
        surf = f.render(to_nfc(token), True, col)
        screen.blit(surf, (cx, cy))
        cx += tw


def draw_progress_bar(screen, rect, frac: float, color=(100, 180, 100)):
    pygame.draw.rect(screen, (60, 60, 60), rect, border_radius=4)
    filled = rect.copy()
    filled.width = max(0, int(rect.width * frac))
    if filled.width:
        pygame.draw.rect(screen, color, filled, border_radius=4)


def draw_speaker_icon(screen, cx: int, cy: int, size: int, active: bool):
    """Simple speaker shape. cx/cy = centre of the icon."""
    col = (120, 200, 255) if active else (80, 90, 120)
    # Body rectangle
    body = pygame.Rect(cx - size // 2, cy - size // 3, size // 2, size * 2 // 3)
    pygame.draw.rect(screen, col, body)
    # Cone
    pts = [
        (cx, cy - size // 3),
        (cx + size // 2, cy - size * 2 // 3),
        (cx + size // 2, cy + size * 2 // 3),
        (cx, cy + size // 3),
    ]
    pygame.draw.polygon(screen, col, pts)
    # Sound wave arc (always drawn; brighter when active)
    wave_col = (160, 220, 255) if active else (100, 110, 140)
    r = int(size * 0.75)
    pygame.draw.arc(
        screen, wave_col,
        pygame.Rect(cx + size // 4, cy - r // 2, r, r),
        -0.7, 0.7, max(2, size // 10),
    )


def _speaker_rect(sw: int, sh: int, base: int) -> pygame.Rect:
    """Clickable area of the speaker icon on the input screen."""
    margin = int(sw * 0.06)
    row_cy = int(sh * 0.46)
    half = int(base * 1.1)
    spk_cx = margin + half
    return pygame.Rect(spk_cx - half, row_cy - half, half * 2, half * 2)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    pygame.display.set_caption("Diktat-Trainer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    sw, sh = screen.get_size()
    clock = pygame.time.Clock()

    base = max(22, min(sw, sh) // 11)
    f_title  = font(int(base * 1.0))
    f_main   = font(int(base * 0.85))
    f_input  = font(int(base * 0.80))
    f_small  = font(int(base * 0.38))
    f_hint   = font(int(base * 0.32))

    BG       = (20,  22,  35)
    CARD     = (32,  36,  54)
    ACCENT   = (80, 160, 255)
    GREEN    = (80, 200, 100)
    RED      = (240, 80,  80)
    YELLOW   = (255, 220,  60)
    WHITE    = (240, 240, 240)
    GRAY     = (130, 130, 150)

    session_id = f"dictation_{int(now_ts())}"
    chosen = random.sample(SENTENCES, min(SESSION_SENTENCES, len(SENTENCES)))

    # ── Pre-record audio ─────────────────────────────────────
    audio_paths: List[Optional[str]] = [None] * len(chosen)
    screen.fill(BG)
    msg = font(int(base * 0.7)).render(to_nfc("Vorbereitung …"), True, WHITE)
    screen.blit(msg, msg.get_rect(center=(sw // 2, sh // 2)))
    pygame.display.flip()

    for i, s in enumerate(chosen):
        # Pump events so window stays responsive
        pygame.event.pump()
        audio_paths[i] = ensure_audio(s["text"])
        # Draw progress
        screen.fill(BG)
        screen.blit(msg, msg.get_rect(center=(sw // 2, sh // 2 - base)))
        bar = pygame.Rect(sw // 4, sh // 2 + 10, sw // 2, base // 2)
        draw_progress_bar(screen, bar, (i + 1) / len(chosen))
        pygame.display.flip()

    # ── Session state ─────────────────────────────────────────
    idx          = 0          # current sentence index
    state        = "input"    # "input" | "feedback" | "copy" | "summary"
    user_text    = ""
    copy_text    = ""         # typed text in the copy-practice screen
    errors: List[Dict]  = []
    had_errors   = False      # explicit flag: were there errors this sentence?
    bad_idx: Set[int]   = set()
    session_errors: List[Dict] = []   # all errors across session

    append_event({
        "type": "session_start", "app_id": APP_ID,
        "session_id": session_id, "ts": now_ts(),
        "num_sentences": len(chosen),
    })

    # Auto-play first sentence
    play_audio(audio_paths[idx])

    running = True
    while running:
        clock.tick(FPS)
        playing = audio_playing()
        spk_rect = _speaker_rect(sw, sh, base)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if state == "input" and spk_rect.collidepoint(ev.pos):
                    stop_audio()
                    play_audio(audio_paths[idx])

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                elif state == "input":
                    if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        errors    = analyse_errors(chosen[idx]["text"], user_text)
                        had_errors = len(errors) > 0
                        bad_idx   = wrong_word_indices(chosen[idx]["text"], user_text)
                        stop_audio()
                        state = "feedback"
                        append_event({
                            "type": "attempt", "app_id": APP_ID,
                            "session_id": session_id,
                            "sentence": chosen[idx]["text"],
                            "user_text": user_text,
                            "error_count": len(errors),
                            "correct": not had_errors,
                            "error_categories": list({e["category"] for e in errors}),
                            "ts": now_ts(),
                        })
                        session_errors.extend(errors)

                    elif ev.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]

                    elif ev.unicode and len(user_text) < MAX_INPUT_CHARS:
                        user_text += ev.unicode

                elif state == "feedback":
                    if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        if had_errors:
                            # Errors this sentence → show copy-practice screen first
                            copy_text = ""
                            state = "copy"
                        else:
                            idx += 1
                            if idx >= len(chosen):
                                state = "summary"
                                append_event({
                                    "type": "session_end", "app_id": APP_ID,
                                    "session_id": session_id,
                                    "total_errors": len(session_errors),
                                    "ts": now_ts(),
                                })
                            else:
                                user_text = ""
                                errors = []
                                had_errors = False
                                bad_idx = set()
                                state = "input"
                                play_audio(audio_paths[idx])

                elif state == "copy":
                    if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        idx += 1
                        if idx >= len(chosen):
                            state = "summary"
                            append_event({
                                "type": "session_end", "app_id": APP_ID,
                                "session_id": session_id,
                                "total_errors": len(session_errors),
                                "ts": now_ts(),
                            })
                        else:
                            user_text = ""
                            copy_text = ""
                            errors = []
                            had_errors = False
                            bad_idx = set()
                            state = "input"
                            play_audio(audio_paths[idx])
                    elif ev.key == pygame.K_BACKSPACE:
                        copy_text = copy_text[:-1]
                    elif ev.unicode and len(copy_text) < MAX_INPUT_CHARS:
                        copy_text += ev.unicode

                elif state == "summary":
                    if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                  pygame.K_SPACE, pygame.K_q):
                        running = False

        # ── Draw ─────────────────────────────────────────────
        screen.fill(BG)

        if state == "input":
            _draw_input(screen, sw, sh, chosen, idx, user_text, playing,
                        spk_rect, f_title, f_input, f_hint,
                        CARD, ACCENT, WHITE, GRAY, base)

        elif state == "feedback":
            _draw_feedback(screen, sw, sh, chosen, idx, user_text, errors, bad_idx,
                           f_title, f_main, f_input, f_small, f_hint,
                           CARD, ACCENT, GREEN, RED, WHITE, GRAY, YELLOW, base)

        elif state == "copy":
            _draw_copy(screen, sw, sh, chosen, idx, copy_text,
                       f_title, f_main, f_input, f_hint,
                       CARD, ACCENT, GREEN, WHITE, GRAY, base)

        elif state == "summary":
            _draw_summary(screen, sw, sh, chosen, session_errors,
                          f_title, f_main, f_small, f_hint,
                          CARD, ACCENT, GREEN, RED, WHITE, GRAY, YELLOW, base)

        pygame.display.flip()

    pygame.quit()


# ─────────────────────────────────────────────────────────────
# Draw helpers  (broken out to keep main() readable)
# ─────────────────────────────────────────────────────────────

def _draw_input(screen, sw, sh, chosen, idx, user_text, playing,
                spk_rect, f_title, f_input, f_hint,
                CARD, ACCENT, WHITE, GRAY, base):

    margin = int(sw * 0.06)
    content_w = sw - 2 * margin

    # Header
    header = to_nfc(f"Diktat  –  Satz {idx + 1} von {len(chosen)}")
    hs = f_title.render(header, True, ACCENT)
    screen.blit(hs, hs.get_rect(midtop=(sw // 2, int(sh * 0.06))))

    # Speaker icon (clickable, to the left of input box)
    spk_cx = spk_rect.centerx
    spk_cy = spk_rect.centery
    spk_size = int(base * 0.8)
    draw_speaker_icon(screen, spk_cx, spk_cy, spk_size, playing)

    # Input box (right of speaker)
    box_x = spk_rect.right + int(sw * 0.02)
    box_y = int(sh * 0.34)
    box_h = int(sh * 0.24)
    box_w = sw - margin - box_x
    box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
    pygame.draw.rect(screen, CARD, box_rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, box_rect, 2, border_radius=10)

    display_text = to_nfc(user_text + "│")
    lines = wrap_text(f_input, display_text, box_w - 24)
    lh = f_input.get_linesize() + 4
    ty = box_y + 14
    for line in lines:
        ls = f_input.render(line, True, WHITE)
        screen.blit(ls, (box_x + 12, ty))
        ty += lh

    # Minimal hint
    hint = to_nfc("Eingabe = fertig")
    hs2 = f_hint.render(hint, True, GRAY)
    screen.blit(hs2, hs2.get_rect(midbottom=(sw // 2, sh - int(sh * 0.03))))

    # Progress bar
    bar_h = max(6, sh // 80)
    bar_rect = pygame.Rect(margin, sh - int(sh * 0.08), content_w, bar_h)
    draw_progress_bar(screen, bar_rect, idx / len(chosen))


def _text_block_height(f, text: str, max_w: int, line_gap: int = 6) -> int:
    lines = wrap_text(f, to_nfc(text), max_w)
    return len(lines) * (f.get_linesize() + line_gap)


def _draw_feedback(screen, sw, sh, chosen, idx, user_text, errors, bad_idx,
                   f_title, f_main, f_input, f_small, f_hint,
                   CARD, ACCENT, GREEN, RED, WHITE, GRAY, YELLOW, base):

    margin = int(sw * 0.06)
    content_w = sw - 2 * margin

    if not errors:
        # ── Correct: show sentence large and centred ──────────
        hs = f_title.render(to_nfc("Richtig!"), True, GREEN)
        screen.blit(hs, hs.get_rect(midtop=(sw // 2, int(sh * 0.05))))

        # Vertically centre the sentence
        lh = f_main.get_linesize() + 6
        lines = wrap_text(f_main, to_nfc(chosen[idx]["text"]), content_w - 40)
        total_h = len(lines) * lh
        start_y = (sh - total_h) // 2
        for i, line in enumerate(lines):
            surf = f_main.render(line, True, GREEN)
            screen.blit(surf, surf.get_rect(midtop=(sw // 2, start_y + i * lh)))

        cont = f_hint.render(to_nfc("Leertaste = weiter"), True, GRAY)
        screen.blit(cont, cont.get_rect(midbottom=(sw // 2, sh - int(sh * 0.02))))
        return

    # ── Errors: header ────────────────────────────────────────
    hs = f_title.render(to_nfc(f"{len(errors)} Fehler"), True, RED)
    screen.blit(hs, hs.get_rect(midtop=(sw // 2, int(sh * 0.03))))

    cy = int(sh * 0.14)
    card_h = int(sh * 0.13)

    # Correct sentence
    pygame.draw.rect(screen, CARD, pygame.Rect(margin, cy, content_w, card_h), border_radius=8)
    draw_text_block(screen, f_main, chosen[idx]["text"], margin, cy, content_w, color=GREEN)
    cy += card_h + int(sh * 0.015)

    # Student's answer
    ans_card = pygame.Rect(margin, cy, content_w, card_h)
    pygame.draw.rect(screen, CARD, ans_card, border_radius=8)
    draw_answer_with_errors(screen, f_main, user_text or "–",
                            margin, cy, content_w, bad_idx, WHITE)
    cy += card_h + int(sh * 0.015)

    # Error explanations – each entry may wrap onto multiple lines
    remaining_h = sh - cy - int(sh * 0.08)
    err_card = pygame.Rect(margin, cy, content_w, remaining_h)
    pygame.draw.rect(screen, CARD, err_card, border_radius=8)
    ey = cy + 10
    max_y = cy + remaining_h - f_hint.get_linesize() - 4
    seen: Set[str] = set()
    lh_small = f_small.get_linesize() + 4
    lh_hint  = f_hint.get_linesize() + 2

    for e in errors:
        if ey > max_y:
            break
        cat = e["category"]
        label = ERROR_LABELS.get(cat, cat)
        ew = e.get("expected_word", "")
        aw = e.get("actual_word", "")
        key = (cat, ew, aw)
        if key in seen:
            continue
        seen.add(key)

        # Category label
        col_label = f_small.render(to_nfc(label + ":"), True, YELLOW)
        screen.blit(col_label, (margin + 10, ey))
        ey += lh_small

        # Detail line(s) – wrapped; strip trailing punct from displayed words.
        ew_d = _strip_punct(ew) if ew else ""
        aw_d = _strip_punct(aw) if aw else ""
        if ew_d and aw_d:
            detail = f"»{ew_d}« statt »{aw_d}«  –  {e['tip']}"
        elif ew_d:
            detail = f"»{ew_d}« fehlt  –  {e['tip']}"
        else:
            detail = f"»{aw_d}« ist zu viel  –  {e['tip']}"
        for dl in wrap_text(f_hint, to_nfc(detail), content_w - 30):
            if ey > max_y:
                break
            ds = f_hint.render(dl, True, WHITE)
            screen.blit(ds, (margin + 18, ey))
            ey += lh_hint
        ey += 4  # gap between errors

    cont = f_hint.render(to_nfc("Leertaste = weiter"), True, GRAY)
    screen.blit(cont, cont.get_rect(midbottom=(sw // 2, sh - int(sh * 0.02))))


def _draw_copy(screen, sw, sh, chosen, idx, copy_text,
               f_title, f_main, f_input, f_hint,
               CARD, ACCENT, GREEN, WHITE, GRAY, base):
    """Practice screen: correct sentence shown above, student types it."""
    margin = int(sw * 0.06)
    content_w = sw - 2 * margin

    hs = f_title.render(to_nfc("Schreib es richtig ab:"), True, ACCENT)
    screen.blit(hs, hs.get_rect(midtop=(sw // 2, int(sh * 0.05))))

    # Correct sentence card
    lh = f_main.get_linesize() + 6
    lines = wrap_text(f_main, to_nfc(chosen[idx]["text"]), content_w - 24)
    card_h = max(int(sh * 0.13), len(lines) * lh + 20)
    card_y = int(sh * 0.17)
    pygame.draw.rect(screen, CARD, pygame.Rect(margin, card_y, content_w, card_h), border_radius=8)
    ty = card_y + 10
    for line in lines:
        surf = f_main.render(line, True, GREEN)
        screen.blit(surf, surf.get_rect(midtop=(sw // 2, ty)))
        ty += lh

    # Input box
    box_y = card_y + card_h + int(sh * 0.04)
    box_h = int(sh * 0.22)
    box_rect = pygame.Rect(margin, box_y, content_w, box_h)
    pygame.draw.rect(screen, CARD, box_rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, box_rect, 2, border_radius=10)

    disp = to_nfc(copy_text + "│")
    ilines = wrap_text(f_input, disp, content_w - 24)
    ilh = f_input.get_linesize() + 4
    iy = box_y + 14
    for line in ilines:
        ls = f_input.render(line, True, WHITE)
        screen.blit(ls, (margin + 12, iy))
        iy += ilh

    hint = f_hint.render(to_nfc("Eingabe = weiter"), True, GRAY)
    screen.blit(hint, hint.get_rect(midbottom=(sw // 2, sh - int(sh * 0.02))))


def _draw_summary(screen, sw, sh, chosen, session_errors,
                  f_title, f_main, f_small, f_hint,
                  CARD, ACCENT, GREEN, RED, WHITE, GRAY, YELLOW, base):
    from collections import Counter

    margin = int(sw * 0.06)
    content_w = sw - 2 * margin

    hs = f_title.render(to_nfc("Zusammenfassung"), True, ACCENT)
    screen.blit(hs, hs.get_rect(midtop=(sw // 2, int(sh * 0.05))))

    n_err = len(session_errors)
    stat_col = GREEN if n_err == 0 else (YELLOW if n_err <= 5 else RED)
    s2 = f_main.render(to_nfc(f"{len(chosen)} Sätze  –  {n_err} Fehler"), True, stat_col)
    screen.blit(s2, s2.get_rect(midtop=(sw // 2, int(sh * 0.16))))

    if not session_errors:
        bravo = f_title.render(to_nfc("Super! Kein einziger Fehler!"), True, GREEN)
        screen.blit(bravo, bravo.get_rect(center=(sw // 2, sh // 2)))
    else:
        cat_count: Counter = Counter(e["category"] for e in session_errors)
        most_common = cat_count.most_common()
        top_count = most_common[0][1]

        card_y = int(sh * 0.26)
        card_h = int(sh * 0.62)
        pygame.draw.rect(screen, CARD, pygame.Rect(margin, card_y, content_w, card_h), border_radius=12)

        ey = card_y + 14
        max_y = card_y + card_h - f_small.get_linesize()
        for cat, count in most_common:
            if ey > max_y:
                break
            cat_label = ERROR_LABELS.get(cat, cat)
            tip = ERROR_TIPS.get(cat, "")
            bar_w = int((content_w - 40) * count / max(1, top_count))
            pygame.draw.rect(screen, (55, 75, 115),
                             pygame.Rect(margin + 12, ey + 4, bar_w, f_small.get_linesize() - 4),
                             border_radius=3)
            ls = f_small.render(to_nfc(f"{count}×  {cat_label}  –  {tip}"), True, WHITE)
            screen.blit(ls, (margin + 16, ey))
            ey += f_small.get_linesize() + 8

    cont = to_nfc("Leertaste = beenden")
    cs = f_hint.render(cont, True, GRAY)
    screen.blit(cs, cs.get_rect(midbottom=(sw // 2, sh - int(sh * 0.02))))


if __name__ == "__main__":
    main()
