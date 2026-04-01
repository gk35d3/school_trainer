"""
Unit tests for the error-analysis logic in apps/diktat_trainer.py.

Covers:
  _strip_punct       – punctuation removal helper
  _word_body         – lowercase+NFC+strip combined
  _classify_word_error – single-word error classification
  analyse_errors     – full sentence comparison (also covers wrong_word_indices
                       indirectly through error categories)

Each test method has a one-line objective comment so failures are
self-explanatory without reading the code.
"""

import unittest

from apps.diktat_trainer import (
    _strip_punct,
    _word_body,
    _classify_word_error,
    analyse_errors,
    wrong_word_indices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cats(errors):
    """Return just the category strings from an error list."""
    return [e["category"] for e in errors]


def _cats_set(errors):
    return set(_cats(errors))


# ─────────────────────────────────────────────────────────────────────────────
# _strip_punct
# ─────────────────────────────────────────────────────────────────────────────

class TestStripPunct(unittest.TestCase):

    def test_strips_trailing_period(self):
        # Objective: trailing period is removed from a word.
        self.assertEqual(_strip_punct("Hund."), "Hund")

    def test_strips_trailing_comma(self):
        # Objective: trailing comma is removed.
        self.assertEqual(_strip_punct("laufen,"), "laufen")

    def test_strips_trailing_exclamation(self):
        # Objective: trailing exclamation mark is removed.
        self.assertEqual(_strip_punct("toll!"), "toll")

    def test_strips_leading_and_trailing(self):
        # Objective: leading and trailing punctuation are both removed.
        self.assertEqual(_strip_punct("«Hallo»"), "Hallo")

    def test_word_without_punct_unchanged(self):
        # Objective: a clean word is returned unchanged.
        self.assertEqual(_strip_punct("Katze"), "Katze")

    def test_preserves_internal_hyphen(self):
        # Objective: hyphens inside a compound are NOT stripped (not in strip set).
        self.assertEqual(_strip_punct("Haus-Tür"), "Haus-Tür")


# ─────────────────────────────────────────────────────────────────────────────
# _word_body
# ─────────────────────────────────────────────────────────────────────────────

class TestWordBody(unittest.TestCase):

    def test_lowercases(self):
        # Objective: result is always lowercase.
        self.assertEqual(_word_body("Schule"), "schule")

    def test_strips_punct_and_lowercases(self):
        # Objective: punctuation stripped AND lowercased together.
        self.assertEqual(_word_body("Schule."), "schule")

    def test_normalises_nfc(self):
        # Objective: NFD-encoded umlauts are normalised to NFC.
        import unicodedata
        nfd_ae = unicodedata.normalize("NFD", "ä")   # a + combining diaeresis
        self.assertEqual(_word_body(nfd_ae), "ä")

    def test_identical_bodies_for_case_variants(self):
        # Objective: "See", "see", "SEE" all produce the same body.
        self.assertEqual(_word_body("See"), _word_body("see"))
        self.assertEqual(_word_body("See"), _word_body("SEE"))


# ─────────────────────────────────────────────────────────────────────────────
# _classify_word_error  –  capitalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyCapitalisation(unittest.TestCase):

    def test_noun_not_capitalised(self):
        # Objective: "Hund" → "hund" is großschreibung (noun not capitalised).
        self.assertEqual(_classify_word_error("Hund", "hund"), "großschreibung")

    def test_noun_not_capitalised_mid_sentence(self):
        # Objective: mid-sentence noun without capital is großschreibung.
        self.assertEqual(_classify_word_error("Schule", "schule"), "großschreibung")

    def test_lowercase_word_wrongly_capitalised(self):
        # Objective: "gerne" → "Gerne" is kleinschreibung.
        self.assertEqual(_classify_word_error("gerne", "Gerne"), "kleinschreibung")

    def test_sentence_start_lowercase(self):
        # Objective: "Wir" → "wir" (sentence start) is großschreibung.
        self.assertEqual(_classify_word_error("Wir", "wir"), "großschreibung")

    def test_all_caps_noun(self):
        # Objective: "See" → "SEE" (all-caps) is großschreibung.
        self.assertEqual(_classify_word_error("See", "SEE"), "großschreibung")

    def test_all_caps_verb(self):
        # Objective: "spielen" → "SPIELEN": expected lowercase, student wrote all-caps.
        # First letter: expected "s" (lower), actual "S" (upper) → kleinschreibung.
        self.assertEqual(_classify_word_error("spielen", "SPIELEN"), "kleinschreibung")

    def test_word_with_trailing_punct_cap(self):
        # Objective: punctuation on expected word does not confuse capitalisation check.
        self.assertEqual(_classify_word_error("See.", "see"), "großschreibung")

    def test_correct_word_returns_rechtschreibung(self):
        # Objective: identical words (caller should not call this) → rechtschreibung fallback.
        self.assertEqual(_classify_word_error("Hund", "Hund"), "rechtschreibung")


# ─────────────────────────────────────────────────────────────────────────────
# _classify_word_error  –  spelling categories
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySpelling(unittest.TestCase):

    def test_sz_ss_expected_sz(self):
        # Objective: "Straße" → "Strasse" is sz_ss.
        self.assertEqual(_classify_word_error("Straße", "Strasse"), "sz_ss")

    def test_sz_ss_actual_sz(self):
        # Objective: "nass" → "naß" is sz_ss (reversed direction).
        self.assertEqual(_classify_word_error("nass", "naß"), "sz_ss")

    def test_umlaut_a(self):
        # Objective: "Mädchen" → "Madchen" is umlaut.
        self.assertEqual(_classify_word_error("Mädchen", "Madchen"), "umlaut")

    def test_umlaut_o(self):
        # Objective: "schön" → "schon" is umlaut.
        self.assertEqual(_classify_word_error("schön", "schon"), "umlaut")

    def test_umlaut_u(self):
        # Objective: "müde" → "mude" is umlaut.
        self.assertEqual(_classify_word_error("müde", "mude"), "umlaut")

    def test_ie_ei_swap_ie(self):
        # Objective: "Tier" → "Teir" is ie_ei.
        self.assertEqual(_classify_word_error("Tier", "Teir"), "ie_ei")

    def test_ie_ei_swap_ei(self):
        # Objective: "klein" → "klien" is ie_ei.
        self.assertEqual(_classify_word_error("klein", "klien"), "ie_ei")

    def test_doppelkonsonant_mm(self):
        # Objective: "schwimmen" → "schwimen" is doppelkonsonant.
        self.assertEqual(_classify_word_error("schwimmen", "schwimen"), "doppelkonsonant")

    def test_doppelkonsonant_ll(self):
        # Objective: "stellen" → "stelen" is doppelkonsonant.
        self.assertEqual(_classify_word_error("stellen", "stelen"), "doppelkonsonant")

    def test_doppelkonsonant_nn(self):
        # Objective: "rennen" → "renen" is doppelkonsonant.
        self.assertEqual(_classify_word_error("rennen", "renen"), "doppelkonsonant")

    def test_ck_missing(self):
        # Objective: "Rücken" → "Rüken" (only ck→k, umlaut preserved) is ck.
        # "Rücken" → "Ruken" would be a compound error (umlaut+ck) and falls to
        # rechtschreibung; this variant isolates the ck error cleanly.
        self.assertEqual(_classify_word_error("Rücken", "Rüken"), "ck")

    def test_ck_written_as_k(self):
        # Objective: "Jacke" → "Jake" is ck.
        self.assertEqual(_classify_word_error("Jacke", "Jake"), "ck")

    def test_tz_missing(self):
        # Objective: "Katze" → "Kaze" is tz.
        self.assertEqual(_classify_word_error("Katze", "Kaze"), "tz")

    def test_tz_written_as_z(self):
        # Objective: "sitzen" → "sizen" is tz.
        self.assertEqual(_classify_word_error("sitzen", "sizen"), "tz")

    def test_completely_wrong_word(self):
        # Objective: totally different word → rechtschreibung.
        self.assertEqual(_classify_word_error("Apfel", "Birne"), "rechtschreibung")

    def test_missing_letter_not_in_any_category(self):
        # Objective: "gerne" → "gern" (simple truncation) → rechtschreibung.
        self.assertEqual(_classify_word_error("gerne", "gern"), "rechtschreibung")


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  perfect input
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysePerfect(unittest.TestCase):

    def test_perfect_sentence_no_errors(self):
        # Objective: a perfectly typed sentence produces zero errors.
        self.assertEqual(analyse_errors("Die Katze schläft.", "Die Katze schläft."), [])

    def test_perfect_sentence_without_period_in_expected(self):
        # Objective: no false period error when expected sentence has no period.
        self.assertEqual(analyse_errors("Super", "Super"), [])

    def test_perfect_multiword(self):
        # Objective: five-word correct sentence returns no errors.
        self.assertEqual(
            analyse_errors("Wir spielen im Garten.", "Wir spielen im Garten."), []
        )


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  capitalisation errors detected via "equal" opcode path
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseCapitalisation(unittest.TestCase):

    def test_sentence_start_lowercase(self):
        # Objective: lowercase first word detected as großschreibung.
        errs = analyse_errors("Wir spielen.", "wir spielen.")
        self.assertIn("großschreibung", _cats(errs))

    def test_noun_not_capitalised_mid_sentence(self):
        # Objective: lowercase noun mid-sentence → großschreibung.
        errs = analyse_errors("Ich sehe den Hund.", "Ich sehe den hund.")
        self.assertIn("großschreibung", _cats(errs))

    def test_verb_wrongly_capitalised(self):
        # Objective: verb capitalised by mistake → kleinschreibung.
        errs = analyse_errors("Wir spielen gerne.", "Wir Spielen gerne.")
        self.assertIn("kleinschreibung", _cats(errs))

    def test_all_caps_noun_detected(self):
        # Objective: ALL-CAPS noun → großschreibung (the SEE bug case).
        errs = analyse_errors("Wir schwimmen im See.", "Wir schwimmen im SEE.")
        self.assertIn("großschreibung", _cats(errs))

    def test_all_caps_noun_only_one_error(self):
        # Objective: ALL-CAPS noun counts as exactly ONE error, not multiple.
        errs = analyse_errors("Wir schwimmen im See.", "Wir schwimmen im SEE.")
        self.assertEqual(len(errs), 1)

    def test_no_false_positive_for_correct_caps(self):
        # Objective: "Die Katze" (correctly capitalised noun) raises no error.
        errs = analyse_errors("Die Katze spielt.", "Die Katze spielt.")
        self.assertEqual(errs, [])


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  spelling errors
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseSpelling(unittest.TestCase):

    def test_sz_ss_detected(self):
        # Objective: ß written as ss is detected as sz_ss.
        errs = analyse_errors("Die Straße ist lang.", "Die Strasse ist lang.")
        self.assertIn("sz_ss", _cats(errs))

    def test_umlaut_dropped_detected(self):
        # Objective: dropped umlaut → umlaut error.
        errs = analyse_errors("Das Mädchen lacht.", "Das Madchen lacht.")
        self.assertIn("umlaut", _cats(errs))

    def test_ie_ei_detected(self):
        # Objective: ie/ei swap → ie_ei error.
        errs = analyse_errors("Das Tier schläft.", "Das Teir schläft.")
        self.assertIn("ie_ei", _cats(errs))

    def test_doppelkonsonant_detected(self):
        # Objective: missing double consonant → doppelkonsonant error.
        errs = analyse_errors("Wir schwimmen gerne.", "Wir schwimen gerne.")
        self.assertIn("doppelkonsonant", _cats(errs))

    def test_ck_detected(self):
        # Objective: ck written as k → ck error.
        errs = analyse_errors("Die Jacke ist warm.", "Die Jake ist warm.")
        self.assertIn("ck", _cats(errs))

    def test_tz_detected(self):
        # Objective: tz written as z → tz error.
        errs = analyse_errors("Die Katze spielt.", "Die Kaze spielt.")
        self.assertIn("tz", _cats(errs))

    def test_completely_different_word(self):
        # Objective: substituted word → rechtschreibung.
        errs = analyse_errors("Ich esse einen Apfel.", "Ich esse einen Birne.")
        self.assertIn("rechtschreibung", _cats(errs))


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  missing / extra words
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseMissingExtra(unittest.TestCase):

    def test_missing_word(self):
        # Objective: omitted word → fehlwort error.
        errs = analyse_errors("Ich gehe nach Hause.", "Ich gehe Hause.")
        self.assertIn("fehlwort", _cats(errs))

    def test_extra_word(self):
        # Objective: extra word typed → extra_wort error.
        errs = analyse_errors("Die Katze schläft.", "Die Katze wirklich schläft.")
        self.assertIn("extra_wort", _cats(errs))

    def test_completely_empty_answer(self):
        # Objective: empty input → every word is fehlwort + satzzeichen.
        errs = analyse_errors("Wir spielen.", "")
        cats = _cats(errs)
        self.assertIn("fehlwort", cats)
        self.assertIn("satzzeichen", cats)


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  punctuation (satzzeichen)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysePunctuation(unittest.TestCase):

    def test_missing_period(self):
        # Objective: missing final period → satzzeichen error.
        errs = analyse_errors("Wir spielen.", "Wir spielen")
        self.assertIn("satzzeichen", _cats(errs))

    def test_no_false_period_error_when_present(self):
        # Objective: period present → no satzzeichen error.
        errs = analyse_errors("Wir spielen.", "Wir spielen.")
        self.assertNotIn("satzzeichen", _cats(errs))

    def test_no_false_period_error_when_expected_has_none(self):
        # Objective: expected sentence without period → no satzzeichen raised.
        errs = analyse_errors("Hallo", "Hallo")
        self.assertNotIn("satzzeichen", _cats(errs))


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  the exact reported "SEE" regression test
# ─────────────────────────────────────────────────────────────────────────────

class TestSeeBugRegression(unittest.TestCase):

    def test_see_regression_five_errors(self):
        # Objective: the originally reported sentence detects exactly 5 errors.
        # Expected: "Wir schwimmen gerne im See."
        # Typed:    "wir schwimen gern im SEE"
        # Errors:   großschreibung(Wir/wir), doppelkonsonant(schwimmen/schwimen),
        #           rechtschreibung(gerne/gern), großschreibung(See/SEE), satzzeichen
        errs = analyse_errors(
            "Wir schwimmen gerne im See.",
            "wir schwimen gern im SEE",
        )
        cats = _cats(errs)
        self.assertEqual(len(errs), 5, f"Expected 5 errors, got {len(errs)}: {cats}")
        self.assertEqual(cats.count("großschreibung"), 2)
        self.assertEqual(cats.count("doppelkonsonant"), 1)
        self.assertEqual(cats.count("rechtschreibung"), 1)
        self.assertEqual(cats.count("satzzeichen"), 1)

    def test_see_not_misclassified_as_correct(self):
        # Objective: "SEE" is NOT silently accepted as correct spelling of "See".
        errs = analyse_errors("im See.", "im SEE")
        self.assertTrue(len(errs) >= 1, "SEE should not be accepted as correct")

    def test_see_no_spurious_extra_errors(self):
        # Objective: "SEE" generates exactly 1 word error + 1 satzzeichen, not more.
        errs = analyse_errors("im See.", "im SEE")
        self.assertEqual(len(errs), 2, f"Expected 2 errors, got {len(errs)}: {_cats(errs)}")


# ─────────────────────────────────────────────────────────────────────────────
# analyse_errors  –  multiple errors in one sentence
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseMultipleErrors(unittest.TestCase):

    def test_cap_and_spelling_together(self):
        # Objective: capitalisation + spelling errors both appear independently.
        # "Die Straße" → "die Strasse": großschreibung on "Die" + sz_ss on "Straße"
        errs = analyse_errors("Die Straße ist breit.", "die Strasse ist breit.")
        cats = _cats(errs)
        self.assertIn("großschreibung", cats)
        self.assertIn("sz_ss", cats)

    def test_three_errors_three_categories(self):
        # Objective: three distinct error types appear in a three-error sentence.
        # "Wir schwimmen im Garten." → "wir schwimen im Garten"
        # großschreibung(Wir/wir), doppelkonsonant(schwimmen/schwimen), satzzeichen
        errs = analyse_errors("Wir schwimmen im Garten.", "wir schwimen im Garten")
        cats = _cats(errs)
        self.assertIn("großschreibung", cats)
        self.assertIn("satzzeichen", cats)
        # "schwimmen" → "schwimen": double-m dropped → doppelkonsonant
        self.assertIn("doppelkonsonant", cats)


# ─────────────────────────────────────────────────────────────────────────────
# wrong_word_indices
# ─────────────────────────────────────────────────────────────────────────────

class TestWrongWordIndices(unittest.TestCase):

    def test_perfect_sentence_no_indices(self):
        # Objective: no word indices flagged for a perfect answer.
        self.assertEqual(
            wrong_word_indices("Die Katze schläft.", "Die Katze schläft."), set()
        )

    def test_flags_wrong_word_index(self):
        # Objective: index of the misspelled word is returned.
        idxs = wrong_word_indices("Ich gehe nach Hause.", "Ich gehe nach Haus.")
        self.assertIn(3, idxs)

    def test_flags_lowercase_noun(self):
        # Objective: a noun written in lowercase is flagged by index.
        idxs = wrong_word_indices("Der Hund läuft.", "Der hund läuft.")
        self.assertIn(1, idxs)

    def test_does_not_flag_correct_words(self):
        # Objective: correct words do not appear in the wrong-index set.
        idxs = wrong_word_indices("Ich gehe nach Hause.", "Ich gehe nach Haus.")
        self.assertNotIn(0, idxs)  # "Ich" is correct
        self.assertNotIn(1, idxs)  # "gehe" is correct


if __name__ == "__main__":
    unittest.main()
