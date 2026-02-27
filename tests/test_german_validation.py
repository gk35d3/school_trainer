import unittest

from apps.deutsch_trainer import WritingItem, build_bilingual_prompt, evaluate_free_answer


class GermanValidationTests(unittest.TestCase):
    # Objective: Build compact WritingItem fixtures for grammar and spelling tests.
    def make_item(self, *, subject, number, gender, min_words=4, min_verbs=1):
        return WritingItem(
            instruction="",
            prompt="",
            subject_keywords=[subject],
            subject_number=number,
            subject_gender=gender,
            tags=["noun_cap", "verb_end", "punct"],
            kind="open_question",
            example="",
            min_words=min_words,
            min_verbs=min_verbs,
            level=1,
        )

    # Objective: Ensure a clean, correct singular answer is accepted.
    def test_accepts_valid_singular_sentence(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, err_idxs, msg = evaluate_free_answer(item, "Ein Hund läuft im Park.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertEqual(msg, "")

    # Objective: Ensure a clean, correct plural answer is accepted.
    def test_accepts_valid_plural_sentence(self):
        item = self.make_item(subject="bienen", number="plural", gender="f")
        ok, err_idxs, msg = evaluate_free_answer(item, "Bienen fliegen und sammeln Nektar.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertEqual(msg, "")

    # Objective: Accept a short, correct Bee answer without false spelling flags.
    def test_accepts_bee_sentence_bienen_sind_hier(self):
        item = self.make_item(subject="bienen", number="plural", gender="f", min_words=3, min_verbs=1)
        ok, err_idxs, msg = evaluate_free_answer(item, "Bienen sind hier.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertEqual(msg, "")

    # Objective: Accept a valid bee sentence variant with short free vocabulary.
    def test_accepts_bee_sentence_bienen_sind_da(self):
        item = self.make_item(subject="bienen", number="plural", gender="f", min_words=3, min_verbs=1)
        ok, err_idxs, msg = evaluate_free_answer(item, "Bienen sind da.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertEqual(msg, "")

    # Objective: Accept a mixed sentence and match singular agreement to the target subject.
    def test_accepts_mixed_sentence_with_hund_isst(self):
        item = self.make_item(subject="hund", number="singular", gender="m", min_words=3, min_verbs=1)
        ok, err_idxs, msg = evaluate_free_answer(item, "Bienen sind da und Hund isst etwas.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertTrue(msg == "" or "Ziel" in msg)

    # Objective: Accept valid preposition usage with umlaut.
    def test_accepts_sentence_with_ueber(self):
        item = self.make_item(subject="vögel", number="plural", gender="m", min_words=4, min_verbs=1)
        ok, err_idxs, msg = evaluate_free_answer(item, "Vögel fliegen über Felder.")
        self.assertTrue(ok)
        self.assertEqual(err_idxs, set())
        self.assertEqual(msg, "")

    # Objective: Detect lowercase sentence starts.
    def test_rejects_lowercase_sentence_start(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, err_idxs, msg = evaluate_free_answer(item, "ein Hund läuft im Park.")
        self.assertFalse(ok)
        self.assertIn(0, err_idxs)
        self.assertIn("Satzanfang", msg)

    # Objective: Detect missing end punctuation.
    def test_rejects_missing_end_punctuation(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, _, msg = evaluate_free_answer(item, "Ein Hund läuft im Park")
        self.assertFalse(ok)
        self.assertIn("Satzzeichen", msg)

    # Objective: Detect wrong pronoun gender/number agreement.
    def test_rejects_wrong_pronoun_for_subject(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, _, msg = evaluate_free_answer(item, "Ein Hund läuft im Park. Sie spielt mit dem Ball.")
        self.assertFalse(ok)
        self.assertIn("Pronomen", msg)

    # Objective: Detect singular subject with plural verb form.
    def test_rejects_singular_subject_with_plural_verb(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, _, msg = evaluate_free_answer(item, "Ein Hund laufen im Park.")
        self.assertFalse(ok)
        self.assertIn("Singular", msg)

    # Objective: Detect plural subject with singular verb form.
    def test_rejects_plural_subject_with_singular_verb(self):
        item = self.make_item(subject="bienen", number="plural", gender="f")
        ok, _, msg = evaluate_free_answer(item, "Bienen fliegt im Park.")
        self.assertFalse(ok)
        self.assertIn("Plural", msg)

    # Objective: Detect missing subject noun from question anchor.
    def test_rejects_missing_subject_keyword(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, _, msg = evaluate_free_answer(item, "Er läuft im Park.")
        self.assertFalse(ok)
        self.assertIn("Subjekt fehlt", msg)

    # Objective: Detect misspelled word that looks close to a known form.
    def test_rejects_spelling_typo(self):
        item = self.make_item(subject="hund", number="singular", gender="m")
        ok, _, msg = evaluate_free_answer(item, "Ein Hund spilen im Park.")
        self.assertFalse(ok)
        self.assertTrue("Rechtschreibung" in msg or "Unbekannt" in msg)

    # Objective: Keep typo detection active after broadening open-vocabulary acceptance.
    def test_rejects_spelling_typo_hund_istt(self):
        item = self.make_item(subject="hund", number="singular", gender="m", min_words=3, min_verbs=1)
        ok, _, msg = evaluate_free_answer(item, "Hund istt da.")
        self.assertFalse(ok)
        self.assertIn("Rechtschreibung", msg)

    # Objective: Show German question plus English hint for comprehension support.
    def test_build_bilingual_prompt(self):
        prompt = build_bilingual_prompt("Was passiert bei einer Überschwemmung?")
        self.assertIn("Frage (DE):", prompt)
        self.assertIn("Question (EN):", prompt)


if __name__ == "__main__":
    unittest.main()
