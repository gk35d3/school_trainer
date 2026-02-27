import random
import unittest

from apps.times_trainer import (
    Problem,
    allowed_rows_and_partner_max,
    allowed_target_tags_for_difficulty,
    assign_tags,
    make_problem_for_target,
    micro_hint,
)


class TimesGenerationTests(unittest.TestCase):
    # Objective: Ensure canonical fact tags stay stable across orientation.
    def test_assign_tags_contains_canonical_fact(self):
        tags = assign_tags(8, 7)
        self.assertIn("fact_7x8", tags)
        self.assertIn("mul_8", tags)
        self.assertIn("mul_7", tags)
        self.assertIn("hard", tags)

    # Objective: Verify curriculum gating at low and high difficulty.
    def test_allowed_rows_gate_progressively(self):
        low_rows, low_partner_max = allowed_rows_and_partner_max(0.1)
        high_rows, high_partner_max = allowed_rows_and_partner_max(0.9)
        self.assertEqual(low_rows, [1, 2, 5, 10])
        self.assertEqual(low_partner_max, 5)
        self.assertIn(7, high_rows)
        self.assertEqual(high_partner_max, 10)

    # Objective: Ensure hard tags are unlocked only at high difficulty.
    def test_allowed_target_tags_unlock_hard_late(self):
        low = allowed_target_tags_for_difficulty(0.2)
        high = allowed_target_tags_for_difficulty(0.9)
        self.assertNotIn("hard", low)
        self.assertIn("hard", high)
        self.assertIn("fact_7x8", high)

    # Objective: Keep generator aligned with requested target tags.
    def test_make_problem_for_target(self):
        random.seed(1234)
        p = make_problem_for_target(0.9, "mul_7", recent_first_operands=[])
        self.assertIsInstance(p, Problem)
        self.assertIn("mul_7", p.tags)
        self.assertEqual(p.op, "*")
        self.assertEqual(p.answer, p.a * p.b)

    # Objective: Ensure tiny hints map to anchor and strategy patterns.
    def test_micro_hint_examples(self):
        self.assertEqual(micro_hint(Problem(10, 6, "*", [])), "Haeng 0 dran.")
        self.assertEqual(micro_hint(Problem(9, 7, "*", [])), "9er: eins weniger.")


if __name__ == "__main__":
    unittest.main()
