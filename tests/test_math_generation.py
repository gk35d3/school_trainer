import random
import unittest

from apps.mathe_trainer import (
    MAX_VALUE,
    MIN_VALUE,
    allowed_tags_for_difficulty,
    assign_tags,
    difficulty_to_limits,
    make_problem_for_target,
)


class MathGenerationTests(unittest.TestCase):
    # Objective: Verify carry and cross-10 tagging for representative addition.
    def test_assign_tags_add_carry(self):
        tags = assign_tags(48, 28, "+")
        self.assertIn("add", tags)
        self.assertIn("add_two_digit", tags)
        self.assertIn("add_carry", tags)
        self.assertIn("add_cross10", tags)

    # Objective: Verify borrow tagging for representative subtraction.
    def test_assign_tags_sub_borrow(self):
        tags = assign_tags(93, 39, "-")
        self.assertIn("sub", tags)
        self.assertIn("sub_two_digit", tags)
        self.assertIn("sub_borrow", tags)

    # Objective: Ensure difficulty bands grow numeric limits monotonically.
    def test_difficulty_limits_increase(self):
        low = difficulty_to_limits(0.1)
        mid = difficulty_to_limits(0.5)
        high = difficulty_to_limits(0.9)
        self.assertLessEqual(low["max_small"], mid["max_small"])
        self.assertLessEqual(mid["max_small"], high["max_small"])
        self.assertLess(low["max_two"], high["max_two"])

    # Objective: Ensure harder tags are only introduced in higher ranges.
    def test_allowed_tags_by_difficulty(self):
        easy = allowed_tags_for_difficulty(0.1)
        hard = allowed_tags_for_difficulty(0.9)
        self.assertIn("add_small", easy)
        self.assertNotIn("add_carry", easy)
        self.assertIn("add_carry", hard)
        self.assertIn("sub_borrow", hard)

    # Objective: Keep generated problems inside 1..100 constraints and valid answers.
    def test_generated_problems_stay_in_bounds(self):
        random.seed(12345)
        targets = [
            "add_small",
            "sub_small",
            "add_cross10",
            "tens",
            "add_two_digit",
            "sub_two_digit",
            "add_carry",
            "sub_borrow",
        ]
        for target in targets:
            for _ in range(20):
                p = make_problem_for_target(0.85, target)
                self.assertGreaterEqual(p.a, MIN_VALUE)
                self.assertGreaterEqual(p.b, MIN_VALUE)
                self.assertLessEqual(p.a, MAX_VALUE)
                self.assertLessEqual(p.b, MAX_VALUE)
                self.assertGreaterEqual(p.answer, MIN_VALUE)
                self.assertLessEqual(p.answer, MAX_VALUE)


if __name__ == "__main__":
    unittest.main()
