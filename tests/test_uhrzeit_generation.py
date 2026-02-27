import random
import unittest

from apps.uhrzeit_trainer import (
    Question,
    assign_tags,
    build_q3,
    format_time,
    make_question_for_target,
    minutes_for_stage,
    normalize_number_input,
    normalize_time_input,
)


class UhrzeitGenerationTests(unittest.TestCase):
    # Objective: Verify flexible time input normalization.
    def test_normalize_time_input_variants(self):
        self.assertEqual(normalize_time_input("3:5"), "3:05")
        self.assertEqual(normalize_time_input("3.05"), "3:05")
        self.assertEqual(normalize_time_input("3 05"), "3:05")
        self.assertEqual(normalize_time_input("11"), "11:00")

    # Objective: Reject invalid time values.
    def test_normalize_time_input_invalid(self):
        self.assertIsNone(normalize_time_input("0:30"))
        self.assertIsNone(normalize_time_input("13:00"))
        self.assertIsNone(normalize_time_input("8:77"))

    # Objective: Verify numeric answer normalization for Q3.
    def test_normalize_number_input(self):
        self.assertEqual(normalize_number_input("0"), 0)
        self.assertEqual(normalize_number_input("59"), 59)
        self.assertIsNone(normalize_number_input("60"))
        self.assertIsNone(normalize_number_input("x"))

    # Objective: Ensure stage minute gates match curriculum.
    def test_minutes_for_stage(self):
        self.assertEqual(minutes_for_stage("A"), [0])
        self.assertEqual(minutes_for_stage("B"), [0, 30])
        self.assertEqual(minutes_for_stage("C"), [0, 15, 30, 45])
        self.assertEqual(minutes_for_stage("D"), list(range(0, 60, 5)))

    # Objective: Ensure tagging includes level and fact tags.
    def test_assign_tags(self):
        tags = assign_tags(3, 45, "C")
        self.assertIn("level_C_quarter", tags)
        self.assertIn("m_45", tags)
        self.assertIn("t_3:45", tags)
        self.assertIn("quarter", tags)

    # Objective: Ensure Q3 answers are in valid numeric range.
    def test_build_q3_answer_range(self):
        q = build_q3(4, 25, "D", assign_tags(4, 25, "D"))
        self.assertEqual(q.question_type, "Q3")
        self.assertIsNotNone(q.expected_number)
        self.assertGreaterEqual(q.expected_number, 0)
        self.assertLessEqual(q.expected_number, 59)

    # Objective: Ensure generator can target a minute-tag and return a valid question.
    def test_make_question_for_target(self):
        random.seed(321)
        q = make_question_for_target(3, "m_30", q_index=0, recent_signatures=[])
        self.assertIsInstance(q, Question)
        self.assertIn("m_30", q.tags)
        self.assertEqual(format_time(q.hour, q.minute), f"{q.hour}:{q.minute:02d}")


if __name__ == "__main__":
    unittest.main()
