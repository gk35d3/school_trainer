import unittest

from core.adaptive_core import (
    build_state_from_events,
    latest_logged_difficulty,
    tag_metrics,
    update_tag_stats,
)


class AdaptiveCoreTests(unittest.TestCase):
    # Objective: Ensure per-tag attempt windows keep only the newest records.
    def test_update_tag_stats_respects_window(self):
        state = {"difficulty": 0.2, "tags": {}}
        for i in range(5):
            update_tag_stats(state, ["noun_cap"], correct=(i % 2 == 0), rt=5.0 + i, tag_window=3)
        attempts = state["tags"]["noun_cap"]["attempts"]
        self.assertEqual(len(attempts), 3)

    # Objective: Ensure metrics return defaults for unseen tags.
    def test_tag_metrics_default_for_unseen(self):
        state = {"difficulty": 0.2, "tags": {}}
        acc, rt, n = tag_metrics(state, "missing", 0.55, 9.0)
        self.assertEqual(acc, 0.55)
        self.assertEqual(rt, 9.0)
        self.assertEqual(n, 0)

    # Objective: Rebuild app-specific state from mixed shared events.
    def test_build_state_from_events_filters_by_app(self):
        events = [
            {"type": "attempt", "app": "math", "tags": ["add_small"], "correct": True, "rt": 2.0},
            {"type": "attempt", "app": "german", "tags": ["noun_cap"], "correct": False, "rt": 8.0},
            {"type": "attempt", "app": "math", "tags": ["add_small"], "correct": False, "rt": 6.0},
        ]
        state = build_state_from_events(
            events,
            app_id="math",
            initial_difficulty=0.15,
            default_acc=0.6,
            default_rt=9.0,
            tag_window=50,
            rt_good=3.0,
            rt_bad=12.0,
            smooth_old=0.9,
            smooth_new=0.1,
            total_seen_key="total_questions_seen",
        )
        self.assertIn("add_small", state["tags"])
        self.assertNotIn("noun_cap", state["tags"])
        self.assertEqual(state["total_questions_seen"], 1)

    # Objective: Pick the newest available difficulty key from logs.
    def test_latest_logged_difficulty_picks_most_recent(self):
        events = [
            {"app": "math", "difficulty_start": 0.2},
            {"app": "german", "difficulty_end": 0.8},
            {"app": "math", "difficulty_end": 0.7},
        ]
        self.assertEqual(latest_logged_difficulty(events, "math", 0.1), 0.7)
        self.assertEqual(latest_logged_difficulty(events, "german", 0.1), 0.8)


if __name__ == "__main__":
    unittest.main()
