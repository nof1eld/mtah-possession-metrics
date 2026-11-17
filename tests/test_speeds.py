"""Speed/Otsu/aggregate diagnostics on generated trajectories only; no match I/O."""
import json
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from test_spell_level import PARAMS as SPELL_PARAMS


PARAMS = dict(
    frames=4 * run.PARAMS["slow_win"],  # ample interior beyond both centred-window boundaries
    walking_ms=1.0,  # representative walking, below the existing 1.5 m/s fixed fallback
    jogging_ms=3.0,  # representative jogging, above the fallback
    outlier_ms=12.0,  # fast isolated slot tests median robustness, not a decision threshold
    comparison_rtol=np.sqrt(np.finfo(np.float64).eps),  # arithmetic roundoff tolerance, not fitted accuracy
)


def linear_players(velocities, n=PARAMS["frames"]):
    velocities = np.asarray(velocities, dtype=np.float64)
    time = np.arange(n) / run.FPS
    xy = np.empty((n, len(velocities), len(SPELL_PARAMS["pitch"])))
    xy[:, :, 0] = (SPELL_PARAMS["origin_x_m"] + time[:, None] * velocities
                   + np.arange(len(velocities)) * SPELL_PARAMS["opponent_distance_m"])
    xy[:, :, 1] = SPELL_PARAMS["origin_y_m"]
    return xy / np.asarray(SPELL_PARAMS["pitch"])


class PlayerSpeedTests(unittest.TestCase):
    def test_metric_displacement_then_median_then_window_mean(self):
        velocities = [PARAMS[key] for key in ("walking_ms", "jogging_ms", "outlier_ms")]
        players = linear_players(velocities)
        provided = np.full(players.shape[:2], np.nan)  # supplied values must be irrelevant by default
        window, empty = run.player_window_speed(players, provided, SPELL_PARAMS["pitch"], run.PARAMS)
        interior = slice(run.PARAMS["slow_win"], -run.PARAMS["slow_win"])
        np.testing.assert_allclose(window[interior], np.median(velocities), rtol=PARAMS["comparison_rtol"])
        self.assertEqual(empty, 2 * run.PARAMS["displacement_half_frames"])

    def test_each_missing_endpoint_invalidates_its_slot(self):
        half = run.PARAMS["displacement_half_frames"]
        centre = PARAMS["frames"] // len(SPELL_PARAMS["pitch"])
        for coordinate in range(len(SPELL_PARAMS["pitch"])):
            with self.subTest(coordinate=coordinate):
                xy = linear_players([PARAMS["walking_ms"], PARAMS["jogging_ms"]]) * SPELL_PARAMS["pitch"]
                xy[centre, 0, coordinate] = np.nan
                speed = run.displacement_speed(xy[:, :, 0], xy[:, :, 1], half)
                self.assertTrue(np.isnan(speed[centre - half, 0]))
                self.assertTrue(np.isnan(speed[centre + half, 0]))
                self.assertTrue(np.isfinite(speed[centre, 0]))  # only the endpoints matter
                self.assertTrue(np.isfinite(speed[centre - half, 1]))
                self.assertTrue(np.isnan(speed[:half]).all())
                self.assertTrue(np.isnan(speed[-half:]).all())

    def test_empty_speed_rows_are_counted_without_warnings(self):
        players = np.full_like(linear_players([PARAMS["walking_ms"]]), np.nan)
        provided = np.full(players.shape[:2], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for computed in (True, False):
                window, empty = run.player_window_speed(players, provided, SPELL_PARAMS["pitch"],
                    dict(run.PARAMS, computed_player_speeds=computed))
                self.assertEqual(empty, len(players))
                self.assertTrue(np.isnan(window).all())
        short = run.displacement_speed(np.zeros(run.PARAMS["displacement_half_frames"]),
                                       np.zeros(run.PARAMS["displacement_half_frames"]))
        self.assertTrue(np.isnan(short).all())

    def test_missing_slots_are_excluded_from_the_median(self):
        players = linear_players([PARAMS["walking_ms"], PARAMS["jogging_ms"]])
        players[:, 0] = np.nan
        window, _ = run.player_window_speed(players, np.full(players.shape[:2], np.nan),
                                            SPELL_PARAMS["pitch"], run.PARAMS)
        self.assertAlmostEqual(window[len(window) // len(SPELL_PARAMS["pitch"])], PARAMS["jogging_ms"])

    def test_provided_column_mean_remains_an_independent_ablation(self):
        players = linear_players([PARAMS["walking_ms"], PARAMS["jogging_ms"], PARAMS["outlier_ms"]])
        provided = np.tile([PARAMS["walking_ms"], np.nan, PARAMS["outlier_ms"]], (len(players), 1))
        window, empty = run.player_window_speed(players, provided, SPELL_PARAMS["pitch"],
            dict(run.PARAMS, computed_player_speeds=False))
        self.assertEqual(empty, 0)
        np.testing.assert_allclose(window[np.isfinite(window)],
            np.mean([PARAMS["walking_ms"], PARAMS["outlier_ms"]]), rtol=PARAMS["comparison_rtol"])


class OtsuTests(unittest.TestCase):
    def test_exact_two_level_log_split_is_geometric_midpoint(self):
        speeds = np.repeat([PARAMS["walking_ms"], PARAMS["jogging_ms"]], PARAMS["frames"])
        expected = np.sqrt(PARAMS["walking_ms"] * PARAMS["jogging_ms"])
        self.assertAlmostEqual(run.otsu_speed_threshold(speeds), expected)
        self.assertEqual(run.otsu_speed_threshold(speeds), run.otsu_speed_threshold(speeds[::-1]))
        threshold, otsu, mode = run.slow_threshold_for_match([speeds])
        self.assertEqual(mode, "otsu")
        self.assertEqual(threshold, otsu)
        fixed, still_otsu, mode = run.slow_threshold_for_match([speeds],
            dict(run.PARAMS, fixed_slow_threshold=True))
        self.assertEqual((fixed, mode), (run.PARAMS["slow_ms"], "fixed"))
        self.assertEqual(still_otsu, otsu)

    def test_empty_constant_and_roundoff_only_inputs_use_fallback(self):
        for speeds in (np.array([]), np.array([np.nan, np.inf]), np.zeros(PARAMS["frames"]),
                       np.full(PARAMS["frames"], PARAMS["jogging_ms"]),
                       PARAMS["jogging_ms"] + np.arange(PARAMS["frames"]) * np.finfo(float).eps):
            with self.subTest(length=len(speeds)):
                threshold, otsu, mode = run.slow_threshold_for_match([speeds])
                self.assertEqual((threshold, mode), (run.PARAMS["slow_ms"], "fallback"))
                self.assertTrue(np.isnan(otsu))

    def test_weighted_multi_level_split_matches_within_class_variance_minimum(self):
        speeds = np.repeat([PARAMS["walking_ms"], PARAMS["jogging_ms"], PARAMS["outlier_ms"]],
            [PARAMS["frames"], run.PARAMS["slow_win"], run.PARAMS["displacement_half_frames"]])
        logs = np.log(speeds)
        levels = np.unique(logs)
        losses = []
        for lower in levels[:-1]:
            left, right = logs[logs <= lower], logs[logs > lower]
            losses.append(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        split = int(np.argmin(losses))
        expected = np.exp((levels[split] + levels[split + 1]) / len(SPELL_PARAMS["pitch"]))
        self.assertAlmostEqual(run.otsu_speed_threshold(speeds), expected)

    def test_zero_speeds_have_a_finite_log_floor_and_invalid_values_are_ignored(self):
        speeds = np.repeat([0.0, PARAMS["jogging_ms"]], PARAMS["frames"])
        threshold = run.otsu_speed_threshold(speeds)
        self.assertAlmostEqual(threshold, np.sqrt(run.PARAMS["otsu_log_floor_ms"] * PARAMS["jogging_ms"]))
        self.assertTrue(np.isfinite(threshold) and threshold > 0)
        self.assertEqual(run.otsu_speed_threshold(np.r_[speeds, np.nan, np.inf, -np.inf]), threshold)


class AggregateDiagnosticsTests(unittest.TestCase):
    def test_half_boundaries_graded_pool_and_pre_kickoff_denominator(self):
        n = PARAMS["frames"]
        walk = linear_players([PARAMS["walking_ms"], PARAMS["walking_ms"]])
        jog = linear_players([PARAMS["jogging_ms"], PARAMS["jogging_ms"]])
        # An ungraded middle block deliberately jumps wildly; it must enter
        # neither displacement windows nor the global Otsu sample pool.
        ungraded = linear_players([PARAMS["outlier_ms"], PARAMS["outlier_ms"]])
        players = np.concatenate([walk, ungraded, jog])
        ball = np.full((len(players), len(SPELL_PARAMS["pitch"])), np.nan)
        provided = np.full(players.shape[:2], np.nan)
        blocks = [(1, 0, n - 1), (2, 2 * n, len(players) - 1)]
        messages = []
        result = run.build_labels(players, provided, ball, np.array([run.HOME, run.AWAY]), len(ball),
            blocks, 2 * n + n // len(SPELL_PARAMS["pitch"]), SPELL_PARAMS["pitch"], log=messages.append)
        threshold = json.loads(messages[0].split(": ", 1)[1])
        self.assertEqual(threshold["mode"], "otsu")
        self.assertAlmostEqual(threshold["otsu_ms"], np.sqrt(PARAMS["walking_ms"] * PARAMS["jogging_ms"]))
        summaries = [json.loads(message.split(": ", 1)[1]) for message in messages if "speed diagnostics:" in message]
        self.assertEqual(len(summaries), len(blocks))
        for summary, expected in zip(summaries, (PARAMS["walking_ms"], PARAMS["jogging_ms"])):
            self.assertEqual(summary["frames"], n)
            self.assertEqual(summary["empty_slot_speed_frames"], 2 * run.PARAMS["displacement_half_frames"])
            np.testing.assert_allclose(list(summary["window_speed_percentiles_ms"].values()), expected,
                                       rtol=PARAMS["comparison_rtol"])
            self.assertEqual(summary["dead_cue_fractions"]["ball_unseen_long"],
                             (n - run.PARAMS["long_ball_gap_frames"]) / n)
        self.assertEqual(summaries[1]["d_share"], 1.0)
        self.assertTrue(np.all(result[n:2 * n] == "X"))

    def test_all_missing_diagnostics_are_null_and_counted_without_warnings(self):
        players = np.full_like(linear_players([PARAMS["walking_ms"], PARAMS["jogging_ms"]]), np.nan)
        ball = np.full((len(players), len(SPELL_PARAMS["pitch"])), np.nan)
        messages = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            run.build_labels(players, np.full(players.shape[:2], np.nan), ball,
                np.array([run.HOME, run.AWAY]), len(ball), [(1, 0, len(ball) - 1)], None,
                SPELL_PARAMS["pitch"], log=messages.append)
        summary = json.loads(messages[-1].split(": ", 1)[1])
        self.assertEqual(summary["empty_slot_speed_frames"], len(ball))
        self.assertEqual(summary["window_speed_nan_fraction"], 1.0)
        self.assertTrue(all(value is None for value in summary["window_speed_percentiles_ms"].values()))
        self.assertFalse(any("NaN" in message or "array(" in message for message in messages))

    def test_long_unseen_counter_resets_on_either_missing_coordinate_and_on_observation(self):
        limit = run.PARAMS["long_ball_gap_frames"]
        for coordinate in range(len(SPELL_PARAMS["pitch"])):
            with self.subTest(coordinate=coordinate):
                ball = np.zeros((PARAMS["frames"], len(SPELL_PARAMS["pitch"])))
                ball[:limit + 1, coordinate] = np.nan
                cue = run.long_unseen_ball(ball)
                self.assertFalse(cue[:limit].any())
                self.assertTrue(cue[limit])
                self.assertFalse(cue[limit + 1:].any())

    def test_diagnostic_ablation_does_not_change_labels(self):
        players = linear_players([PARAMS["walking_ms"], PARAMS["jogging_ms"]])
        ball = players[:, run.AWAY].copy()
        provided = np.full(players.shape[:2], np.nan)
        args = (players, provided, ball, np.array([run.HOME, run.AWAY]), len(ball),
                [(1, 0, len(ball) - 1)], None, SPELL_PARAMS["pitch"])
        first = run.build_labels(*args, log=lambda _: None)
        messages = []
        second = run.build_labels(*args, p=dict(run.PARAMS, speed_diagnostics=False), log=messages.append)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(any("speed diagnostics" in value or "slow threshold" in value for value in messages))


class UnseenBallMotionTests(unittest.TestCase):
    def test_walking_with_established_dead_state_stays_dead(self):
        players = linear_players([PARAMS["walking_ms"], PARAMS["walking_ms"]])
        ball = np.tile([-SPELL_PARAMS["outside_m"] / SPELL_PARAMS["pitch"][0],
                        SPELL_PARAMS["origin_y_m"] / SPELL_PARAMS["pitch"][1]], (len(players), 1))
        gap = slice(run.PARAMS["slow_win"], len(ball) - run.PARAMS["slow_win"])
        ball[gap] = np.nan
        provided = np.full(players.shape[:2], np.nan)
        prediction = run.build_labels(players, provided, ball, np.array([run.HOME, run.AWAY]), len(ball),
            [(1, 0, len(ball) - 1)], None, SPELL_PARAMS["pitch"], log=lambda _: None)
        self.assertEqual(prediction[gap.start - 1], "D")
        self.assertTrue(np.all(prediction[gap] == "D"))
        window, _ = run.player_window_speed(players, provided, SPELL_PARAMS["pitch"], run.PARAMS)
        self.assertTrue(np.all(window[gap] < run.PARAMS["slow_ms"]))

    def test_jogging_with_unseen_ball_holds_either_previous_team(self):
        for previous in (run.HOME, run.AWAY):
            with self.subTest(previous=previous):
                players = linear_players([PARAMS["jogging_ms"], PARAMS["jogging_ms"]])
                ball = players[:, previous].copy()
                gap = slice(run.PARAMS["slow_win"], len(ball) - run.PARAMS["slow_win"])
                ball[gap] = np.nan
                provided = np.full(players.shape[:2], np.nan)
                prediction = run.build_labels(players, provided, ball, np.array([run.HOME, run.AWAY]),
                    len(ball), [(1, 0, len(ball) - 1)], None, SPELL_PARAMS["pitch"], log=lambda _: None)
                self.assertEqual(prediction[gap.start - 1], run.LETTERS[previous])
                self.assertTrue(np.all(prediction[gap] == run.LETTERS[previous]))
                window, _ = run.player_window_speed(players, provided, SPELL_PARAMS["pitch"], run.PARAMS)
                self.assertTrue(np.all(window[gap] > run.PARAMS["slow_ms"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
