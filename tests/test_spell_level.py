"""Deterministic generated scenarios; never loads match assets or label CSVs.

Run directly, or through tests/test_synthetic.py. Spell reports compare the
previous stillness weight with corroboration, on identical generated fixtures.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from evaluation import possession_metrics as metrics


# Fixture geometry and durations describe invented scenarios, not fitted model defaults.
PARAMS = dict(
    pitch=(105.3, 68.0),  # the documented pitch dimensions, used only for unit conversion
    flank_frames=4 * run.PARAMS["still_win"],  # stable context longer than both rolling windows
    brief_frames=2 * run.FPS,  # a controlled two-second turnover must remain a separate spell
    stationary_frames=4 * run.PARAMS["still_win"],  # stillness persists well beyond the detector window
    gap_frames=4 * run.PARAMS["max_gap"],  # exceeds interpolation and can contain a genuine transition
    dead_context_frames=2 * run.PARAMS["still_win"],  # observed dead evidence either side of a gap
    origin_x_m=25.0, origin_y_m=34.0,  # invented central starting point, away from pitch edges
    ball_ms=2.0,  # live ball moves above the stillness threshold, below the flight threshold
    jogging_ms=2.0,  # jogging exceeds the existing slow-player threshold
    background_offsets_m=((12.0, 12.0), (12.0, -12.0), (-12.0, 12.0), (-12.0, -12.0)),
    # Four moving background slots outnumber the two near-ball slots, remain
    # farther than opponent_distance_m, and make the described jogging median real.
    owner_distance_m=0.25, opponent_distance_m=6.25,  # clear control with an opposing player farther away
    contested_distance_m=2.25,  # nearby opponent makes stillness-only false-D evidence observable
    outside_m=2.0,  # a clearly out-of-bounds ball independently corroborates dead play
    metric_flank_frames=100, metric_event_frames=20,  # simple integer intervals with analytical IoUs
)


def labels(*spells):
    return np.concatenate([np.full(length, label, dtype="<U1") for label, length in spells])


def score(truth, prediction):
    gt = metrics.label_codes(truth)
    pred = metrics.label_codes(prediction)
    runs = metrics.valid_runs(gt)
    return metrics.spell_scores(gt, metrics.fill_abstentions(pred, runs), runs)


def fixture(truth, missing=None, stationary=None, contested=False):
    """Generate near-ball candidates plus jogging background slots; D is out of bounds."""
    n = len(truth)
    pitch = np.asarray(PARAMS["pitch"])
    time = np.arange(n) / run.FPS
    ball_m = np.column_stack([
        PARAMS["origin_x_m"] + PARAMS["ball_ms"] * time,
        np.full(n, PARAMS["origin_y_m"]),
    ])
    if stationary is not None:
        ball_m[stationary] = ball_m[stationary.start]
    ball_m[truth == "D"] = [-PARAMS["outside_m"], PARAMS["origin_y_m"]]
    players = np.repeat(ball_m[:, None, :], len((run.HOME, run.AWAY)), axis=1)
    opponent_distance = PARAMS["contested_distance_m"] if contested else PARAMS["opponent_distance_m"]
    players[:, run.HOME, 0] += np.where(truth == "H", PARAMS["owner_distance_m"], opponent_distance)
    players[:, run.AWAY, 0] += np.where(truth == "A", PARAMS["owner_distance_m"], opponent_distance)
    offsets = np.asarray(PARAMS["background_offsets_m"])
    background = np.empty((n, len(offsets), len(pitch)))
    background[:, :, 0] = PARAMS["origin_x_m"] + PARAMS["jogging_ms"] * time[:, None] + offsets[:, 0]
    background[:, :, 1] = PARAMS["origin_y_m"] + offsets[:, 1]
    players = np.concatenate([players, background], axis=1)
    speeds = np.full(players.shape[:2], PARAMS["jogging_ms"])
    ball = ball_m / pitch
    if missing is not None:
        ball[missing] = np.nan
    team = np.tile([run.HOME, run.AWAY], players.shape[1] // len((run.HOME, run.AWAY)))
    return players / pitch, speeds, ball, team


def predict(truth, *, p=run.PARAMS, **kwargs):
    players, speeds, ball, team = fixture(truth, **kwargs)
    return run.build_labels(players, speeds, ball, team, len(ball),
                            [(1, 0, len(ball) - 1)], None, PARAMS["pitch"],
                            p=p, log=lambda _: None)


class SpellMetricTests(unittest.TestCase):
    def test_brief_turnover_has_three_spells(self):
        flank, event = PARAMS["metric_flank_frames"], PARAMS["metric_event_frames"]
        truth = labels(("H", flank), ("A", event), ("H", flank))
        perfect = score(truth, truth)
        merged = score(truth, np.full_like(truth, "H"))
        self.assertEqual(perfect["n_gt"], len(("H", "A", "H")))
        self.assertEqual(perfect["mean_f1"], 1.0)
        # One long H can match only one true H at the lower IoU thresholds.
        overlap = flank / len(truth)
        expected = np.mean([2 / (merged["n_gt"] + merged["n_pred"])
                            if threshold <= overlap else 0.0 for threshold in metrics.SPELL_IOUS])
        self.assertAlmostEqual(merged["mean_f1"], expected)

    def test_false_stoppage_splits_one_spell(self):
        flank, event = PARAMS["metric_flank_frames"], PARAMS["metric_event_frames"]
        prediction = labels(("A", flank), ("D", event), ("A", flank))
        truth = np.full_like(prediction, "A")
        result = score(truth, prediction)
        gt_spells = metrics.possession_spells(metrics.label_codes(truth), [(0, len(truth))])
        pred_spells = metrics.possession_spells(metrics.label_codes(prediction), [(0, len(truth))])
        overlaps = metrics.spell_overlaps(gt_spells, pred_spells)
        self.assertTrue(all(iou == flank / len(truth) for iou, _, _ in overlaps))
        self.assertEqual(result["n_pred"], len(("A", "A")))
        for threshold, values in result["per_iou"].items():
            self.assertEqual(values["n_matched"], int(threshold <= flank / len(truth)))

    def test_gap_boundary_iou_is_measured(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("A", flank + gap // 2), ("H", flank + gap - gap // 2))
        delayed = labels(("A", flank + gap), ("H", flank))
        runs = [(0, len(truth))]
        overlaps = metrics.spell_overlaps(
            metrics.possession_spells(metrics.label_codes(truth), runs),
            metrics.possession_spells(metrics.label_codes(delayed), runs))
        self.assertEqual(len(overlaps), len(("A", "H")))
        self.assertEqual(sorted(iou for iou, _, _ in overlaps), sorted([
            (flank + gap // 2) / (flank + gap), flank / (flank + gap - gap // 2)]))
        self.assertEqual(score(truth, delayed)["mean_f1"], 1.0)


class PipelineSpellTests(unittest.TestCase):
    def report(self, name, truth, prediction, **fixture_kwargs):
        before = predict(truth, p=dict(run.PARAMS, still_w=run.PARAMS["still_w_uncorroborated"]),
                         **fixture_kwargs)
        baseline = score(truth, before)
        result = score(truth, prediction)
        print("{}: spellF1 {:.3f} -> {:.3f} | spells true {} pred {}".format(
            name, baseline["mean_f1"], result["mean_f1"], result["n_gt"], result["n_pred"]))
        return result

    def test_brief_controlled_turnovers(self):
        for first, second in (("H", "A"), ("A", "H")):
            with self.subTest(first=first):
                truth = labels((first, PARAMS["flank_frames"]),
                               (second, PARAMS["brief_frames"]),
                               (first, PARAMS["flank_frames"]))
                prediction = predict(truth)
                self.assertEqual(self.report("brief turnover " + first, truth, prediction)["mean_f1"], 1.0)
                self.assertTrue(np.array_equal(truth, prediction))

    def test_stationary_live_ball_does_not_split_a_spell(self):
        flank, event = PARAMS["flank_frames"], PARAMS["stationary_frames"]
        for team in ("A", "H"):
            with self.subTest(team=team):
                truth = labels((team, flank + event + flank))
                kwargs = dict(stationary=slice(flank, flank + event), contested=True)
                prediction = predict(truth, **kwargs)
                self.assertEqual(self.report("false stoppage " + team, truth, prediction, **kwargs)["mean_f1"], 1.0)
                self.assertTrue(np.all(prediction == team))

    def test_transition_inside_ball_gap_holds_until_observed(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("A", flank + gap // 2), ("H", flank + gap - gap // 2))
        missing = slice(flank, flank + gap)
        prediction = predict(truth, missing=missing)
        result = self.report("gap transition", truth, prediction, missing=missing)
        self.assertEqual(result["mean_f1"], 1.0)
        self.assertTrue(np.all(prediction[missing] == "A"))
        self.assertEqual(prediction[missing.stop], "H")

    def test_unseen_ball_with_jogging_players_keeps_dead_state(self):
        flank, context, gap = (PARAMS[key] for key in ("flank_frames", "dead_context_frames", "gap_frames"))
        truth = labels(("A", flank), ("D", context + gap + context), ("A", flank))
        missing = slice(flank + context, flank + context + gap)
        prediction = predict(truth, missing=missing)
        result = self.report("unseen dead ball, jogging players", truth, prediction, missing=missing)
        self.assertTrue(np.all(prediction[missing] == "D"))
        self.assertEqual(result["mean_f1"], 1.0)


class DeadBallEvidenceTests(unittest.TestCase):
    def test_stillness_ties_but_corroborated_stillness_wins(self):
        truth = labels(("A", PARAMS["stationary_frames"]))
        for slow, outside in ((False, False), (True, False), (False, True)):
            with self.subTest(slow=slow, outside=outside):
                players, speeds, ball, team = fixture(truth, stationary=slice(0, len(truth)))
                # Equal team distances give neutral live evidence, the most
                # permissive case for a false dead-ball decision.
                players[:, run.HOME] = players[:, run.AWAY]
                if slow:
                    speeds[:] = 0.0  # stationary players are unambiguously below slow_ms
                    players[:] = players[0]  # coordinates must agree with the provided zero speeds
                if outside:
                    ball[:, 0] = -PARAMS["outside_m"] / PARAMS["pitch"][0]
                logp = run.block_logp(players, speeds, ball, team, PARAMS["pitch"], run.PARAMS)
                centre = len(truth) // len(PARAMS["pitch"])
                live = logp[centre, :run.DEAD].max()
                path = run.decode_with_holds(logp, np.isfinite(ball).all(axis=1),
                    run.PARAMS["switch_pen"], run.PARAMS["dead_pen"], run.AWAY)
                if slow or outside:
                    self.assertGreater(logp[centre, run.DEAD], live)
                    self.assertEqual(path[centre], run.DEAD)
                else:
                    self.assertEqual(logp[centre, run.DEAD], live)
                    self.assertTrue(np.all(path == run.AWAY))

    def test_ablation_restores_stillness_only_false_stoppage(self):
        flank, event = PARAMS["flank_frames"], PARAMS["stationary_frames"]
        truth = labels(("A", flank + event + flank))
        p = dict(run.PARAMS, still_w=run.PARAMS["still_w_uncorroborated"])
        prediction = predict(truth, p=p, stationary=slice(flank, flank + event), contested=True)
        self.assertTrue(np.any(prediction == "D"))
        self.assertLess(score(truth, prediction)["mean_f1"], 1.0)


class MissingBallPolicyTests(unittest.TestCase):
    def test_short_interpolated_gap_still_holds_raw_previous_team(self):
        flank, gap = PARAMS["flank_frames"], run.PARAMS["max_gap"]
        for first, second in (("A", "H"), ("H", "A")):
            with self.subTest(first=first):
                truth = labels((first, flank + gap // 2), (second, flank + gap - gap // 2))
                missing = slice(flank, flank + gap)
                prediction = predict(truth, missing=missing)
                self.assertTrue(np.all(prediction[missing] == first))
                self.assertEqual(prediction[missing.stop], second)

    def test_missing_either_coordinate_holds_previous_team(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("A", flank + gap + flank))
        for coordinate in range(len(PARAMS["pitch"])):
            with self.subTest(coordinate=coordinate):
                players, speeds, ball, team = fixture(truth)
                ball[flank:flank + gap, coordinate] = np.nan
                prediction = run.build_labels(players, speeds, ball, team, len(ball),
                    [(1, 0, len(ball) - 1)], None, PARAMS["pitch"], log=lambda _: None)
                self.assertTrue(np.all(prediction == "A"))

    def test_entirely_unseen_block_abstains_without_a_prior(self):
        truth = labels(("A", PARAMS["flank_frames"]))
        prediction = predict(truth, missing=slice(None))
        self.assertTrue(np.all(prediction == "X"))
        self.assertEqual(score(truth, prediction)["n_pred"], 0)

    def test_leading_unseen_frames_abstain_until_observed(self):
        truth = labels(("A", PARAMS["flank_frames"]))
        gap = PARAMS["gap_frames"]
        prediction = predict(truth, missing=slice(0, gap))
        self.assertTrue(np.all(prediction[:gap] == "X"))
        self.assertTrue(np.all(prediction[gap:] == "A"))

    def test_halves_do_not_share_a_previous_team(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("H", flank), ("X", gap), ("A", flank))
        second = flank + gap
        players, speeds, ball, team = fixture(truth, missing=slice(second, second + gap))
        prediction = run.build_labels(players, speeds, ball, team, len(ball),
            [(1, 0, flank - 1), (2, second, len(ball) - 1)], None,
            PARAMS["pitch"], log=lambda _: None)
        self.assertTrue(np.all(prediction[:flank] == "H"))
        self.assertTrue(np.all(prediction[flank:second + gap] == "X"))
        self.assertTrue(np.all(prediction[second + gap:] == "A"))

    def test_documented_pre_kickoff_dead_state_survives_missing_restart(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("D", flank), ("A", flank))
        for missing in (slice(flank, flank + gap), slice(flank, None)):
            with self.subTest(entire_restart_unseen=missing.stop is None):
                players, speeds, ball, team = fixture(truth, missing=missing)
                prediction = run.build_labels(players, speeds, ball, team, len(ball),
                    [(2, 0, len(ball) - 1)], flank, PARAMS["pitch"], log=lambda _: None)
                stop = len(ball) if missing.stop is None else missing.stop
                self.assertTrue(np.all(prediction[:stop] == "D"))
                self.assertTrue(np.all(prediction[stop:] == "A"))

    def test_unseen_frames_ignore_emissions_and_keep_any_prior(self):
        # Even arbitrarily opposing scores cannot establish a state on unseen frames.
        n = PARAMS["gap_frames"]
        logp = np.zeros((n, len((run.HOME, run.AWAY, run.DEAD))))
        logp[:, run.HOME] = run.PARAMS["clip_m"]
        for previous in (run.HOME, run.AWAY, run.DEAD):
            with self.subTest(previous=previous):
                path = run.decode_with_holds(logp, np.zeros(n, dtype=bool),
                    run.PARAMS["switch_pen"], run.PARAMS["dead_pen"], previous)
                self.assertTrue(np.all(path == previous))

    def test_tied_emissions_keep_any_previous_state(self):
        n = PARAMS["gap_frames"]
        logp = np.zeros((n, len((run.HOME, run.AWAY, run.DEAD))))
        logp[:, run.DEAD] = run.PARAMS["dead_bias"]
        for previous in (None, run.HOME, run.AWAY, run.DEAD):
            with self.subTest(previous=previous):
                path = run.decode_with_holds(logp, np.ones(n, dtype=bool),
                    run.PARAMS["switch_pen"], run.PARAMS["dead_pen"], previous)
                self.assertTrue(np.all(path == (run.ABSTAIN if previous is None else previous)))

    def test_tied_sequence_scores_keep_previous_team_symmetrically(self):
        # A unit emission advantage accumulated exactly switch_pen times balances
        # one switch. This constructs an exact tie, rather than tuning a threshold.
        advantage = -run.PARAMS["dead_bias"]
        length = int(run.PARAMS["switch_pen"] / advantage)
        for previous, opponent in ((run.HOME, run.AWAY), (run.AWAY, run.HOME)):
            with self.subTest(previous=previous):
                logp = np.zeros((length, len((run.HOME, run.AWAY, run.DEAD))))
                logp[:, opponent] = advantage
                logp[:, run.DEAD] = run.PARAMS["dead_bias"]
                path = run.decode_with_holds(logp, np.ones(length, dtype=bool),
                    run.PARAMS["switch_pen"], run.PARAMS["dead_pen"], previous)
                self.assertTrue(np.all(path == previous))

    def test_disabling_policy_reproduces_legacy_decoder(self):
        flank, context, gap = (PARAMS[key] for key in ("flank_frames", "dead_context_frames", "gap_frames"))
        truth = labels(("A", flank), ("D", context + gap + context), ("A", flank))
        missing = slice(flank + context, flank + context + gap)
        p = dict(run.PARAMS, hold_missing_ball=False)
        players, speeds, ball, team = fixture(truth, missing=missing)
        logp = run.block_logp(players, speeds, ball, team, PARAMS["pitch"], p)
        legacy = run.LETTERS[run.viterbi(logp, p["switch_pen"], p["dead_pen"])]
        ablated = predict(truth, p=p, missing=missing)
        self.assertTrue(np.array_equal(ablated, legacy))
        self.assertTrue(np.any(ablated[missing] != "D"))

    def test_generated_csv_is_byte_identical_on_repeat(self):
        flank, gap = PARAMS["flank_frames"], PARAMS["gap_frames"]
        truth = labels(("A", flank + gap // 2), ("H", flank + gap - gap // 2))
        missing = slice(flank, flank + gap)
        first = predict(truth, missing=missing)
        second = predict(truth, missing=missing)
        self.assertTrue(np.array_equal(first, second))
        with tempfile.TemporaryDirectory(prefix="synthetic-possession-") as folder:
            first_path = os.path.join(folder, "first.csv")
            second_path = os.path.join(folder, "second.csv")
            run.write_csv(first_path, first)
            run.write_csv(second_path, second)
            with open(first_path, "rb") as first_file, open(second_path, "rb") as second_file:
                self.assertEqual(first_file.read(), second_file.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
