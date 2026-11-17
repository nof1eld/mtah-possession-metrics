"""End-to-end test on SYNTHETIC tracking (no competition data). Run: python tests/test_synthetic.py"""
import os, sys, unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from test_spell_level import score

PARAMS = dict(
    background_min=0.1, background_max=0.3,  # keep non-owners away from the centre in this fixture
    occlusion_period=10, occlusion_frames=3,  # fixed 30% occlusion pattern, not random draws
    pitch_length_m=105.3,  # documented pitch length for synthetic metre/normalised conversion
)


def make(n=20000, slots=46):
    team = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    pl = np.full((n, slots, 2), np.nan, np.float32); sp = np.full((n, slots), 2.0, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    grid = np.linspace(PARAMS["background_min"], PARAMS["background_max"], slots)
    base = np.column_stack([grid, grid[::-1]]).astype(np.float32)
    truth = np.full(n, "X", dtype="<U1")
    # possessions of 300 frames alternating H/A; dead stretch of 150 frames between
    t, who = 1000, 0
    while t + 450 < 8000:
        for f in range(t, t + 300):
            owner = np.where(team == who)[0][0]
            pl[f] = base; pl[f, owner] = [0.5 + who * 0.1, 0.5]
            ball[f] = [0.5 + who * 0.1 + 0.002, 0.5]
            if f % PARAMS["occlusion_period"] < PARAMS["occlusion_frames"]:
                ball[f] = np.nan
            truth[f] = "HA"[who]
        for f in range(t + 300, t + 450):                    # dead: still ball, slow players
            pl[f] = base; sp[f] = 0.3; ball[f] = [0.2, 0.0]; truth[f] = "D"
        t += 450; who = 1 - who
    # Background slots physically jog/walk at their stated speeds. Reflection
    # keeps them in the existing corner region, away from the central owner.
    metres = np.cumsum(sp[:, 0], dtype=np.float64) / run.FPS
    pitch_length = PARAMS["pitch_length_m"]
    span = (PARAMS["background_max"] - PARAMS["background_min"]) * pitch_length
    for slot in range(slots):
        is_owner = (slot == 0) | (slot == slots // 2)
        if not is_owner:
            phase = (base[slot, 0] - PARAMS["background_min"]) * pitch_length
            pl[:, slot, 0] = PARAMS["background_min"] + np.abs((phase + metres) % (2 * span) - span) / pitch_length
    return team, pl, sp, ball, truth

team, pl, sp, ball, truth = make()
for cache_dir in (None, ""):
    with patch.object(run, "parse_atd", return_value=(pl, sp, ball)) as parser:
        parsed = run.load_cached("synthetic-unused.atd", len(ball), cache_dir)
        parser.assert_called_once_with("synthetic-unused.atd", len(ball))
        assert all(actual is expected for actual, expected in zip(parsed, (pl, sp, ball)))
print("disabled cache: None and empty string OK")
labels = run.build_labels(pl, sp, ball, team, len(ball), [(1, 1000, 8000)], None, (105.3, 68.0))
inside = truth != "X"
acc = np.mean(labels[inside] == truth[inside])
n_true = len(run.possessions(np.array(["HAD".index(c) for c in truth[1000:8001] if c != "X"])))
n_pred = len(run.possessions(np.array(["HADX".index(c) for c in labels[1000:8001]])))
print("frame acc {:.3f} | possessions true {} pred {}".format(acc, n_true, n_pred))
before = run.build_labels(pl, sp, ball, team, len(ball), [(1, 1000, 8000)], None, (105.3, 68.0),
                         p=dict(run.PARAMS, still_w=run.PARAMS["still_w_uncorroborated"]), log=lambda _: None)
print("smoke: spellF1 {:.3f} -> {:.3f}".format(score(truth, before)["mean_f1"], score(truth, labels)["mean_f1"]))
assert labels[:1000].tolist() == ["X"] * 1000 and labels[8001:].tolist() == ["X"] * (len(labels) - 8001)
assert acc > 0.85 and abs(n_true - n_pred) <= 2
d = run.viterbi(np.zeros((10, 3)), 5, 5); assert len(d) == 10
print("OK")

# One command runs the smoke test and the focused spell-level regressions.
import test_spell_level
import test_speeds
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                        for module in (test_spell_level, test_speeds)]))
if not result.wasSuccessful():
    raise SystemExit(1)
