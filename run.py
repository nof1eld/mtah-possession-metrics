"""Ball-possession pipeline: per-frame evidence + Viterbi decoding with switch penalties.

    python run.py --out evaluation/results/my_method.csv

Reads data/tactical/BAR-ATH_tactical_atd.{xml,txt} through games.py (MTAH_DATA_DIR respected).
Deterministic: no randomness anywhere. Same data + same code => byte-identical CSV.

Idea (see docs/03_approach_playbook.md)
  1. Per frame: which team has the nearest player to the ball (signed distance gap, metres),
     down-weighted when the ball is far from everyone (probably in flight) or moving fast.
  2. Dead-ball evidence: ball almost stationary for ~2 s, players slow, ball outside the pitch.
     Player motion uses per-slot +/-12-frame displacement, frame medians and a 50-frame mean;
     slow-player cutoff is whole-match unsupervised Otsu on log speeds (fixed 1.5 m/s fallback).
  3. A 3-state Viterbi decoder (H, A, D) where changing team is expensive. This is the
     principled replacement for the baseline's majority-vote window: a hand-over needs about
     1 s of ~2 m advantage (switch_pen / tau), chosen from structure, NOT tuned on the public GT.
  4. Raw unseen-ball frames and tied evidence hold the previous state, including D.
     Leading uncertainty is X until a state is established; interpolation cannot trigger a switch.
  5. Ungraded frames -> X. Second half: graded from 110808, kick-off at 113742 -> D before it.

Every number in PARAMS is a structural default, not a fitted value. The public 5-minute GT
may only be used to check the pipeline is correct (FAQ.md): do not tune PARAMS on it.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = 25
HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
SECOND_HALF_KICKOFF = 113742  # FAQ.md: graded from 110808 but kick-off is here; before = all D

PARAMS = dict(
    hold_missing_ball=True,  # no observation or tied evidence establishes a change; retain H/A/D, or X if unknown
    computed_player_speeds=True,  # slot displacement avoids relying on provided speed-column units/quality
    fixed_slow_threshold=False,  # unsupervised whole-match Otsu by default; fixed slow_ms is the ablation/fallback
    speed_diagnostics=True,  # aggregate-only diagnostics; disable logging without changing labels
    displacement_half_frames=12,  # requested +/-12 frames: a 24-frame (0.96 s) displacement baseline at 25 fps
    rolling_valid_fraction=0.6,  # retain existing requirement: at least 60% of a centred window must be valid
    long_ball_gap_frames=50,  # requested >2 s unseen cue; diagnostic only, not a new decoder permission
    speed_percentiles=(5, 25, 50, 75, 95),  # requested aggregate distribution summary
    otsu_log_floor_ms=np.finfo(np.float64).eps,  # numerical floor for log(0), not a physical speed threshold
    otsu_constant_rtol=np.sqrt(np.finfo(np.float64).eps),  # treat roundoff-scale speed variation as constant
    max_gap=25,        # interpolate ball gaps up to 1 s
    clip_m=8.0,        # cap the distance advantage used as evidence
    tau=2.0,           # metres of advantage per unit of evidence
    switch_pen=25.0,   # ~1 s of 2 m advantage (25 frames * 2 m / tau) to hand possession over
    dead_pen=15.0,     # ~0.6 s to enter/leave dead ball
    far_m=3.0, far_w=0.3,      # nearest player farther than this => ball probably in flight
    fast_ms=12.0, fast_w=0.4,  # ball faster than this (m/s) => in flight
    still_ms=1.0, still_win=50,    # ball displacement speed < 1 m/s for 2 s => stationary
    slow_ms=1.5,                  # requested fixed ablation/fallback for walking versus jogging
    slow_win=50,                  # requested two-second mean of per-frame slot medians
    out_margin_m=0.5,              # ball this far outside the lines => out of play
    dead_bias=-1.0,
    still_w=1.0,       # cancels dead_bias: stillness alone ties neutral live evidence (hold prior state);
                       # slow players or out-of-bounds evidence must corroborate it to favour D
    still_w_uncorroborated=2.0,  # previous weight, retained only for the corroboration ablation
    slow_w=1.0, out_w=1.5,
)


# ----------------------------------------------------------------------------- loading
def team_of_slots(meta_path):
    """0 for home, 1 for away, per track slot. First <Team> in the XML is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def parse_atd(raw_path, n_frames):
    """Raw ATD -> (players_xy [n,slots,2] normalised, players_speed [n,slots], ball_xy [n,2]).

    Line format: <frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>   (ball last, NaN when unseen).
    """
    pl_xy = pl_sp = ball = None
    slots = None
    with open(raw_path) as fh:
        for line in fh:
            flat = np.array(line.strip().replace(":", ";").replace(",", ";").split(";"),
                            dtype=np.float32)
            if slots is None:
                slots = (len(flat) - 1) // 3 - 1
                pl_xy = np.full((n_frames, slots, 2), np.nan, np.float32)
                pl_sp = np.full((n_frames, slots), np.nan, np.float32)
                ball = np.full((n_frames, 2), np.nan, np.float32)
            if len(flat) != 1 + 3 * (slots + 1):
                continue
            i = int(flat[0]) - 1                      # frame_id counts from 1
            if not 0 <= i < n_frames:
                continue
            groups = flat[1:1 + 3 * slots].reshape(slots, 3)
            pl_xy[i], pl_sp[i] = groups[:, :2], groups[:, 2]
            ball[i] = flat[1 + 3 * slots:3 + 3 * slots]
    return pl_xy, pl_sp, ball


def load_cached(raw_path, n_frames, cache_dir):
    """Parse once, reuse. The cache holds derived tracking data: keep it local, never commit."""
    if not cache_dir:
        return parse_atd(raw_path, n_frames)
    st = os.stat(raw_path)
    tag = "{}_{}_{}".format(st.st_size, int(st.st_mtime), n_frames)
    path = os.path.join(cache_dir, "atd_{}.npz".format(hashlib.md5(tag.encode()).hexdigest()[:10]))
    if cache_dir and os.path.isfile(path):
        z = np.load(path)
        return z["pl_xy"], z["pl_sp"], z["ball"]
    out = parse_atd(raw_path, n_frames)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, pl_xy=out[0], pl_sp=out[1], ball=out[2])
    return out


# ----------------------------------------------------------------------------- signals
def fill_short_gaps(v, max_gap):
    """Linear interpolation of NaN gaps no longer than max_gap; longer gaps stay NaN."""
    s = pd.Series(v, dtype="float64")
    filled = np.array(s.interpolate(limit_area="inside").to_numpy(), dtype=np.float64)
    nan = np.isnan(v)
    edges = np.diff(np.concatenate(([0], nan.view(np.int8), [0])))
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if b - a > max_gap:
            filled[a:b] = np.nan
    return filled


def displacement_speed(xm, ym, half=PARAMS["displacement_half_frames"]):
    """Metres/second from finite endpoints at +/-half frames; also accepts (T, slots)."""
    n = len(xm)
    sp = np.full(np.shape(xm), np.nan)
    if n <= 2 * half:
        return sp
    xm = np.where(np.isfinite(xm), xm, np.nan)
    ym = np.where(np.isfinite(ym), ym, np.nan)
    d = np.hypot(xm[2 * half:] - xm[:-2 * half], ym[2 * half:] - ym[:-2 * half])
    sp[half:n - half] = d / (2 * half) * FPS
    return sp


def rolling(v, win, how, valid_fraction=PARAMS["rolling_valid_fraction"]):
    r = pd.Series(v).rolling(win, center=True, min_periods=max(1, int(win * valid_fraction)))
    return getattr(r, how)().to_numpy()


def player_window_speed(pl_xy, pl_sp, pitch, p):
    """Return centred window speed and count of frames with no valid slot speed.

    Default: per-slot displacement in metres -> median over valid slots ->
    centred rolling mean. The ablation retains the provided-column mean.
    Empty rows are left NaN and counted, without invoking a warning reduction.
    """
    if p["computed_player_speeds"]:
        xy = pl_xy.astype(np.float64) * np.asarray(pitch)
        speed = displacement_speed(xy[:, :, 0], xy[:, :, 1], p["displacement_half_frames"])
    else:
        speed = np.asarray(pl_sp, dtype=np.float64)
    valid = np.isfinite(speed)
    count = valid.sum(axis=1)
    has_speed = count > 0
    per_frame = np.full(len(speed), np.nan)
    if p["computed_player_speeds"]:
        per_frame[has_speed] = np.nanmedian(np.where(valid[has_speed], speed[has_speed], np.nan), axis=1)
    else:
        np.divide(np.where(valid, speed, 0.0).sum(axis=1), count,
                  out=per_frame, where=has_speed)
    return (rolling(per_frame, p["slow_win"], "mean", p["rolling_valid_fraction"]),
            int(np.sum(~has_speed)))


def otsu_speed_threshold(window_speed, p=PARAMS):
    """Exact numpy-only Otsu split on log speeds; NaN means no usable split.

    All distinct adjacent log speeds are candidate splits, avoiding a histogram
    bin hyperparameter. Choose the lowest split on exact variance ties. The
    returned m/s boundary is the geometric midpoint between adjacent classes.
    """
    speed = np.asarray(window_speed, dtype=np.float64)
    speed = speed[np.isfinite(speed) & (speed >= 0)]
    if not len(speed) or np.allclose(speed, speed[0], rtol=p["otsu_constant_rtol"],
                                    atol=p["otsu_log_floor_ms"]):
        return np.nan
    values, counts = np.unique(np.log(np.maximum(speed, p["otsu_log_floor_ms"])), return_counts=True)
    if len(values) < 2:
        return np.nan
    counts = counts.astype(np.float64)
    weight = np.cumsum(counts)[:-1]
    sums = np.cumsum(counts * values)[:-1]
    total = counts.sum()
    mean = np.sum(counts * values) / total
    between = (mean * weight - sums) ** 2 / (weight * (total - weight))
    split = int(between.argmax())
    return float(np.exp((values[split] + values[split + 1]) / 2))


def slow_threshold_for_match(windows, p=PARAMS):
    """Pool only the caller's graded, separately computed half windows; no labels."""
    pooled = np.concatenate(windows) if windows else np.empty(0)
    otsu = otsu_speed_threshold(pooled, p)
    if p["fixed_slow_threshold"]:
        return p["slow_ms"], otsu, "fixed"
    if not np.isfinite(otsu):
        return p["slow_ms"], otsu, "fallback"
    return otsu, otsu, "otsu"


def long_unseen_ball(ball, p=PARAMS):
    """True from the 51st consecutive unseen frame by default; resets per half."""
    observed = np.isfinite(ball).all(axis=1)
    indices = np.arange(len(ball))
    last_seen = np.maximum.accumulate(np.where(observed, indices, -1))
    return indices - last_seen > p["long_ball_gap_frames"]


def half_speed_diagnostics(window_speed, empty_count, cues, labels, p):
    """Only aggregate values; never return positions, frame identifiers or label runs."""
    valid = window_speed[np.isfinite(window_speed)]
    quantiles = np.percentile(valid, p["speed_percentiles"]) if len(valid) else [None] * len(p["speed_percentiles"])
    return dict(
        frames=len(labels), empty_slot_speed_frames=empty_count,
        window_speed_nan_fraction=float(np.mean(~np.isfinite(window_speed))),
        window_speed_percentiles_ms={str(q): None if value is None else float(value)
                                     for q, value in zip(p["speed_percentiles"], quantiles)},
        dead_cue_fractions={name: float(np.mean(cue)) for name, cue in cues.items()},
        d_share=float(np.mean(labels == "D")),
    )


def block_logp(pl_xy, pl_sp, ball, team, pitch, p, window_speed=None, slow_threshold=None, return_cues=False):
    """(T,3) log-scores for [H, A, D] on one block of frames."""
    L, W = pitch
    bx = fill_short_gaps(ball[:, 0].astype(np.float64), p["max_gap"]) * L
    by = fill_short_gaps(ball[:, 1].astype(np.float64), p["max_gap"]) * W
    has_ball = np.isfinite(bx) & np.isfinite(by)

    px = pl_xy[:, :, 0].astype(np.float64) * L
    py = pl_xy[:, :, 1].astype(np.float64) * W
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team == HOME].min(axis=1) if (team == HOME).any() else np.full(len(bx), np.inf)
    d_away = dist[:, team == AWAY].min(axis=1) if (team == AWAY).any() else np.full(len(bx), np.inf)

    usable = has_ball & np.isfinite(d_home) & np.isfinite(d_away)
    with np.errstate(invalid="ignore"):
        s = np.where(usable, np.clip(d_away - d_home, -p["clip_m"], p["clip_m"]), 0.0)

    speed = displacement_speed(bx, by, p["displacement_half_frames"])
    w = np.where(usable, 1.0, 0.0)
    w = np.where(np.minimum(d_home, d_away) > p["far_m"], w * p["far_w"], w)
    w = np.where(np.nan_to_num(speed, nan=0.0) > p["fast_ms"], w * p["fast_w"], w)
    ev = w * s

    still = rolling(speed, p["still_win"], "max", p["rolling_valid_fraction"]) < p["still_ms"]
    if window_speed is None:
        window_speed, _ = player_window_speed(pl_xy, pl_sp, pitch, p)
    slow = window_speed < (p["slow_ms"] if slow_threshold is None else slow_threshold)
    raw_x, raw_y = ball[:, 0].astype(np.float64) * L, ball[:, 1].astype(np.float64) * W
    m = p["out_margin_m"]
    out = (raw_x < -m) | (raw_x > L + m) | (raw_y < -m) | (raw_y > W + m)    # NaN -> False

    dead = p["dead_bias"] + p["still_w"] * still + p["slow_w"] * slow + p["out_w"] * out
    logp = np.stack([ev / (2 * p["tau"]), -ev / (2 * p["tau"]), dead], axis=1)
    if return_cues:
        return logp, dict(stillness=still, slow_players=slow, out_of_bounds=out,
                          ball_unseen_long=long_unseen_ball(ball, p))
    return logp


# ----------------------------------------------------------------------------- decoding
def viterbi(logp, switch_pen, dead_pen):
    """Legacy decoder, retained for the --no-hold-missing-ball ablation."""
    T = len(logp)
    trans = np.array([[0, -switch_pen, -dead_pen],
                      [-switch_pen, 0, -dead_pen],
                      [-dead_pen, -dead_pen, 0]], dtype=np.float64)
    score = np.empty((T, 3))
    back = np.zeros((T, 3), dtype=np.int8)
    score[0] = logp[0]
    for t in range(1, T):
        cand = score[t - 1][:, None] + trans
        back[t] = cand.argmax(0)
        score[t] = cand.max(0) + logp[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = score[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_with_holds(logp, observed, switch_pen, dead_pen, initial_state=None):
    """Decode decisive observations, then carry labels through raw gaps and ties.

    Missing frames contribute neither emissions nor transitions, including short
    interpolated gaps. A tied per-frame maximum is also insufficient to establish
    a state change. Without a prior, the first unique emission winner establishes
    the initial state; earlier frames remain X. An explicit prior (the documented
    pre-kickoff D) survives even if the block has no decisive observations.

    Keep the complete optimal trellis so exact score ties have no label-order
    bias. When optimal continuations tie, prefer the previous state. If several
    new states remain possible, carry the previous label until they resolve.
    """
    path = np.full(len(logp), ABSTAIN if initial_state is None else initial_state, dtype=np.int8)
    if not len(logp):
        return path
    decisive = (logp == logp.max(axis=1, keepdims=True)).sum(axis=1) == 1
    indices = np.flatnonzero(observed & decisive)
    if not len(indices):
        return path

    evidence = logp[indices]
    if initial_state is None:
        # The winner is unique here; argmax cannot choose a team on a tie.
        initial_state = int(evidence[0].argmax())
        offset = 0
    else:
        # Virtual prior frame preserves the known D before the second-half kickoff.
        evidence = np.vstack([np.zeros_like(evidence[0]), evidence])
        offset = 1

    n, states = evidence.shape
    trans = np.array([[0, -switch_pen, -dead_pen],
                      [-switch_pen, 0, -dead_pen],
                      [-dead_pen, -dead_pen, 0]], dtype=np.float64)
    scores = np.full((n, states), -np.inf)
    scores[0, initial_state] = evidence[0, initial_state]
    back = np.zeros((n, states, states), dtype=bool)
    for t in range(1, n):
        candidates = scores[t - 1][:, None] + trans
        best = candidates.max(axis=0)
        back[t] = candidates == best[None, :]
        scores[t] = best + evidence[t]

    # Which states can reach an optimal endpoint through optimal predecessors?
    viable = np.zeros((n, states), dtype=bool)
    viable[-1] = scores[-1] == scores[-1].max()
    for t in range(n - 1, 0, -1):
        viable[t - 1] = (back[t] & viable[t][None, :]).any(axis=1)

    chosen = np.full(n, initial_state, dtype=np.int8)
    possible = np.zeros(states, dtype=bool)
    possible[initial_state] = True
    previous = initial_state
    for t in range(1, n):
        options = (back[t] & possible[:, None]).any(axis=0) & viable[t]
        if options[previous]:
            # Staying is one of the equally optimal continuations.
            possible[:] = False
            possible[previous] = True
        elif options.sum() == 1:
            previous = int(np.flatnonzero(options)[0])
            possible = options
        else:
            # No arbitrary H/A ordering: carry the label while branches remain tied.
            possible = options
        chosen[t] = previous

    path[indices] = chosen[offset:]
    carry = np.maximum.accumulate(np.where(observed & decisive, np.arange(len(logp)), -1))
    known = carry >= 0
    path[known] = path[carry[known]]
    return path


def possessions(codes):
    """Maximal H/A runs; D and leading X are excluded. Returns (start, end_exclusive) pairs."""
    out, start = [], None
    for i, c in enumerate(codes):
        if c not in (HOME, AWAY) or (start is not None and c != codes[start]):
            if start is not None:
                out.append((start, i))
            start = i if c in (HOME, AWAY) else None
        elif start is None and c in (HOME, AWAY):
            start = i
    if start is not None:
        out.append((start, len(codes)))
    return out


# ----------------------------------------------------------------------------- main
def build_labels(pl_xy, pl_sp, ball, team, n_frames, blocks, kickoff2, pitch, p=PARAMS, log=print):
    labels = np.full(n_frames, "X", dtype="<U1")
    contexts = []
    for half, first, last in blocks:
        sl = slice(first, last + 1)
        window, empty_count = player_window_speed(pl_xy[sl], pl_sp[sl], pitch, p)
        contexts.append((half, first, last, window, empty_count))
    slow_threshold, otsu, mode = slow_threshold_for_match([item[3] for item in contexts], p)
    if p["speed_diagnostics"]:
        log("slow threshold: " + json.dumps(dict(
            otsu_ms=float(otsu) if np.isfinite(otsu) else None,
            used_ms=float(slow_threshold), mode=mode,
            source="displacement_median" if p["computed_player_speeds"] else "provided_mean"),
            sort_keys=True, allow_nan=False))
    for half, first, last, window, empty_count in contexts:
        full_sl = slice(first, last + 1)
        logp, cues = block_logp(pl_xy[full_sl], pl_sp[full_sl], ball[full_sl], team, pitch, p,
                                window_speed=window, slow_threshold=slow_threshold, return_cues=True)
        start = first
        initial_state = None
        if half == 2 and kickoff2 is not None and first < kickoff2 <= last:
            labels[first:kickoff2] = "D"
            start = kickoff2
            initial_state = DEAD
        sl = slice(start, last + 1)
        if start != first:
            # Preserve the existing ball-evidence boundary at kickoff; speed
            # windows and diagnostic denominators still cover the full graded half.
            logp = block_logp(pl_xy[sl], pl_sp[sl], ball[sl], team, pitch, p,
                              window_speed=window[start - first:], slow_threshold=slow_threshold)
        if p["hold_missing_ball"]:
            path = decode_with_holds(logp, np.isfinite(ball[sl]).all(axis=1),
                                     p["switch_pen"], p["dead_pen"], initial_state)
        else:
            path = viterbi(logp, p["switch_pen"], p["dead_pen"])
        labels[sl] = LETTERS[path]
        spells = possessions(path)
        alive = int(np.sum((path == HOME) | (path == AWAY)))
        log("half {}: {} frames | ball seen {:.1f}% | H {:.1f}% A {:.1f}% D {:.1f}% | "
            "{} possessions, mean {:.1f}s".format(
                half, last + 1 - first, 100 * float(np.mean(np.isfinite(ball[full_sl]).all(axis=1))),
                100 * float(np.mean(labels[full_sl] == "H")), 100 * float(np.mean(labels[full_sl] == "A")),
                100 * float(np.mean(labels[full_sl] == "D")), len(spells),
                alive / FPS / max(1, len(spells))))
        if p["speed_diagnostics"]:
            log("half {} speed diagnostics: {}".format(half, json.dumps(
                half_speed_diagnostics(window, empty_count, cues, labels[full_sl], p),
                sort_keys=True, allow_nan=False)))
    return labels


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output CSV, e.g. evaluation/results/my_method.csv")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"),
                    help="parsed-tracking cache ('' to disable). Local only, gitignored.")
    ap.add_argument("--no-hold-missing-ball", action="store_false", dest="hold_missing_ball",
                    default=PARAMS["hold_missing_ball"],
                    help="ablate raw-gap state holding and stable tie handling; use the legacy decoder")
    ap.add_argument("--no-dead-corroboration", action="store_const", dest="still_w",
                    const=PARAMS["still_w_uncorroborated"], default=PARAMS["still_w"],
                    help="ablate corroboration by restoring the previous stillness weight")
    ap.add_argument("--provided-player-speeds", action="store_false", dest="computed_player_speeds",
                    default=PARAMS["computed_player_speeds"],
                    help="ablate displacement speeds; use the provided-column mean")
    ap.add_argument("--fixed-slow-threshold", action="store_true", default=PARAMS["fixed_slow_threshold"],
                    help="use slow_ms (1.5 m/s) instead of whole-match Otsu")
    ap.add_argument("--no-speed-diagnostics", action="store_false", dest="speed_diagnostics",
                    default=PARAMS["speed_diagnostics"], help="suppress aggregate speed diagnostics")
    args = ap.parse_args()

    meta, raw = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n = source_frames("tactical", GAME)
    pl_xy, pl_sp, ball = load_cached(raw, n, args.cache_dir or None)
    team = team_of_slots(meta)
    if len(team) != pl_xy.shape[1]:
        raise SystemExit("slot count mismatch: metadata {} vs tracking {}".format(len(team), pl_xy.shape[1]))

    labels = build_labels(pl_xy, pl_sp, ball, team, n, half_video_blocks(GAME),
                          SECOND_HALF_KICKOFF, GAMES[GAME]["pitch"],
                          p=dict(PARAMS, hold_missing_ball=args.hold_missing_ball, still_w=args.still_w,
                                 computed_player_speeds=args.computed_player_speeds,
                                 fixed_slow_threshold=args.fixed_slow_threshold,
                                 speed_diagnostics=args.speed_diagnostics))
    write_csv(args.out, labels)
    with open(args.out, "rb") as fh:
        print("wrote {} ({} rows) sha256 {}".format(args.out, n, hashlib.sha256(fh.read()).hexdigest()[:16]))
    print("next: python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
