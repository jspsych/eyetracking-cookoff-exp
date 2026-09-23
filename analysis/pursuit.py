"""
What a timing error *costs*, shown on the moving-target block.

**This script demonstrates a consequence. It is not a way to measure the lag.** That distinction
was learned the hard way and is worth stating plainly, because the output below looks like a
measurement and two of its numbers will happily be quoted as one.

The demonstration is sound and needs almost no assumptions: apply the loopback correction, watch
the error against a moving target fall, and watch the amount it falls by grow with target speed.
That is the argument for measuring display and camera lag at all, and a stationary target cannot
make it.

The *estimation* is not sound. One observable — gaze-to-target distance — carries the hardware
lag, the eye's own pursuit latency, pursuit gain, catch-up saccades, and a tracker whose
accuracy is itself speed-dependent. Three speeds cannot identify five unknowns. On the session
this was built from, estimating the lag per speed gave 222, 120 and 154 ms for a quantity that
is one number; the pooled slope gave 107 ms only because two biases happened to cancel.

**To validate the loopback, run a laboratory eye tracker alongside the webcam instead.** Both
instruments watch the same eye make the same movement, so every biological unknown above is
common to both and cancels in the difference. What is left between the two gaze traces is
instrumental: the webcam pipeline's lag minus the lab tracker's own, which is small and
documented. That turns an unidentifiable problem into a subtraction, and it needs a handful of
sessions rather than a sample.

---

On a stationary target a stale gaze timestamp costs nothing. On a target moving at v, a
timestamp error of dt puts the gaze sample against the wrong position by v * dt — an error that
is pure bookkeeping, has nothing to do with how well the tracker sees the eye, and grows without
limit as the stimulus speeds up.

This script measures that three ways, from the same recording:

1. **The behavioral latency.** Shift the gaze timestamps by a trial dt, look up where the
   target was at each shifted time, and find the dt that minimizes the distance. That dt is the
   *total* lag between a photon leaving the screen and a gaze sample being stamped — display lag
   plus camera lag plus the eye's own pursuit latency.

2. **The optical latency.** `saccade-time-sync` measured display + camera lag directly, with a
   screen-to-webcam loopback, before the participant did anything. Subtracting it from (1)
   leaves the biological part.

   What that residual *should* be depends on the stimulus, and it is easy to quote the wrong
   number here. The familiar ~100-130 ms figure is the latency to pursuit **onset** for an
   unpredictable step-ramp. This block is not that: the target runs at constant velocity,
   reverses on a fixed schedule and repeats, which is about as predictable as motion gets, and
   pursuit of predictable motion is anticipatory — steady-state phase lag falls toward zero and
   can even lead. So a residual near zero is the expected result here, not a broken one, and it
   makes the total lag almost entirely hardware.

3. **The spatial cost.** Error against the target, with and without the correction, at each
   speed. This is the number that matters to anyone who does not care about latency as such: how
   many pixels of the measured error were never about the eye.

Two estimators, and the difference between them matters.

**Per-speed optima (the headline).** For each speed on its own, find the shift that minimizes
the distance. The location of that minimum is set by where the systematic displacement cancels,
and adding isotropic noise of any size changes its precision but not its position — so this
needs no assumption about how accurate the tracking is at different speeds. A genuine timing lag
is a constant of the hardware, so **the estimates must agree across speeds**, and their spread
is the honest error bar. If they disagree, the number is not a lag.

**Savings-versus-speed slope (secondary, and assumption-laden).** The pixels the correction
saves should grow linearly with speed, with slope equal to the lag. That is true only if the
corrected error is itself flat in speed: the fit actually recovers `lag - b`, where `b` is the
rate at which the corrected error grows. Pursuit gets harder at speed — more catch-up saccades,
more motion blur — so `b` is positive in practice and **this estimator reads low**. It is
reported with `b` beside it so the bias is visible rather than implied.

What both share: a constant offset — a miscalibration, a head shift — cannot produce error that
scales with speed, so the shape of the relationship rules that out either way.

Usage:
    python pursuit.py session.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from replay import (
    PATCH_LEN,
    Station,
    fit_saccade,
    fit_webgazer,
    load,
    predict_saccade,
    predict_webgazer,
    unpack,
    webgazer_features,
)

TRACKERS = ("saccade", "webgazer")


def load_pursuit(path: str, dim: int) -> list[dict]:
    """The pursuit trials, with their target paths decoded."""
    with open(path) as handle:
        rows = json.load(handle)
    if isinstance(rows, dict):
        rows = rows.get("trials", rows.get("data", []))

    runs = []
    for row in rows:
        if row.get("trial_type") != "cookoff-pursuit" or row.get("page_hidden"):
            continue
        if not row.get("n_frames") or not row.get("target_path"):
            continue

        path_rows = unpack(row["target_path"], "i2", 3).astype(np.float64)
        left = unpack(row["webgazer_gray_left"], "u1", PATCH_LEN)
        right = unpack(row["webgazer_gray_right"], "u1", PATCH_LEN)
        frame_t = np.asarray(row["frame_t"], dtype=float)
        n = min(len(left), len(right), len(frame_t))

        embeddings = unpack(row["saccade_embedding"], "f2", dim).astype(np.float64)
        listed = row.get("embedded_frames")
        embed_index = np.arange(len(embeddings)) if listed is None else np.asarray(listed, int)
        keep = (embed_index >= 0) & (embed_index < n) & (np.arange(len(embed_index)) < len(embeddings))
        embed_index, embeddings = embed_index[keep], embeddings[keep]

        runs.append({
            "condition": row.get("condition", ""),
            "speed": float(row.get("speed", 0.0)),
            "viewport": np.array([row["viewport_width"], row["viewport_height"]], float),
            "path_t": path_rows[:, 0],
            "path_xy": path_rows[:, 1:3],
            "frame_t": frame_t[:n],
            "embed_index": embed_index,
            "embeddings": embeddings,
            "features": webgazer_features(left[:n], right[:n]),
        })
    return runs


def target_at(run: dict, times: np.ndarray) -> np.ndarray:
    """Where the target was at each of `times`, by linear interpolation of the painted path."""
    x = np.interp(times, run["path_t"], run["path_xy"][:, 0], left=np.nan, right=np.nan)
    y = np.interp(times, run["path_t"], run["path_xy"][:, 1], left=np.nan, right=np.nan)
    return np.column_stack([x, y])


def path_velocity(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Target velocity per logged animation frame, as (vector px/s, speed px/s)."""
    dt = np.gradient(run["path_t"])
    safe = np.where(dt == 0, 1.0, dt)
    v = np.column_stack([
        np.gradient(run["path_xy"][:, 0]) / safe,
        np.gradient(run["path_xy"][:, 1]) / safe,
    ]) * 1000.0
    return v, np.linalg.norm(v, axis=1)


def steady_mask(run: dict, times: np.ndarray, floor: float = 0.5) -> np.ndarray:
    """
    Frames from the part of the run where the target is actually going round.

    The threshold is relative to the run's own median speed, not to the nominal speed. On an
    ellipse the instantaneous speed varies by design — fastest across the top and bottom,
    slowest at the sides — so an absolute cutoff would throw away the sides of every lap. What
    it does need to remove is the eased ramp at each end, where the speed passes through zero,
    and half the median separates those cleanly from the slowest part of a real lap.

    The path is differentiated rather than flagged by the plugin, so this needs no cooperation
    from the stimulus code and stays correct if the motion ever changes shape again.
    """
    _, speed = path_velocity(run)
    at_frames = np.interp(times, run["path_t"], speed, left=0.0, right=0.0)
    return at_frames >= floor * np.median(speed[speed > 0])


def decompose(run: dict, gaze: np.ndarray, times: np.ndarray, shift: float) -> dict | None:
    """
    Split the gaze error into the three things a closed path can tell apart.

    On a straight sweep a timing lag, a gain deficit and a miscalibration all leave the eye
    behind the target and nothing separates them. Going round an ellipse they point in different
    directions:

      - **tangential** — along the path. A timing lag puts the eye at an earlier phase, so this
        is negative (behind) by `speed x lag` however the target happens to be moving. Dividing
        it by the speed at that frame turns it back into milliseconds.
      - **radial** — toward the center. A gain deficit means the eye traces a smaller ellipse,
        so the error points inward and does not depend on which way round it is going.
      - **the mean raw error** — a constant screen-space offset rotates relative to the path and
        averages to nothing tangentially or radially, but survives here.

    Still shared between the first of those and the participant: the eye's own pursuit latency
    also trails the target. This separates instrument-plus-eye from gain and from offset; it does
    not separate the instrument from the eye. That needs a second instrument.
    """
    shifted = times - shift
    target = target_at(run, shifted)
    ok = np.isfinite(target[:, 0]) & steady_mask(run, shifted)
    if ok.sum() < 8:
        return None

    v, speed = path_velocity(run)
    vx = np.interp(shifted, run["path_t"], v[:, 0])
    vy = np.interp(shifted, run["path_t"], v[:, 1])
    spd = np.hypot(vx, vy)
    good = ok & (spd > 1e-6)
    if good.sum() < 8:
        return None

    error = gaze[good] - target[good]
    tangent = np.column_stack([vx[good], vy[good]]) / spd[good][:, None]

    # A radial direction only means something on a path that encloses area. On the straight
    # sweeps this block used before the ellipse, "toward the center" is along the line itself,
    # so the radial and tangential components are the same measurement wearing two hats.
    spread = run["path_xy"].std(axis=0)
    closed = spread.min() > 0.2 * spread.max()

    center = np.array([np.mean(run["path_xy"][:, 0]), np.mean(run["path_xy"][:, 1])])
    inward = center - target[good]
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)

    tangential = np.einsum("ij,ij->i", error, tangent)
    radial = np.einsum("ij,ij->i", error, inward)
    return {
        "n": int(good.sum()),
        "closed": bool(closed),
        "tangential": float(np.median(tangential)),
        "radial": float(np.median(radial)) if closed else float("nan"),
        "offset": error.mean(axis=0),
        "lag_ms": float(np.median(-tangential / spd[good]) * 1000.0),
        "speed": float(np.median(spd[good])),
    }


def gaze_for(run: dict, tracker: str, kernel, coefficients) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame predicted gaze in viewport pixels, and the frame times they belong to."""
    if tracker == "saccade":
        if len(run["embed_index"]) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        gaze = predict_saccade(kernel, run["embeddings"]) * run["viewport"]
        return gaze, run["frame_t"][run["embed_index"]]
    gaze = predict_webgazer(coefficients, run["features"])
    return gaze, run["frame_t"]


def best_lag(runs, tracker, kernel, coefficients, grid) -> tuple[float, np.ndarray]:
    """
    The timestamp shift that best lines gaze up with the target, pooled over every run.

    Pooling across speeds matters: a single speed cannot separate a lag from a constant spatial
    offset, because at one velocity both produce the same displacement. Several speeds can.
    """
    curve = np.full(len(grid), np.nan)
    for i, shift in enumerate(grid):
        residuals = []
        for run in runs:
            gaze, times = gaze_for(run, tracker, kernel, coefficients)
            if len(gaze) == 0:
                continue
            shifted = times - shift
            target = target_at(run, shifted)
            ok = np.isfinite(target[:, 0]) & steady_mask(run, shifted)
            if ok.sum() == 0:
                continue
            residuals.append(np.linalg.norm(gaze[ok] - target[ok], axis=1))
        if residuals:
            curve[i] = np.median(np.concatenate(residuals))
    if np.all(np.isnan(curve)):
        return float("nan"), curve
    return float(grid[int(np.nanargmin(curve))]), curve


def error_at(runs, tracker, kernel, coefficients, shift) -> dict[str, list]:
    """Median error per speed condition, with gaze timestamps shifted by `shift` ms."""
    by_condition: dict[str, list] = {}
    for run in runs:
        gaze, times = gaze_for(run, tracker, kernel, coefficients)
        if len(gaze) == 0:
            continue
        shifted = times - shift
        target = target_at(run, shifted)
        ok = np.isfinite(target[:, 0]) & steady_mask(run, shifted)
        if ok.sum() == 0:
            continue
        err = np.linalg.norm(gaze[ok] - target[ok], axis=1)
        by_condition.setdefault(run["condition"], []).append((run["speed"], err))
    return by_condition


def summarize(by_condition) -> list[tuple[str, float, float, int]]:
    out = []
    for condition, entries in by_condition.items():
        speed = entries[0][0]
        err = np.concatenate([e for _, e in entries])
        out.append((condition, speed, float(np.median(err)), len(err)))
    return sorted(out, key=lambda r: r[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--encode-crops", metavar="MODEL.onnx", default=None)
    parser.add_argument("--max-lag", type=float, default=400.0,
                        help="widest timestamp shift to search, in ms (default 400)")
    args = parser.parse_args()

    stations, session = load(args.session, model_path=args.encode_crops)
    if not stations:
        raise SystemExit("no usable static stations — the calibration has to come from somewhere")

    dim = int(session.get("embedding_dim") or 128)
    runs = load_pursuit(args.session, dim)
    if not runs:
        raise SystemExit("no usable pursuit trials in that file")

    # Calibrate on the *stationary* targets, exactly as both tools are meant to be calibrated,
    # and let the moving target be nothing but a test set.
    kernel = fit_saccade(stations)
    coefficients = fit_webgazer(stations)

    loopback = session.get("timing_lag_ms")
    print(f"participant   {session.get('participant_id', '?')}")
    print(f"calibration   {len(stations)} stationary stations")
    print(f"pursuit       {len(runs)} runs at "
          f"{', '.join(f'{r[1]:.0f}' for r in summarize(error_at(runs, 'saccade', kernel, coefficients, 0)))} px/s")
    print(f"loopback lag  {loopback} ms  (verdict {session.get('timing_verdict')})")

    print("\nwhere the error points, with the loopback correction applied")
    print("  tangential = along the path (negative is behind the target)  -> timing")
    print("  radial     = toward the center of the ellipse                -> pursuit gain")
    print("  offset     = mean error vector, fixed in screen coordinates  -> calibration")
    print(f"\n{'tracker':<10} {'speed':>6} {'tangential':>11} {'as lag':>8} "
          f"{'radial':>8} {'offset':>16} {'n':>6}")
    for tracker in TRACKERS:
        for run in sorted(runs, key=lambda r: r["speed"]):
            gaze, times = gaze_for(run, tracker, kernel, coefficients)
            if len(gaze) == 0:
                continue
            d = decompose(run, gaze, times, loopback or 0.0)
            if d is None:
                continue
            radial = f"{d['radial']:.0f}px" if d["closed"] else "n/a"
            print(f"{tracker:<10} {run['speed']:>6.0f} {d['tangential']:>10.0f}px "
                  f"{d['lag_ms']:>6.0f}ms {radial:>8} "
                  f"{'(' + format(d['offset'][0], '.0f') + ', ' + format(d['offset'][1], '.0f') + ')':>16} "
                  f"{d['n']:>6}")
    print("  A lag that is real reads the same in the `as lag` column at every speed.")
    print("  `n/a` in the radial column means the path did not enclose any area — an older")
    print("  straight-sweep session, where radial and tangential are the same measurement.")

    grid = np.arange(-50.0, args.max_lag + 1e-9, 2.0)

    # Per speed, independently. This is the estimate that does not assume the tracking is
    # equally accurate at every speed, and the agreement between the columns is the evidence
    # that what is being measured is a constant of the hardware rather than of the task.
    by_speed = sorted({r["speed"] for r in runs})
    print("\nbest-fitting shift, each speed on its own")
    print(f"{'tracker':<10} " + " ".join(f"{v:>8.0f}" for v in by_speed) + f"{'spread':>9} {'median':>8}")
    solo = {}
    for tracker in TRACKERS:
        per = []
        for v in by_speed:
            subset = [r for r in runs if r["speed"] == v]
            value, _ = best_lag(subset, tracker, kernel, coefficients, grid)
            per.append(value)
        solo[tracker] = np.array(per)
        spread = float(np.nanmax(per) - np.nanmin(per))
        print(f"{tracker:<10} " + " ".join(f"{p:>8.0f}" for p in per) +
              f"{spread:>9.0f} {np.nanmedian(per):>8.0f}")
    print("  A timing lag is one number, so these columns agreeing is the result; their spread")
    print("  is the uncertainty. Runs per speed is what buys that agreement — one is not enough.")

    print("\nbest-fitting timestamp shift, pooled over speeds")
    print(f"{'tracker':<10} {'total lag':>10} {'minus loopback':>15}   interpretation")
    results = {}
    for tracker in TRACKERS:
        lag, _ = best_lag(runs, tracker, kernel, coefficients, grid)
        results[tracker] = lag
        residual = lag - loopback if loopback is not None else float("nan")
        note = ""
        if np.isfinite(residual):
            # Predictable repetitive motion is pursued anticipatorily, so the residual should be
            # small. A *large* one is the interesting failure: it means either the participant
            # was not tracking or the loopback under-measured the hardware lag.
            if residual < -40:
                note = "negative: the eye led the target, or the loopback over-measured"
            elif residual <= 60:
                note = "near zero, as predictable motion should be — the lag is hardware"
            elif residual <= 150:
                note = "some pursuit lag on top of the hardware"
            else:
                note = "large: check the participant was actually tracking"
        print(f"{tracker:<10} {lag:>9.0f} ms {residual:>14.0f} ms   {note}")

    if loopback is None:
        print("\nno loopback measurement in this session; nothing to correct with")
        return

    print("\nthe cost of not correcting: median error against the target, in px")
    print(f"{'speed':>8}  {'tracker':<10} {'uncorrected':>12} {'corrected':>10} {'saved':>8}")
    saved_rows = []
    for tracker in TRACKERS:
        raw = summarize(error_at(runs, tracker, kernel, coefficients, 0.0))
        fixed = summarize(error_at(runs, tracker, kernel, coefficients, loopback))
        for (cond, speed, e_raw, _), (_, _, e_fix, _) in zip(raw, fixed):
            print(f"{speed:>7.0f}  {tracker:<10} {e_raw:>11.0f} {e_fix:>10.0f} {e_raw - e_fix:>8.0f}")
            saved_rows.append((tracker, speed, e_raw - e_fix))

    # The prediction that makes this a measurement rather than an anecdote: the pixels saved by
    # the correction must scale with speed, with slope = the lag in seconds.
    print("\npixels saved by the correction, regressed on target speed")
    print("(slope is the lag ONLY if the corrected error is flat in speed; b says how far it is not)")
    for tracker in TRACKERS:
        points = [(s, d) for t, s, d in saved_rows if t == tracker]
        if len(points) < 2:
            continue
        speeds = np.array([p[0] for p in points])
        saved = np.array([p[1] for p in points])
        slope, intercept = np.polyfit(speeds, saved, 1)

        # How fast the corrected error itself grows with speed. The slope above understates the
        # lag by roughly this much, so it is printed next to it rather than left implicit.
        fixed = summarize(error_at(runs, tracker, kernel, coefficients, loopback))
        b, _ = np.polyfit(np.array([f[1] for f in fixed]), np.array([f[2] for f in fixed]), 1)
        print(f"  {tracker:<10} slope {slope * 1000:>6.0f} ms   intercept {intercept:>6.1f} px"
              f"   b {b * 1000:>+5.0f} ms  -> bias-adjusted {(slope + b) * 1000:>5.0f} ms"
              f"   (loopback said {loopback:.0f} ms)")


if __name__ == "__main__":
    main()
