"""
Saccade latency: how long after a target appears does the eye set off for it?

Every station is already a step task. The target appears somewhere new while
the eye is still on the previous target, and the eye crosses. Recorded with `record_settle`, a
station contains the whole arrival — the eye on the old target, the saccade, the fixation — and
the gap between target onset and departure is the classic saccadic reaction time, with a
well-known value (about 200 ms for a regular visually-guided saccade) to check against.

The measurement is deliberately not a velocity threshold. Webcam gaze at 24 fps has one sample
every 42 ms and a saccade lasts 30-50, so the saccade is usually *between* two samples: any
velocity criterion is measuring the sampling grid. What survives at this rate is the **position
transition** — the eye is near the old target, then near the new one — and the crossing time is
recoverable even though the movement itself is not.

So the method here fits, per station, the moment of a step: the time that best splits the
recorded frames into "still at the old target" and "arrived at the new one". With frames every
42 ms, one station's estimate is coarse; the median over sixty of them is not.

Two things the estimate inherits and cannot separate on its own:

  - **Display and camera lag.** A gaze sample stamped at time T shows the eye as it was
    `timing_lag_ms` earlier. Left uncorrected the latency comes out that much too long. The
    loopback measured it, so it is subtracted, and the script reports both.
  - **The step's own onset.** The target appears at full opacity on the first frame, which
    is why: a stimulus that fades in over 180 ms would add an unrecoverable amount.

Usage:
    python saccades.py session.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from replay import PATCH_LEN, fit_saccade, fit_webgazer, load, predict_saccade, predict_webgazer, unpack, webgazer_features

TRACKERS = ("saccade", "webgazer")


def load_arrivals(path: str, dim: int) -> list[dict]:
    """Stations that recorded their settle and know where the eye came from."""
    with open(path) as handle:
        rows = json.load(handle)
    if isinstance(rows, dict):
        rows = rows.get("trials", rows.get("data", []))

    out = []
    for row in rows:
        if row.get("trial_type") != "cookoff-capture" or row.get("page_hidden"):
            continue
        if row.get("practice") or not row.get("n_frames"):
            continue
        # Needs both halves: frames from before the eye arrived, and a previous target to have
        # arrived *from*.
        if not row.get("fixation_from") or row.get("previous_target_x") is None:
            continue

        left = unpack(row["webgazer_gray_left"], "u1", PATCH_LEN)
        right = unpack(row["webgazer_gray_right"], "u1", PATCH_LEN)
        frame_t = np.asarray(row["frame_t"], dtype=float)
        n = min(len(left), len(right), len(frame_t))

        embeddings = unpack(row["saccade_embedding"], "f2", dim).astype(np.float64)
        listed = row.get("embedded_frames")
        idx = np.arange(len(embeddings)) if listed is None else np.asarray(listed, int)
        keep = (idx >= 0) & (idx < n) & (np.arange(len(idx)) < len(embeddings))
        idx, embeddings = idx[keep], embeddings[keep]

        out.append({
            "frame_t": frame_t[:n],
            "embed_index": idx,
            "embeddings": embeddings,
            "features": webgazer_features(left[:n], right[:n]),
            "viewport": np.array([row["viewport_width"], row["viewport_height"]], float),
            "from": np.array([row["previous_target_x"], row["previous_target_y"]], float),
            "to": np.array([row["target_x"], row["target_y"]], float),
            "settle": float(row["settle_duration"]),
        })
    return out


def crossing_time(times: np.ndarray, gaze: np.ndarray, origin, destination) -> float | None:
    """
    When the eye was halfway between the two targets.

    Gaze is projected onto the line joining them, which turns a two-dimensional arrival into one
    number running 0 (at the origin) to 1 (at the destination) and discards the perpendicular
    noise. The returned time is where that number crosses 0.5, **linearly interpolated between
    the samples either side**.

    The interpolation is the whole point. Snapping to the nearest sample instead — taking the
    midpoint of the bracketing pair — quantizes every estimate onto the 42 ms sampling grid and
    biases the median by up to a full frame, which on a synthetic session with a known answer
    showed up as a +39 ms error and an interquartile range of exactly zero. Interpolating uses
    the partial sample the saccade actually landed in, and recovers the crossing to a few ms.

    What this returns is the saccade's **midpoint**, not its onset: the eye is halfway across
    roughly half a saccade duration after it sets off, so onset is about 15-25 ms earlier for a
    movement of this size. That offset is a physical constant of the eye rather than an artefact
    of the method, and `main` states it rather than quietly subtracting a guess.

    None when the frames show no crossing — the eye was already at the destination when
    recording opened, or never arrived.
    """
    axis = destination - origin
    length = np.linalg.norm(axis)
    if length < 1e-6 or len(times) < 6:
        return None
    unit = axis / length
    projected = ((gaze - origin) @ unit) / length

    # A crossing has to start near the origin and end near the destination, or there is
    # nothing to time.
    if np.median(projected[:2]) > 0.4 or np.median(projected[-3:]) < 0.6:
        return None

    above = np.nonzero(projected >= 0.5)[0]
    if len(above) == 0:
        return None
    i = int(above[0])
    if i == 0:
        return None  # already past halfway on the first frame: the arrival was not captured
    j = i - 1

    span = projected[i] - projected[j]
    if span <= 1e-9:
        return None
    fraction = (0.5 - projected[j]) / span
    return float(times[j] + fraction * (times[i] - times[j]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--encode-crops", metavar="MODEL.onnx", default=None)
    args = parser.parse_args()

    stations, session = load(args.session, model_path=args.encode_crops)
    if not stations:
        raise SystemExit("no usable stations")

    dim = int(session.get("embedding_dim") or 128)
    arrivals = load_arrivals(args.session, dim)
    if not arrivals:
        raise SystemExit(
            "this session has no recorded arrivals.\n"
            "Saccade latency needs frames from before the eye reached the target, which means\n"
            "`design.record_settle: true` in js/config.js. A session recorded without it starts\n"
            "its capture window after the settle, by which time the saccade is long over."
        )

    kernel = fit_saccade(stations)
    coefficients = fit_webgazer(stations)
    lag = session.get("timing_lag_ms")

    print(f"participant   {session.get('participant_id', '?')}")
    print(f"calibration   {len(stations)} stations, fixation frames only")
    print(f"arrivals      {len(arrivals)} stations recorded the eye arriving")
    print(f"sampling      ~{1000 / (session.get('warmup_fps') or 24):.0f} ms between gaze frames")
    print(f"loopback lag  {lag} ms")

    print("\ntime from the target appearing to the eye being halfway across")
    print(f"{'tracker':<10} {'n':>4} {'raw':>10} {'minus loopback':>15}   interpretation")
    # Without WebGPU the encoder is deferred and budgeted: a handful of frames a station, one
    # every ~250 ms. That cannot time a 40 ms saccade -- the crossing lands late, by ~95 ms on
    # one pilot -- and since the budget moved to fixation there are no arrival frames at all.
    # WebGazer's features are cut from every frame on every machine, so it still has latency.
    deferred = session.get("encoder_mode") == "deferred"
    for tracker in TRACKERS:
        if tracker == "saccade" and deferred:
            print(f"{tracker:<10} {'—':>4}  not measured: without WebGPU saccade.js encodes only a few "
                  f"fixation frames a station,\n{'':16}too sparse to time a saccade. WebGazer "
                  f"below uses every frame.")
            continue
        latencies, amplitudes = [], []
        for a in arrivals:
            if tracker == "saccade":
                if len(a["embed_index"]) < 6:
                    continue
                gaze = predict_saccade(kernel, a["embeddings"]) * a["viewport"]
                times = a["frame_t"][a["embed_index"]]
            else:
                gaze = predict_webgazer(coefficients, a["features"])
                times = a["frame_t"]
            t = crossing_time(times, gaze, a["from"], a["to"])
            if t is not None:
                latencies.append(t)
                amplitudes.append(float(np.linalg.norm(a["to"] - a["from"])))

        if not latencies:
            print(f"{tracker:<10} {0:>4}  — no station showed a usable crossing")
            continue

        raw = float(np.median(latencies))
        corrected = raw - lag if lag is not None else float("nan")
        note = ""
        if np.isfinite(corrected):
            # ~200 ms is the textbook figure for a visually-guided saccade to an unpredictable
            # onset. Here the *timing* is predictable even though the location is not — targets
            # arrive on a fixed schedule — and temporal predictability pulls latency well below
            # that, so a practiced participant in the 140-200 ms band is expected rather than
            # surprising. Below ~90 ms the movement was anticipatory and not driven by the
            # target at all.
            onset = corrected - 20
            if onset < 90:
                note = f"onset ~{onset:.0f} ms: anticipatory, not driven by the target"
            elif onset <= 200:
                note = f"onset ~{onset:.0f} ms: normal for a predictable schedule"
            elif onset <= 300:
                note = f"onset ~{onset:.0f} ms: typical of an unpredictable onset"
            else:
                note = f"onset ~{onset:.0f} ms: slow — check for missed arrivals"
        print(f"{tracker:<10} {len(latencies):>4} {raw:>11.0f} ms {corrected:>14.0f} ms   {note}")

        if tracker == "saccade" and len(latencies) > 8:
            # A sanity check the data can fail: latency should be roughly flat in amplitude,
            # whereas anything that scales with distance is the crossing itself, not the
            # decision to make it.
            slope, _ = np.polyfit(np.array(amplitudes), np.array(latencies), 1)
            q = np.percentile(latencies, [25, 75])
            print(f"{'':<10} {'':>4}   IQR {q[0]:.0f}-{q[1]:.0f} ms, "
                  f"{slope * 500:+.0f} ms per 500 px of amplitude")

    print("\nSubtract roughly 15-25 ms more for saccade *onset*: these are halfway crossings,\n"
          "and the eye is halfway across about half a saccade duration after it sets off.")

    if lag is None:
        print("\nno loopback measurement in this session, so the raw figure is all there is —\n"
              "and it is display lag plus camera lag plus the participant, added together.")


if __name__ == "__main__":
    main()
