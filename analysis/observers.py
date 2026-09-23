"""
What the recording says about the observer and their setup, beyond where they looked.

Every frame already carries enough to describe the conditions it was collected under, and none
of it needed an extra measurement: the eye patches give illumination, the eye boxes give head
position and viewing distance, the landmarks give head pose, and the aspect ratios give eye
state. In a one-participant pilot that is a sanity check. Across participants it is the set of
covariates that explains why one session is twice as accurate as another, which is otherwise
the most frustrating thing about a webcam comparison.

Each measure is also correlated against per-frame error, because a covariate nobody relates to
the outcome is a curiosity rather than a covariate.

    python observers.py session.json

Two conventions worth stating, both of which are easy to get backwards:

  - **"left" is the subject's left**, following WebGazer's own naming. In an unmirrored camera
    frame that eye sits at *higher* image x, so head roll is the angle from the left eye to the
    right one. Taking it the other way round puts the answer near 180 degrees and makes the
    standard deviation meaningless through wraparound.
  - **Blinks are a relative drop**, not a fixed threshold. Eye aspect ratio depends on eye shape,
    camera angle and distance, so a cutoff that finds blinks in one observer finds none in the
    next. The test here is against that observer's own median.
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np

from replay import fit_saccade, load, predict_saccade


def unpack(b64: str | None, dtype: str, stride: int) -> np.ndarray:
    if not b64:
        return np.zeros((0, stride), dtype=np.dtype(dtype))
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, stride)


def gather(path: str) -> dict:
    """Per-frame covariates from every analysable capture trial, plus the block each came from."""
    with open(path) as handle:
        raw = json.load(handle)
    rows = raw.get("trials", raw.get("data", [])) if isinstance(raw, dict) else raw
    rows = [
        r for r in rows
        if r.get("trial_type") == "cookoff-capture"
        and not r.get("practice")
        and not r.get("page_hidden")
        and not r.get("window_blurred")
        and r.get("n_frames")
    ]
    if not rows:
        raise SystemExit("no analysable capture trials in that file")

    stack = lambda field, dt, stride: np.vstack(  # noqa: E731
        [unpack(r[field], dt, stride) for r in rows]
    ).astype(float)
    # head_pose arrived later than the other streams; a session without it gets NaN rows, so
    # everything below can treat "no pose" and "no pose on this frame" the same way.
    pose = np.vstack([
        unpack(r["head_pose"], "f2", 16).astype(float) if r.get("head_pose")
        else np.full((r["n_frames"], 16), np.nan)
        for r in rows
    ])
    return {
        "pose": pose,
        "left": stack("webgazer_gray_left", "u1", 60),
        "right": stack("webgazer_gray_right", "u1", 60),
        "box": stack("webgazer_eye_box", "i2", 8),
        "head": stack("head_landmarks", "f2", 27),
        "ear": stack("eye_aspect_ratio", "f2", 2),
        "weight": stack("saccade_weight", "f2", 1).ravel(),
        "block": np.concatenate([[r["block"]] * r["n_frames"] for r in rows]),
        "n_trials": len(rows),
    }


def head_pose_angles(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Yaw, pitch and roll in degrees, and z, from column-major 4x4 head-pose matrices.

    The same decomposition as saccade.js's `headPoseFromMatrix`: R = Ry(yaw) Rx(pitch) Rz(roll),
    and element (row r, column c) of a column-major matrix is m[c * 4 + r]. Which direction is
    positive follows MediaPipe's face-model axes; check a sign against a deliberate head turn
    before reading anything into it.
    """
    at = lambda r, c: pose[:, c * 4 + r]  # noqa: E731
    pitch = np.degrees(np.arcsin(np.clip(-at(1, 2), -1, 1)))
    yaw = np.degrees(np.arctan2(at(0, 2), at(2, 2)))
    roll = np.degrees(np.arctan2(at(1, 0), at(1, 1)))
    return yaw, pitch, roll, pose[:, 14]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    args = parser.parse_args()

    d = gather(args.session)
    left, right, box, head, ear = d["left"], d["right"], d["box"], d["head"], d["ear"]
    block = d["block"]
    n = len(left)

    # Box centers in camera pixels. Interocular distance stands in for viewing distance: the
    # face subtends fewer pixels the further away it is, and the eyes are a fixed span apart.
    lc = box[:, 0:2] + box[:, 2:4] / 2
    rc = box[:, 4:6] + box[:, 6:8] / 2
    iod = np.linalg.norm(lc - rc, axis=1)
    mid = (lc + rc) / 2
    roll = np.degrees(np.arctan2(lc[:, 1] - rc[:, 1], lc[:, 0] - rc[:, 0]))
    yaw, pitch, pose_roll, pose_z = head_pose_angles(d["pose"])

    print(f"{n} frames from {d['n_trials']} trials\n")

    print("ILLUMINATION")
    print(f"  eye-region brightness     {left.mean():5.1f} / 255   (left {left.mean():.0f}, "
          f"right {right.mean():.0f})")
    print(f"  within-patch contrast     {left.std(axis=1).mean():5.1f} sd")
    asym = left.mean() - right.mean()
    print(f"  left minus right          {asym:+5.1f}"
          + ("   <- lit from one side; one eye has less signal" if abs(asym) > 20 else ""))

    print("\nHEAD")
    print(f"  eye midpoint in frame     x {mid[:, 0].mean():5.1f} +- {mid[:, 0].std():4.2f} px"
          f"   y {mid[:, 1].mean():5.1f} +- {mid[:, 1].std():4.2f} px")
    print(f"  interocular distance      {iod.mean():5.1f} +- {iod.std():4.2f} px"
          f"   -> viewing distance stable to {100 * iod.std() / iod.mean():.1f}%")
    print(f"  head roll                 {roll.mean():+5.2f} +- {roll.std():4.2f} deg")
    print(f"  nose tip (normalized)     x {head[:, 0].mean():.3f} +- {head[:, 0].std():.4f}"
          f"   y {head[:, 1].mean():.3f} +- {head[:, 1].std():.4f}")
    has_pose = bool(np.isfinite(yaw).any())
    if has_pose:
        print(f"  head pose (MediaPipe)     yaw {np.nanmean(yaw):+5.1f} +- {np.nanstd(yaw):4.1f} deg"
              f"   pitch {np.nanmean(pitch):+5.1f} +- {np.nanstd(pitch):4.1f}"
              f"   roll {np.nanmean(pose_roll):+5.1f} +- {np.nanstd(pose_roll):4.1f}")
        print(f"                            distance {np.nanmean(-pose_z):5.1f} +- "
              f"{np.nanstd(pose_z):4.1f} model units (about cm, under MediaPipe's assumed camera)")
    else:
        print("  head pose                 not recorded in this session")

    print("\nEYE STATE")
    lo = ear.min(axis=1)
    threshold = 0.7 * np.median(lo)
    closed = lo < threshold
    print(f"  eye aspect ratio          {ear.mean():.3f} +- {ear.std():.3f}"
          f"   (min seen {lo.min():.3f})")
    print(f"  frames below 70% of this observer's median ({threshold:.3f}): "
          f"{closed.sum()} ({100 * closed.mean():.1f}%)")
    if closed.sum() == 0:
        print("    no blinks inside any capture window — people suppress blinking during "
              "effortful fixation,\n    and the windows are only about a second long")
    print(f"  model quality weight      median {np.median(d['weight']):.2f}"
          f"   10th pct {np.percentile(d['weight'], 10):.2f}")

    print("\nDRIFT ACROSS BLOCKS")
    print(f"  {'block':>6} {'brightness':>11} {'iod px':>8} {'mid x':>7} {'mid y':>7} {'roll':>7}"
          + (f" {'yaw':>7} {'pitch':>7}" if has_pose else ""))
    for b in sorted(set(block)):
        m = block == b
        print(f"  {int(b):>6} {left[m].mean():>11.1f} {iod[m].mean():>8.1f}"
              f" {mid[m, 0].mean():>7.1f} {mid[m, 1].mean():>7.1f} {roll[m].mean():>+7.2f}"
              + (f" {np.nanmean(yaw[m]):>+7.2f} {np.nanmean(pitch[m]):>+7.2f}" if has_pose else ""))

    # Do any of them relate to the outcome? Fitted in-sample on purpose: the question here is
    # whether a covariate tracks error within a session, not how well the model generalizes.
    stations, _ = load(args.session)
    kernel = fit_saccade(stations)
    names = ["brightness", "contrast", "iod", "roll", "mid_x", "ear", "weight"]
    if has_pose:
        names += ["yaw", "pitch"]
    err, cov = [], {k: [] for k in names}
    offset = 0
    for s in stations:
        gaze = predict_saccade(kernel, s.embeddings) * s.viewport
        e = np.linalg.norm(gaze - s.target_px, axis=1)
        idx = offset + s.embed_index
        idx = idx[idx < n]
        e = e[: len(idx)]
        if len(e) == 0:
            offset += s.n_all_frames
            continue
        err.append(e)
        cov["brightness"].append(left[idx].mean(axis=1))
        cov["contrast"].append(left[idx].std(axis=1))
        cov["iod"].append(iod[idx])
        cov["roll"].append(roll[idx])
        cov["mid_x"].append(mid[idx, 0])
        cov["ear"].append(ear[idx].mean(axis=1))
        cov["weight"].append(d["weight"][idx])
        if has_pose:
            cov["yaw"].append(yaw[idx])
            cov["pitch"].append(pitch[idx])
        offset += s.n_all_frames

    if err:
        err = np.concatenate(err)
        print(f"\nCORRELATION WITH saccade.js ERROR   ({len(err)} frames, in-sample fit)")
        for name, parts in cov.items():
            v = np.concatenate(parts)
            ok = np.isfinite(v)
            if ok.sum() < 3 or v[ok].std() < 1e-9:
                continue
            r = float(np.corrcoef(v[ok], err[ok])[0, 1])
            flag = "  <-" if abs(r) > 0.2 else ""
            print(f"  {name:<12} r = {r:+.2f}{flag}")
        print("  Within one session these are weak by construction: the conditions barely varied.")
        print("  They earn their keep across participants, where they vary a great deal.")


if __name__ == "__main__":
    main()
