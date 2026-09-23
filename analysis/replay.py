"""
Replay a cook-off session through both trackers' calibration and prediction.

The session recorded what each tracker would have seen and stopped there. This is the other
half: it fits each tracker's own calibration on the same frames and measures how far each one
lands from the same held-out points.

Each tracker is replayed the way it actually works, not the way that would make the comparison
tidy:

  saccade.js  averages the embeddings collected at a calibration point into one row (weighted by
              the model's own per-frame quality score), then solves a ridge from embeddings to
              targets in viewport fractions centered on 0.5, with no intercept and lambda 1.0
              (3.0 at nine points or fewer). Prediction is a dot product.

  WebGazer    takes one feature vector per calibration click — no averaging — and solves a ridge
              from the 120 equalized patch values to raw screen pixels, no intercept, k = 1e-5.
              Its calibration window holds the most recent 50 clicks and nothing older.

On a machine with no WebGPU the encoder is deferred to a worker and given a per-station budget,
so saccade.js has an embedding for only a subset of the frames while WebGazer has features for
all of them. Comparing them on different frames would not be a comparison, so everything below
runs on the intersection: `embedded_frames` says which frames carry both, and both trackers are
fitted and scored on exactly those. Frames the worker never reached carry their raw crop
instead; `--encode-crops` runs them through the model here, and without it they are reported and
skipped rather than quietly dropped.

Two things are deliberately left out, because they smooth the estimate rather than produce it:
WebGazer's Kalman filter and saccade.js's multi-frame smoothing. Both act on a stream of live
predictions; this measures the accuracy of a single frame's prediction. Turn them on in your own
analysis if that is the question you have — the per-frame data supports it.

Usage:
    python replay.py session.json
    python replay.py session.json --drop-misses --fold-by block
    python replay.py session.json --encode-crops ../lib/saccade/eye_embedding.onnx
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ unpacking

PATCH_LEN = 60
CROP_LEN = 144 * 36


def unpack(b64: str | None, dtype: str, stride: int) -> np.ndarray:
    """One of the packed per-frame streams, as (n_frames, stride). Always little-endian."""
    if not b64:
        return np.zeros((0, stride), dtype=np.dtype(dtype))
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, stride)


def equalize_histogram(patch: np.ndarray, step: int = 5) -> np.ndarray:
    """
    webgazer 3.5.3, util.mjs: equalizeHistogram(src, step).

    The sampling is load-bearing and looks like a bug until you check: the histogram is built
    from every 5th pixel, so 12 of the 60 decide the mapping. The output is float, not uint8.
    """
    hist = np.bincount(patch[::step], minlength=256).astype(np.float64)
    hist = np.cumsum(hist) * (255.0 * step / len(patch))
    return hist[patch]


def webgazer_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """webgazer's getEyeFeats for trackEye 'both': left eye then right eye, 120 numbers."""
    out = np.empty((len(left), 2 * PATCH_LEN), dtype=np.float64)
    for i in range(len(left)):
        out[i, :PATCH_LEN] = equalize_histogram(left[i])
        out[i, PATCH_LEN:] = equalize_histogram(right[i])
    return out


# ------------------------------------------------------------------ the session


@dataclass
class Station:
    block: int
    point_set: str
    target_frac: np.ndarray  # (2,) viewport fractions
    target_px: np.ndarray  # (2,) viewport pixels
    viewport: np.ndarray  # (2,)
    embeddings: np.ndarray  # (m, d) -- one row per entry of embed_index
    weights: np.ndarray  # (m,)
    embed_index: np.ndarray  # (m,) indices into frame_t
    features: np.ndarray  # (n, 120) -- every frame; WebGazer's side is never thinned
    frame_t: np.ndarray  # (n,)
    probe_outcome: str
    probed: bool
    n_crops: int = 0
    grid_level: int | None = None  # 9, 13 or 25 for a calibration-grid point, else None

    @property
    def is_test(self) -> bool:
        """A held-out position: the validation grid or a random point, never a grid point."""
        return self.grid_level is None

    @property
    def n(self) -> int:
        """Frames both trackers have, which is the only number an analysis should use."""
        return len(self.embed_index)

    @property
    def n_all_frames(self) -> int:
        return len(self.frame_t)

    @property
    def matched_features(self) -> np.ndarray:
        """WebGazer's features on the frames saccade.js also has an embedding for."""
        return self.features[self.embed_index]


def encode_crops(model_path: str, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run raw 144x36 crops through the eye model here, the way the browser would have.

    Same preprocessing as @saccadejs/core's OrtEmbeddingModel and the embed worker: the 0-255
    bytes widened to float32, shaped (1, 36, 144, 1). Nothing scaled, nothing centered.
    """
    import onnxruntime as ort  # optional; only needed when a session has leftover crops

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    names = [o.name for o in session.get_outputs()]
    weight_name = next((n for n in names[1:] if n in ("cal_weight", "weight", "quality")), None)

    embeddings, weights = [], []
    for crop in crops:
        tensor = crop.reshape(1, 36, 144, 1).astype(np.float32)
        out = session.run(None, {input_name: tensor})
        embeddings.append(np.asarray(out[0]).ravel())
        weights.append(float(np.ravel(out[names.index(weight_name)])[0]) if weight_name else np.nan)
    return np.asarray(embeddings, dtype=np.float64), np.asarray(weights, dtype=np.float64)


GRID_OUTER = (0.05, 0.5, 0.95)
GRID_INNER = (0.275, 0.725)


def grid_level_of(row: dict) -> int | None:
    """
    Which of saccade.js's nested grids a station belongs to: 9, 13, 25, or None for a test point.

    Sessions from before the 5x5 grid carry no `grid_level`; their calibration points were the
    13-point grid, labeled `point_set: "train"`, so the level is recovered from the position.
    """
    if "grid_level" in row:
        return row["grid_level"]
    if row.get("point_set") not in ("train", "grid"):
        return None
    x, y = row["target_x_frac"], row["target_y_frac"]
    near = lambda v, grid: any(abs(v - g) < 1e-6 for g in grid)
    if near(x, GRID_OUTER) and near(y, GRID_OUTER):
        return 9
    if near(x, GRID_INNER) and near(y, GRID_INNER):
        return 13
    return 25


def load(path: str, drop_misses: bool = False, model_path: str | None = None) -> tuple[list[Station], dict]:
    with open(path) as handle:
        raw = json.load(handle)
    interactions: list = []
    if isinstance(raw, dict):  # the saved format wraps the trials alongside the interaction log
        interactions = raw.get("interactions", [])
        rows = raw.get("trials", raw.get("data", []))
    else:
        rows = raw

    session = {}
    stations: list[Station] = []
    skipped = {"practice": 0, "hidden": 0, "empty": 0, "miss": 0, "fullscreen_exited": 0,
               "window_blurred": 0, "crops_encoded": 0, "crops_unencoded": 0, "arrival_frames": 0}

    for row in rows:
        for key in (
            "participant_id", "embedding_model", "embedding_dim", "execution_provider",
            "timing_lag_ms", "timing_verdict", "warmup_fps", "video_width", "video_height",
            "encoder_mode", "encoder_budget_per_station", "encoder_worker_ms", "encoder_inline_ms",
            "webgpu_available", "webgpu_adapter", "webgpu_fallback_adapter", "webgpu_vendor",
            "saccade_live_fps_median", "saccade_live_fps_p10", "saccade_live_embed_ms_median",
            "saccade_live_backend", "ridge_predict_us", "hardware_concurrency",
        ):
            if row.get(key) is not None:
                session.setdefault(key, row[key])

        if row.get("trial_type") != "cookoff-capture":
            continue
        if row.get("practice"):
            skipped["practice"] += 1
            continue
        if row.get("page_hidden"):
            skipped["hidden"] += 1
            continue
        # Leaving fullscreen resizes the viewport, so this trial's target positions are on a
        # different screen from the ones the rest of the fit was collected on.
        if row.get("fullscreen_exited"):
            skipped["fullscreen_exited"] += 1
            continue
        # The window lost focus while still visible: nothing throttled, the camera kept
        # delivering, and the frames look perfectly healthy. The participant was looking at
        # another window. This is the failure that leaves no other trace.
        if row.get("window_blurred"):
            skipped["window_blurred"] += 1
            continue
        if not row.get("n_frames"):
            skipped["empty"] += 1
            continue
        if drop_misses and row.get("probe_outcome") == "miss":
            skipped["miss"] += 1
            continue

        dim = int(session.get("embedding_dim") or 128)
        left = unpack(row["webgazer_gray_left"], "u1", PATCH_LEN)
        right = unpack(row["webgazer_gray_right"], "u1", PATCH_LEN)
        frame_t = np.asarray(row["frame_t"], dtype=float)
        n_all = min(len(left), len(right), len(frame_t))
        if n_all == 0:
            skipped["empty"] += 1
            continue

        embeddings = unpack(row["saccade_embedding"], "f2", dim).astype(np.float64)
        weights = unpack(row["saccade_weight"], "f2", 1).astype(np.float64).ravel()

        # null means the encoder ran inline and every frame has an embedding.
        listed = row.get("embedded_frames")
        embed_index = (
            np.arange(len(embeddings)) if listed is None else np.asarray(listed, dtype=int)
        )

        # Frames the worker never reached kept their raw crop instead of an embedding.
        n_crops = 0
        crop_frames = row.get("crop_frames")
        if crop_frames and row.get("saccade_crop"):
            crops = unpack(row["saccade_crop"], "u1", CROP_LEN)
            n_crops = len(crop_frames)
            if model_path:
                extra_e, extra_w = encode_crops(model_path, crops)
                embeddings = np.vstack([embeddings, extra_e]) if len(embeddings) else extra_e
                weights = np.concatenate([weights, extra_w])
                embed_index = np.concatenate([embed_index, np.asarray(crop_frames, dtype=int)])
                skipped["crops_encoded"] += n_crops
            else:
                skipped["crops_unencoded"] += n_crops

        # Keep only frames that exist on both sides, then sort so frame order is time order.
        keep = (embed_index < n_all) & (embed_index >= 0)
        keep &= np.arange(len(embed_index)) < len(embeddings)
        embed_index, embeddings = embed_index[keep], embeddings[keep]
        weights = weights[keep] if len(weights) >= len(keep) else np.ones(len(embed_index))
        order = np.argsort(embed_index)
        embed_index, embeddings, weights = embed_index[order], embeddings[order], weights[order]
        if len(embed_index) == 0:
            skipped["empty"] += 1
            continue

        # Frames before the settle ended are the eye arriving, not fixating. Sessions recorded
        # without `record_settle` have no such frames and `fixation_from` is absent, so this is
        # a no-op on them. Letting arrival frames into the fit would calibrate partly on where
        # the participant used to be looking.
        fixation_from = row.get("fixation_from")
        if fixation_from:
            keep_frames = np.arange(n_all) >= fixation_from
            drop = ~keep_frames
            if drop.any():
                skipped["arrival_frames"] += int(drop.sum())
            in_fixation = np.isin(embed_index, np.nonzero(keep_frames)[0])
            embed_index, embeddings = embed_index[in_fixation], embeddings[in_fixation]
            if len(weights) >= len(in_fixation):
                weights = weights[: len(in_fixation)][in_fixation]
            if len(embed_index) == 0:
                skipped["empty"] += 1
                continue

        viewport = np.array([row["viewport_width"], row["viewport_height"]], dtype=float)
        stations.append(
            Station(
                block=int(row["block"]),
                point_set=row.get("point_set", ""),
                target_frac=np.array([row["target_x_frac"], row["target_y_frac"]]),
                target_px=np.array([row["target_x"], row["target_y"]], dtype=float),
                viewport=viewport,
                embeddings=embeddings,
                weights=np.nan_to_num(weights, nan=1.0),
                embed_index=embed_index,
                features=webgazer_features(left[:n_all], right[:n_all]),
                frame_t=frame_t[:n_all],
                probe_outcome=row.get("probe_outcome", ""),
                probed=bool(row.get("probe")),
                n_crops=n_crops,
                grid_level=grid_level_of(row),
            )
        )
    session["skipped"] = skipped
    session["interactions"] = interactions
    return stations, session


# ------------------------------------------------------------------ the two fits


def ridge_no_intercept(X: np.ndarray, Y: np.ndarray, penalty: float) -> np.ndarray:
    """Both trackers solve the same system: (X'X + kI) B = X'Y, with no intercept column."""
    gram = X.T @ X + penalty * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ Y)


def fit_saccade(train: list[Station], frames: str = "mean", penalty: float | None = None,
                normalize_weights: bool = False) -> np.ndarray:
    """
    saccade.js: one weighted-mean embedding per calibration point, targets as viewport
    fractions centered on 0.5, lambda by point count.

    `frames="mean"` is what saccade.js does: `runCalibration` holds each point for a capture
    window and averages the embeddings it collects, weighted by the model's own per-frame
    quality score.

    `frames="middle"` takes a single frame per point instead. That is not how saccade.js
    calibrates, and it is here as a control: WebGazer calibrates from single frames, because a
    click happens at an instant, so the default comparison gives saccade.js an averaged
    calibration row and WebGazer a single-frame one. That asymmetry is faithful to both tools
    and it favors saccade.js. This flag removes it, and the difference between the two runs is
    how much of saccade.js's margin is the averaging rather than the embedding.

    `penalty` overrides saccade.js's lambda rule. `normalize_weights` rescales the row weights to
    mean 1 before the solve. saccade.js does not: its rows are weighted by the model's quality
    score, which averages about 0.4, and that shrinks the data term while leaving the penalty
    alone -- so the fit behaves as if lambda were about 2.5x what it is set to. Normalizing
    keeps the weights' job (trusting good points over bad) without that side effect.
    """
    rows, targets, row_weights = [], [], []
    for station in train:
        w = station.weights
        total = w.sum()
        if frames == "middle":
            times = station.frame_t[station.embed_index]
            k = int(np.argmin(np.abs(times - times.mean())))
            mean = station.embeddings[k]
            row_weights.append(float(w[k]) if total > 1e-6 else 1.0)
        else:
            # grids.ts falls back to the plain mean when every frame scored ~0.
            mean = (station.embeddings * w[:, None]).sum(0) / total if total > 1e-6 else station.embeddings.mean(0)
            row_weights.append(w.mean() if total > 1e-6 else 1.0)
        rows.append(mean)
        targets.append(station.target_frac - 0.5)

    X = np.asarray(rows)
    Y = np.asarray(targets)
    row_weights = np.asarray(row_weights)
    if normalize_weights:
        row_weights = row_weights / row_weights.mean()
    s = np.sqrt(row_weights + 1e-7)[:, None]
    if penalty is None:
        # lambdaFor counts distinct targets, not rows: the grid collected in two blocks is
        # still one grid's worth of points.
        n_targets = len({tuple(np.round(st.target_frac, 6)) for st in train})
        penalty = 3.0 if n_targets <= 9 else 1.0
    return ridge_no_intercept(X * s, Y * s, penalty)


def predict_saccade(kernel: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Back to viewport fractions."""
    return embeddings @ kernel + 0.5


def fit_webgazer(train: list[Station], window: int = 50, frame: str = "middle") -> np.ndarray:
    """
    WebGazer: one sample per calibration click, in raw screen pixels, k = 1e-5.

    `frame` chooses which camera frame stands in for the click. A real click lands at one
    instant, so "middle" — the frame nearest the center of the capture window — is the closest
    analogue. "all" uses every frame of every calibration point instead.

    Watch the interaction between "all" and `window`. WebGazer's calibration store is a
    DataWindow of 50 *samples*, so feeding it every frame does not give it more calibration: it
    gives it the last 50 frames, which come from the last two points, and the fit collapses.
    That is WebGazer's real behavior, not an artefact of this script. Pass `window=0` to lift
    the cap and ask the different question — how WebGazer would do with a calibration it was
    never designed to hold.
    """
    X, Y = [], []
    for station in train:
        matched = station.matched_features
        if frame == "all":
            X.append(matched)
            Y.append(np.repeat(station.target_px[None, :], len(matched), axis=0))
        else:
            times = station.frame_t[station.embed_index]
            k = int(np.argmin(np.abs(times - times.mean())))
            X.append(matched[k : k + 1])
            Y.append(station.target_px[None, :])

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    # util_regression.InitRegression: the click window is a DataWindow of 50, so an older
    # calibration point is not in the fit at all.
    if window and len(X) > window:
        X, Y = X[-window:], Y[-window:]
    return ridge_no_intercept(X, Y, 1e-5)


def predict_webgazer(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    return features @ coefficients


# ------------------------------------------------------------------ evaluation


def evaluate(stations: list[Station], fold_by: str, wg_frame: str, wg_window: int,
             sacc_frame: str = "mean", collect: list | None = None) -> dict:
    """Leave-one-fold-out over blocks, predicting every held-out frame from both fits."""
    folds = sorted({s.block for s in stations})
    if len(folds) < 2:
        raise SystemExit("need at least two blocks to hold one out")

    errors = {"saccade": [], "webgazer": []}
    per_fold = []

    for held in folds:
        train = [s for s in stations if s.block != held]
        test = [s for s in stations if s.block == held]
        if fold_by == "point_set":
            train = [s for s in train if not s.is_test] or train

        kernel = fit_saccade(train, frames=sacc_frame)
        coefficients = fit_webgazer(train, window=wg_window, frame=wg_frame)

        fold_errors = {"saccade": [], "webgazer": []}
        for station in test:
            viewport = station.viewport
            # Both are scored in the same units: pixels of this participant's viewport.
            # Both scored on the same frames: the ones saccade.js has an embedding for.
            sacc_px = predict_saccade(kernel, station.embeddings) * viewport
            wg_px = predict_webgazer(coefficients, station.matched_features)
            fold_errors["saccade"].append(np.linalg.norm(sacc_px - station.target_px, axis=1))
            fold_errors["webgazer"].append(np.linalg.norm(wg_px - station.target_px, axis=1))

            if collect is not None:
                collect.append({
                    "block": int(station.block),
                    "point_set": station.point_set,
                    "target": [round(float(station.target_px[0])), round(float(station.target_px[1]))],
                    "viewport": [int(viewport[0]), int(viewport[1])],
                    "probe_outcome": station.probe_outcome,
                    "saccade": [[round(float(x), 1), round(float(y), 1)] for x, y in sacc_px],
                    "webgazer": [[round(float(x), 1), round(float(y), 1)] for x, y in wg_px],
                })

        summary = {"block": held, "n_train": len(train), "n_test": len(test)}
        for name in errors:
            joined = np.concatenate(fold_errors[name])
            errors[name].append(joined)
            summary[name] = float(np.median(joined))
        per_fold.append(summary)

    return {
        "per_fold": per_fold,
        "overall": {
            name: {
                "median_px": float(np.median(np.concatenate(v))),
                "mean_px": float(np.mean(np.concatenate(v))),
                "p90_px": float(np.percentile(np.concatenate(v), 90)),
                "n_frames": int(len(np.concatenate(v))),
            }
            for name, v in errors.items()
        },
    }


def score(stations: list[Station], kernel: np.ndarray, coefficients: np.ndarray) -> dict:
    """Per-frame error of both fits on these stations, in px and as a fraction of the diagonal."""
    sacc, wg, diag = [], [], []
    for station in stations:
        sacc_px = predict_saccade(kernel, station.embeddings) * station.viewport
        wg_px = predict_webgazer(coefficients, station.matched_features)
        sacc.append(np.linalg.norm(sacc_px - station.target_px, axis=1))
        wg.append(np.linalg.norm(wg_px - station.target_px, axis=1))
        diag.append(np.full(station.n, np.hypot(*station.viewport)))
    sacc, wg, diag = np.concatenate(sacc), np.concatenate(wg), np.concatenate(diag)
    return {
        "saccade_px": float(np.median(sacc)),
        "webgazer_px": float(np.median(wg)),
        "saccade_diag": float(np.median(sacc / diag)),
        "webgazer_diag": float(np.median(wg / diag)),
        "n_stations": len(stations),
        "n_frames": int(len(sacc)),
    }


def forward(stations: list[Station], levels: tuple[int, ...], wg_frame: str, wg_window: int,
            sacc_frame: str = "mean") -> list[dict]:
    """
    Calibrate the way a live study does -- once, on a grid, then predict what comes after.

    For each grid size, fit both trackers on one block's grid points (`grid_level <= n`) and
    score them on the test points of that block and every later one. Calibrating on block 0
    and testing on block 2 is the drift a single calibration suffers; calibrating on block 2
    itself is what recalibrating there would have bought. Nothing is ever fitted on a block
    later than the one it is tested on.
    """
    blocks = sorted({s.block for s in stations})
    rows = []
    for n in levels:
        for cal_block in blocks:
            cal = [s for s in stations if s.block == cal_block and s.grid_level is not None
                   and s.grid_level <= n]
            if not any(s.grid_level == n for s in cal):
                continue  # this session did not collect a grid that size
            # Fewer than n when a station was dropped (a hidden tab, a miss with --drop-misses).
            n_targets = len({tuple(np.round(s.target_frac, 6)) for s in cal})
            kernel = fit_saccade(cal, frames=sacc_frame)
            coefficients = fit_webgazer(cal, window=wg_window, frame=wg_frame)
            for test_block in blocks:
                if test_block < cal_block:
                    continue
                test = [s for s in stations if s.block == test_block and s.is_test]
                if not test:
                    continue
                rows.append({"grid": n, "cal_points": n_targets, "cal_block": cal_block,
                             "test_block": test_block, **score(test, kernel, coefficients)})
    return rows


def lambda_sweep(stations: list[Station], levels: tuple[int, ...], lambdas: list[float],
                 sacc_frame: str = "mean") -> list[dict]:
    """
    saccade.js's error by lambda and grid size, on the deployment split only: calibrate on
    block 0's grid, test on every block's test points. Once with the row weights as saccade.js
    uses them and once normalized to mean 1, because the weights change what a lambda means.
    """
    cal_block = min(s.block for s in stations)
    test = [s for s in stations if s.is_test]
    rows = []
    for n in levels:
        cal = [s for s in stations if s.block == cal_block and s.grid_level is not None
               and s.grid_level <= n]
        if not any(s.grid_level == n for s in cal) or not test:
            continue
        for lam in lambdas:
            for normalize in (False, True):
                kernel = fit_saccade(cal, frames=sacc_frame, penalty=lam, normalize_weights=normalize)
                errors = np.concatenate([
                    np.linalg.norm(predict_saccade(kernel, st.embeddings) * st.viewport - st.target_px, axis=1)
                    for st in test])
                rows.append({"grid": n, "lambda": lam, "normalized": normalize,
                             "saccade_px": float(np.median(errors))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="a saved session JSON")
    parser.add_argument("--drop-misses", action="store_true",
                        help="exclude stations where a color change went unnoticed")
    parser.add_argument("--fold-by", choices=["block", "point_set"], default="block")
    parser.add_argument("--export", metavar="OUT.json", default=None,
                        help="also write every held-out prediction to this file, for plotting")
    parser.add_argument("--encode-crops", metavar="MODEL.onnx", default=None,
                        help="encode any frames the browser's worker could not reach, using "
                             "this .onnx (needs onnxruntime installed)")
    parser.add_argument("--webgazer-frames", choices=["middle", "all"], default="middle",
                        help="'middle' mimics one click per calibration point (default)")
    parser.add_argument("--saccade-frames", choices=["mean", "middle"], default="mean",
                        help="'mean' is saccade.js's own calibration (default); 'middle' uses "
                             "one frame per point, matching how WebGazer calibrates")
    parser.add_argument("--grids", default="9,13,25",
                        help="calibration grid sizes for the forward-only analysis (default "
                             "9,13,25; each is saccade.js's grid of that size, nested)")
    parser.add_argument("--lambdas", default=None, metavar="0.1,0.3,1,3",
                        help="also sweep saccade.js's ridge penalty over these values, with the "
                             "row weights as shipped and normalized to mean 1")
    parser.add_argument("--webgazer-window", type=int, default=50,
                        help="WebGazer's calibration store, in samples; 0 lifts the cap "
                             "(default 50, which is what WebGazer actually keeps)")
    args = parser.parse_args()

    stations, session = load(args.session, drop_misses=args.drop_misses,
                            model_path=args.encode_crops)
    if not stations:
        raise SystemExit("no usable stations in that file")

    frames = sum(s.n for s in stations)
    all_frames = sum(s.n_all_frames for s in stations)
    print(f"participant   {session.get('participant_id', '?')}")
    print(f"model         {session.get('embedding_model', '?')} on {session.get('execution_provider', '?')}")
    print(f"camera        {session.get('video_width')}x{session.get('video_height')}, "
          f"this recorder ran at ~{session.get('warmup_fps')} fps")


    # What live tracking would have done, which is a different number and the one people ask
    # for. A fallback WebGPU adapter is software: it reports as WebGPU and performs like wasm.
    gpu = "no WebGPU"
    if session.get("webgpu_available"):
        if session.get("webgpu_fallback_adapter"):
            gpu = "WebGPU (software fallback adapter)"
        elif session.get("webgpu_adapter"):
            gpu = f"WebGPU ({session.get('webgpu_vendor') or 'unknown vendor'})"
        else:
            gpu = "WebGPU present, no adapter"
    print(f"machine       {gpu}, {session.get('hardware_concurrency')} cores")
    def num(key, digits=1):
        """Round for display. These arrive as raw floats and print at full precision."""
        value = session.get(key)
        return "?" if value is None else f"{value:.{digits}f}"

    print(f"live tracking saccade.js managed {num('saccade_live_fps_median')} fps "
          f"(p10 {num('saccade_live_fps_p10')}) on {session.get('saccade_live_backend')}, "
          f"{num('saccade_live_embed_ms_median')} ms/frame in the model")
    if session.get("ridge_predict_us") is not None:
        print(f"              the ridge prediction itself: {num('ridge_predict_us', 2)} us/frame")
    print(f"timing        lag {session.get('timing_lag_ms')} ms, verdict {session.get('timing_verdict')}")
    print(f"encoder       {session.get('encoder_mode', 'inline')}"
          + (f", {session.get('encoder_budget_per_station')} frames/station budget"
             if session.get('encoder_budget_per_station') else ""))
    print(f"stations      {len(stations)} kept, {frames} frames with both trackers "
          f"({all_frames} with WebGazer alone)")
    print(f"skipped       {session['skipped']}")
    events = session.get("interactions") or []
    if events:
        counts = Counter(e.get("event") for e in events)
        print(f"interactions  {dict(counts)}")
    sizes = {tuple(s.viewport.astype(int)) for s in stations}
    if len(sizes) > 1:
        print(f"              WARNING: {len(sizes)} different viewport sizes across kept trials "
              f"{sorted(sizes)}.\n              Positions from different sizes are not comparable; "
              f"check fullscreen_exited and window resizing.")
    if session["skipped"]["crops_unencoded"]:
        print(f"              {session['skipped']['crops_unencoded']} frames carry a raw crop and "
              f"no embedding.\n              Pass --encode-crops <model.onnx> to use them.")

    probed = [s for s in stations if s.probed]
    if probed:
        hits = sum(s.probe_outcome == "hit" for s in probed)
        print(f"attention     {hits}/{len(probed)} color changes caught")

    collected: list | None = [] if args.export else None
    result = evaluate(stations, args.fold_by, args.webgazer_frames, args.webgazer_window,
                      args.saccade_frames, collected)
    sacc_rule = "weighted mean over each point's frames" if args.saccade_frames == "mean" else "one frame per point"
    wg_rule = "one frame per point (a simulated click)" if args.webgazer_frames == "middle" else "every frame"
    wg_cap = f", capped at the last {args.webgazer_window}" if args.webgazer_window else ", no cap"
    print("\ncalibration   every held-in station is one calibration point, for both trackers")
    print(f"              saccade.js: {sacc_rule}")
    print(f"              webgazer:   {wg_rule}{wg_cap}")
    print("\nheld-out error, in pixels of the participant's viewport")
    print(f"{'fold':>6}  {'train':>6} {'test':>5}  {'saccade.js':>11}  {'webgazer':>9}")
    for fold in result["per_fold"]:
        print(f"{fold['block']:>6}  {fold['n_train']:>6} {fold['n_test']:>5}  "
              f"{fold['saccade']:>11.1f}  {fold['webgazer']:>9.1f}")

    levels = tuple(int(n) for n in args.grids.split(","))
    fwd = forward(stations, levels, args.webgazer_frames, args.webgazer_window, args.saccade_frames)
    if fwd:
        print("\nforward only: calibrate on one block's grid, test on the test points of that block")
        print("and every later one. Median error in px, and as a % of the viewport diagonal.")
        print(f"{'grid':>5}  {'cal':>3} -> {'test':<4}  {'saccade.js':>18}  {'webgazer':>18}")
        for r in fwd:
            print(f"{r['grid']:>5}  {r['cal_block']:>3} -> {r['test_block']:<4}  "
                  f"{r['saccade_px']:>8.1f} ({100 * r['saccade_diag']:4.1f}%)   "
                  f"{r['webgazer_px']:>8.1f} ({100 * r['webgazer_diag']:4.1f}%)"
                  + (f"   {r['cal_points']} of {r['grid']} points" if r["cal_points"] < r["grid"] else ""))

        by = {(r["grid"], r["cal_block"], r["test_block"]): r for r in fwd}
        gains = [(n, t, by[(n, 0, t)], by[(n, t, t)]) for (n, c, t) in by
                 if c == 0 and t > 0 and (n, t, t) in by]
        if gains:
            print("\nrecalibration: the same test points, from the block-0 calibration and from a")
            print("fresh one on that block's own grid (saccade.js, median px)")
            for n, t, once, fresh in gains:
                print(f"{n:>5}  block {t}   once {once['saccade_px']:6.1f}   recalibrated "
                      f"{fresh['saccade_px']:6.1f}   gain {once['saccade_px'] - fresh['saccade_px']:+6.1f}")
    else:
        print("\nforward only: no block has both grid points and test points; skipped")

    sweep = []
    if args.lambdas:
        sweep = lambda_sweep(stations, levels, [float(v) for v in args.lambdas.split(",")],
                             args.saccade_frames)
        print("\nsaccade.js by ridge penalty: calibrate on block 0's grid, test on every block's")
        print("test points (median px). 'as shipped' weights rows by the model's quality score;")
        print("'normalized' rescales those weights to mean 1 first.")
        print(f"{'grid':>5}  {'lambda':>7}  {'as shipped':>10}  {'normalized':>10}")
        by = {(r["grid"], r["lambda"], r["normalized"]): r["saccade_px"] for r in sweep}
        for n, lam in dict.fromkeys((r["grid"], r["lambda"]) for r in sweep):
            print(f"{n:>5}  {lam:>7g}  {by[(n, lam, False)]:>10.1f}  {by[(n, lam, True)]:>10.1f}")

    if args.export:
        with open(args.export, "w") as handle:
            json.dump({
                "participant": session.get("participant_id"),
                "fold_by": args.fold_by,
                "saccade_frames": args.saccade_frames,
                "webgazer_frames": args.webgazer_frames,
                "webgazer_window": args.webgazer_window,
                "overall": result["overall"],
                "per_fold": result["per_fold"],
                "forward": fwd,
                "lambda_sweep": sweep,
                "stations": collected,
            }, handle)
        print(f"\nwrote {len(collected)} held-out stations to {args.export}")

    print("\noverall")
    for name, stats in result["overall"].items():
        print(f"  {name:<10} median {stats['median_px']:7.1f} px   "
              f"mean {stats['mean_px']:7.1f}   p90 {stats['p90_px']:7.1f}   "
              f"({stats['n_frames']} frames)")


if __name__ == "__main__":
    main()
