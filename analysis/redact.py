"""
Make the public copy of a session: the numbers, none of the pictures.

Two image streams come out of a recording session and they are not comparable.

  - WebGazer's `webgazer_gray_*` patches are 10x6 grayscale per eye. Sixty pixels, with the iris
    spanning about three of them — below any recognition threshold, and the actual input to
    WebGazer's regression. These are kept: they *are* the numbers.
  - `saccade_crop` is the raw 144x36 strip across both eyes, roughly 50 x 36 px each. That is a
    recognizable picture of somebody's eyes. It is written only when the encoder was deferred to
    a worker that missed the drain deadline, so it is empty on any machine with WebGPU — but on
    a remote study it will not always be empty. These are removed.

The order of operations matters, and it is why this is a script rather than a `del`. Run with
`--encode-crops` and the crops are put through the eye model *first*, so every frame they
covered leaves with a 128-number embedding instead of being dropped. The published dataset then
loses no frames at all: it loses the pixels those frames were made of. Without the model the
frames are dropped and counted, which is worse data and still correct.

    python redact.py session.json --out public/session.json \\
        --encode-crops ../lib/saccade/eye_embedding.onnx

What this is not: anonymisation. Gaze is behavioral and the patches carry enough per-pixel
statistics to link two sessions of the same person. This removes the recognizable images; the
rest is a data-use agreement's job.
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np

CROP_LEN = 144 * 36

# Fields removed unconditionally: these are the recognizable pictures.
IMAGE_FIELDS = ("saccade_crop", "crop_frames")

# Fields removed on request. Each is useful and each carries some linkage risk, so the tradeoff
# is stated rather than decided here.
OPTIONAL = {
    "head_landmarks": "nine facial points per frame — head-position covariate, weak face geometry",
    "head_pose": "head rotation and position per frame — a covariate, carries no face geometry",
    "user_agent": "browser and OS string — a fingerprinting surface, not needed for analysis",
    "webgpu_description": "GPU model string — a fingerprinting surface",
}


def f16_b64(rows: np.ndarray) -> str | None:
    """Encode as base64 little-endian float16, the same packing the browser writes."""
    if len(rows) == 0:
        return None
    return base64.b64encode(np.ascontiguousarray(rows, dtype="<f2").tobytes()).decode()


def unpack(b64: str | None, dtype: str, stride: int) -> np.ndarray:
    if not b64:
        return np.zeros((0, stride), dtype=np.dtype(dtype))
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, stride)


def encode_crops(model_path: str, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run raw crops through the eye model, exactly as the browser's worker would have.

    Same preprocessing as `@saccadejs/core`'s OrtEmbeddingModel and `js/embed-worker.js`: the
    0-255 bytes widened to float32, shaped (1, 36, 144, 1). Nothing scaled, nothing centered.
    """
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        raise SystemExit(
            "--encode-crops needs onnxruntime:  pip install onnxruntime\n"
            "Without it the crops cannot be turned into embeddings. Rerun without the flag to\n"
            "strip them anyway — those frames are then lost from the public copy."
        ) from None

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    names = [o.name for o in session.get_outputs()]
    weight_name = next((n for n in names[1:] if n in ("cal_weight", "weight", "quality")), None)

    embeddings, weights = [], []
    for crop in crops:
        out = session.run(None, {input_name: crop.reshape(1, 36, 144, 1).astype(np.float32)})
        embeddings.append(np.asarray(out[0]).ravel())
        weights.append(float(np.ravel(out[names.index(weight_name)])[0]) if weight_name else np.nan)
    return np.asarray(embeddings), np.asarray(weights)


def rescue_row(row: dict, model_path: str, dim: int) -> int:
    """
    Turn this row's crops into embeddings, merged into the streams already there.

    The merged frame indices are sorted, because `embedded_frames` is read as an index into
    `frame_t` and an analysis is entitled to assume it runs forward.
    """
    crop_frames = row.get("crop_frames")
    if not crop_frames or not row.get("saccade_crop"):
        return 0

    crops = unpack(row["saccade_crop"], "u1", CROP_LEN)
    if len(crops) == 0:
        return 0
    extra_e, extra_w = encode_crops(model_path, crops)

    have = unpack(row.get("saccade_embedding"), "f2", dim).astype(np.float64)
    have_w = unpack(row.get("saccade_weight"), "f2", 1).astype(np.float64).ravel()
    listed = row.get("embedded_frames")
    index = np.arange(len(have)) if listed is None else np.asarray(listed, dtype=int)

    index = np.concatenate([index, np.asarray(crop_frames, dtype=int)])
    embeddings = np.vstack([have, extra_e]) if len(have) else extra_e
    weights = np.concatenate([have_w, extra_w]) if len(have_w) else extra_w

    order = np.argsort(index)
    row["embedded_frames"] = [int(i) for i in index[order]]
    row["saccade_embedding"] = f16_b64(embeddings[order])
    row["saccade_weight"] = f16_b64(weights[order].reshape(-1, 1))
    return len(crops)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    parser.add_argument("--out", required=True, help="where to write the public copy")
    parser.add_argument("--encode-crops", metavar="MODEL.onnx", default=None,
                        help="encode crops into embeddings before removing them, so no frame "
                             "is lost (needs onnxruntime installed)")
    parser.add_argument("--strip", nargs="*", choices=sorted(OPTIONAL), default=[],
                        metavar="FIELD", help="also remove: " + ", ".join(sorted(OPTIONAL)))
    args = parser.parse_args()

    with open(args.session) as handle:
        raw = json.load(handle)
    # Keep the wrapper if there is one: the interaction log travels with the trials, and
    # unwrapping here would quietly drop it from every public copy.
    wrapper = raw if isinstance(raw, dict) else None
    rows = wrapper.get("trials", wrapper.get("data", [])) if wrapper else raw

    before = len(json.dumps(rows))
    dim = next((int(r["embedding_dim"]) for r in rows if r.get("embedding_dim")), 128)

    crop_frames_present = sum(len(r.get("crop_frames") or []) for r in rows)
    rescued = 0
    if args.encode_crops and crop_frames_present:
        print(f"encoding {crop_frames_present} crop frames through the model…")
        for row in rows:
            rescued += rescue_row(row, args.encode_crops, dim)

    removed = {field: 0 for field in IMAGE_FIELDS}
    for field in args.strip:
        removed[field] = 0
    for row in rows:
        for field in IMAGE_FIELDS:
            if row.pop(field, None) is not None:
                removed[field] += 1
        for field in args.strip:
            if row.pop(field, None) is not None:
                removed[field] += 1
        # Say so in the data rather than leaving the reader to infer it from an absence.
        row["public_release"] = True
        row["crops_encoded_before_release"] = bool(args.encode_crops) and crop_frames_present > 0

    if wrapper is not None:
        wrapper["trials"] = rows
    output = wrapper if wrapper is not None else rows
    after = len(json.dumps(output))
    with open(args.out, "w") as handle:
        json.dump(output, handle)

    print(f"\nwrote {args.out}")
    print(f"  {before / 1e6:.2f} MB -> {after / 1e6:.2f} MB")
    for field, count in removed.items():
        if count:
            note = OPTIONAL.get(field, "raw eye image data")
            print(f"  removed {field:<22} from {count:>3} rows   ({note})")
    if crop_frames_present:
        if rescued:
            print(f"  {rescued} frames kept as embeddings that would otherwise have been lost")
        else:
            print(f"  {crop_frames_present} frames LOST — rerun with --encode-crops to keep them")
    else:
        print("  no crops were present in this session (the encoder never fell behind)")

    kept = [f for f in ("webgazer_gray_left", "saccade_embedding", "head_landmarks")
            if any(r.get(f) for r in rows)]
    print(f"  image-derived fields remaining: {', '.join(kept) or 'none'}")


if __name__ == "__main__":
    main()
