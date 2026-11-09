"""
Encode the eye crops in raw sessions, so the dataset has saccade.js on every frame.

A session whose CPU workers fell behind saves the frames they never reached as raw 144x36 eye
crops instead of embeddings (`crop_frames`, `saccade_crop`). Those frames have WebGazer's
features but no saccade.js embedding until the crops go through the same model offline. On a
slow machine that can be most of the session, and since the workers go through the queue in
order, whole later blocks — which leaves cross-validation across blocks with nothing to test on.

For each session in the data folder that has crops, this writes `encoded/<same name>` beside it:
the crops encoded exactly as `redact.py --encode-crops` does (same model, same preprocessing as
the browser's worker), merged into `embedded_frames` in frame order, and the crops themselves
dropped. Each row gains `offline_frames`, the frames encoded here, so an analysis of what live
tracking had can leave them out. build_dataset.py reads the encoded copy when there is one.

    python encode_crops.py ../data                  # every session with crops, once
    python encode_crops.py ../data --force          # redo the ones already encoded

Needs numpy and onnxruntime. The copies stay in the gitignored data folder: they are raw
sessions minus the pictures, not public copies (see redact.py for those).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from redact import rescue_row

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "lib", "saccade", "eye_embedding.onnx")


def encode_session(path: str, out: str, model: str) -> int:
    with open(path) as handle:
        raw = json.load(handle)
    wrapper = raw if isinstance(raw, dict) else None
    rows = wrapper.get("trials", wrapper.get("data", [])) if wrapper else raw
    dim = next((int(r["embedding_dim"]) for r in rows if r.get("embedding_dim")), 128)

    encoded = 0
    for row in rows:
        frames = list(row.get("crop_frames") or [])
        if not frames or not row.get("saccade_crop"):
            continue
        encoded += rescue_row(row, model, dim)
        row["offline_frames"] = sorted(int(i) for i in frames)
        row["crop_frames"] = None
        row["saccade_crop"] = None

    if wrapper is not None:
        wrapper["trials"] = rows
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(wrapper if wrapper is not None else rows, handle)
    return encoded


def has_crops(path: str) -> bool:
    with open(path) as handle:
        raw = json.load(handle)
    rows = raw.get("trials", raw.get("data", [])) if isinstance(raw, dict) else raw
    return any(r.get("crop_frames") and r.get("saccade_crop") for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="the raw sessions (data/)")
    parser.add_argument("--pattern", default="cookoff_*.json")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--force", action="store_true", help="re-encode sessions already done")
    args = parser.parse_args()

    for path in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
        out = os.path.join(args.folder, "encoded", os.path.basename(path))
        if not args.force and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
            continue
        if not has_crops(path):
            continue
        n = encode_session(path, out, args.model)
        print(f"{os.path.basename(path)[:40]}…  {n} crops encoded")


if __name__ == "__main__":
    main()
