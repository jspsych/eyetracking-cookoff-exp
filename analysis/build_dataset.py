"""
Build the clean version 1.4 dataset the saccade.js paper (jspsych/saccadejs-paper) analyzes.

Keeps only sessions recorded with version 1.4.x of the experiment (the main data collection),
and for each one:

  - reads the copy in `data/encoded/` when there is one, where encode_crops.py has already put
    the raw eye crops through the model, so every frame has a saccade.js embedding;
  - removes the raw eye images (`saccade_crop`, `crop_frames`), failing if any crop was never
    encoded rather than silently losing those frames;
  - replaces the Prolific IDs with pseudonyms (S001, ...) from data/pseudonyms.csv, which is private,
    and drops the Prolific study and session IDs.

Writes `<out>/sessions/<code>.json.gz` (same JSON format as the raw sessions, so the cook-off
analysis code reads them unchanged) and `<out>/participants.csv`, one row per session with four
Prolific demographics joined on (age, sex, ethnicity, country of residence), plus a
README.md copied from dataset-README.md.

    python3 build_dataset.py ../data/saccadejs-validation   # from analysis/; reads ../data

Standard library only: the crops are already encoded by encode_crops.py, so no model is needed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import re
import shutil

IMAGE_FIELDS = ("saccade_crop", "crop_frames")
HERE = os.path.dirname(os.path.abspath(__file__))
ID_FIELDS = ("prolific_pid", "prolific_study_id", "prolific_session_id")
VERSION = re.compile(r"^cookoff_v(1\.4\.\d+)_([^_]+)_")

# Prolific export columns kept in participants.csv: the four the paper reports. The rest
# are identifiers, timestamps, Prolific bookkeeping, or demographics that would make most
# participants unique in combination (country of birth, nationality, language, ...).
DEMOGRAPHICS = {
    "Age": "age",
    "Sex": "sex",
    "Ethnicity simplified": "ethnicity",
    "Country of residence": "country_of_residence",
}


def session_summary(code: str, version: str, rows: list[dict]) -> dict:
    """One row of participants.csv: the session-level facts an analysis filters on."""
    first = next((r for r in rows if r.get("trial_type") == "cookoff-capture"), rows[-1])
    stations = [r for r in rows if r.get("trial_type") == "cookoff-capture" and not r.get("practice")]
    probes = [r for r in stations if r.get("probe")]
    hits = sum(r.get("probe_outcome") == "hit" for r in probes)
    return {
        "session": code,
        "version": version,
        "n_stations": len(stations),
        "n_probes": len(probes),
        "probe_hit_rate": round(hits / len(probes), 4) if probes else None,
        "encoder_mode": first.get("encoder_mode"),
        "execution_provider": first.get("execution_provider"),
        "camera_type": first.get("camera_type"),
        "eyewear": first.get("eyewear"),
        "camera_fps": first.get("camera_fps"),
        "saccade_live_fps_median": first.get("saccade_live_fps_median"),
        "timing_lag_ms": first.get("timing_lag_ms"),
        "timing_verdict": first.get("timing_verdict"),
        "browser": first.get("browser"),
        "os": first.get("os"),
        "screen_width": first.get("screen_width"),
        "screen_height": first.get("screen_height"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output folder")
    parser.add_argument("--data", default=os.path.join(HERE, "..", "data"),
                        help="the raw session folder (default: ../data)")
    args = parser.parse_args()

    data = args.data
    with open(os.path.join(data, "pseudonyms.csv")) as handle:
        codes = {row["file"]: row["code"] for row in csv.DictReader(handle)}

    demographics = {}
    for path in glob.glob(os.path.join(data, "prolific", "*.csv")):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                demographics[row["Participant id"]] = row

    os.makedirs(os.path.join(args.out, "sessions"), exist_ok=True)
    summaries = []
    for path in sorted(glob.glob(os.path.join(data, "cookoff_v1.4.*.json"))):
        name = os.path.basename(path)
        match = VERSION.match(name)
        if not match:
            continue
        version, pid = match.groups()
        code = codes.get(name)
        if code is None:
            raise SystemExit(f"{name} has no pseudonym in pseudonyms.csv")

        encoded = os.path.join(data, "encoded", name)
        with open(encoded if os.path.exists(encoded) else path) as handle:
            session = json.load(handle)
        rows = session["trials"]

        for row in rows:
            if row.get("saccade_crop"):
                raise SystemExit(f"{name}: unencoded eye crops remain; run encode_crops.py first")
            for field in IMAGE_FIELDS + ID_FIELDS:
                row.pop(field, None)
            if "participant_id" in row:
                row["participant_id"] = code

        text = json.dumps(session)
        if pid in text:
            raise SystemExit(f"{name}: the Prolific ID still appears in the cleaned session")
        with gzip.open(os.path.join(args.out, "sessions", f"{code}.json.gz"), "wt") as handle:
            handle.write(text)

        summary = session_summary(code, version, rows)
        prolific = demographics.get(pid, {})
        summary.update({new: prolific.get(old) for old, new in DEMOGRAPHICS.items()})
        summaries.append(summary)
        print(f"{code}  v{version}  {summary['n_stations']} stations")

    with open(os.path.join(args.out, "participants.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    shutil.copy(os.path.join(HERE, "dataset-README.md"), os.path.join(args.out, "README.md"))
    print(f"\n{len(summaries)} sessions written to {args.out}")


if __name__ == "__main__":
    main()
