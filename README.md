# The Cook-Off

A jsPsych experiment that collects what **saccade.js** and **WebGazer** would each need to
predict gaze — and deliberately predicts nothing.

Running two webcam eye trackers at once does not fit in a 30 fps budget. Each wants its own
MediaPipe face mesh, its own canvas readback, and its own prediction stack, and the loser of
that contest is whichever one the browser happens to starve. So this records the *inputs*: per
camera frame, saccade.js's eye embedding and WebGazer's two eye patches, timestamped, with the
target the participant was looking at. Calibration and prediction happen offline, afterwards,
from identical frames — which is the only way to ask both trackers the same question.

The expensive half runs once. One `drawImage`, one `getImageData`, one MediaPipe detection per
frame; both trackers' inputs are cut from that one detection. What is left is 20 ms of GPU
encode overlapping 7 ms of CPU work — roughly 50 fps of headroom on a 2024 laptop.

On a machine with no WebGPU that 20 ms becomes one to several hundred on single-threaded wasm,
which is longer than a camera frame. There the encoder moves to a Web Worker and the frame loop
stops waiting for it; see [Slow machines](#slow-machines).

## Hosting

The study runs from GitHub Pages: every push to `main` is live once the Pages build finishes.
The page is not cross-origin isolated there, and does not need to be: the CPU encoder is a pool
of single-threaded workers, each encoding its own frames, which works on any host (splitting one
frame across threads needed isolation and did not scale). `cross_origin_isolated` in the data
says whether a session had it.

1.3.0 was also served from https://cogsciresearch.vassar.edu/jdeleeuw/eyetracking-cookoff/,
which sends the COOP/COEP headers (`deploy/eyetracking-cookoff.conf`, deployed with
`scripts/deploy-vassar.sh`). That copy is no longer updated; from 1.3.1 GitHub Pages is the only
deployment. ONNX Runtime is served from `lib/ort/` (`scripts/sync-ort.sh`), so the workers do
not depend on the CDN. `python3 scripts/serve.py 8791 --isolate` sends the isolation headers
locally, and `test/pool-check.html` times the CPU encoder with 1 to 4 workers (run it in a
foreground tab).

## Running it

saccade.js is not published to npm yet, so the experiment is served from local builds:

```bash
cd ../saccadejs && npm install && npm run build   # once
cd -                                              # back here
./scripts/sync-saccade.sh                         # copies the bundles + the 20 MB .onnx into lib/
python3 scripts/serve.py                          # localhost:8777, caching off
```

`scripts/serve.py` sends `no-store`. `python3 -m http.server` sends no cache headers at all,
which lets Chrome apply its own heuristics and quietly serve you the previous version of a file
you just edited.

Then open <http://localhost:8777/index.html> in Chrome or Edge. A webcam and a secure context
(`https://` or `localhost`) are required; the browser will ask for camera permission at the
setup screen.

`scripts/sync-saccade.sh` takes the path to the saccade.js repo as an optional argument and
defaults to `../saccadejs`. `lib/` is gitignored — re-run the script after rebuilding saccade.js.

### A short build for the moving-target block

<http://localhost:8777/pursuit-only.html> — about two minutes once the model is cached.

It is not pursuit alone, because there is no such thing that can be analyzed: both trackers are
calibrated on *stationary* targets, so with no static block `pursuit.py` has nothing to fit a
kernel from and the recording cannot be turned into gaze at all. The short build keeps the
shortest path to an analyzable file — camera preview, time sync, warm-up, one 13-point
calibration block, then the pursuit runs — and drops screening, the frame-rate check, practice,
and two of the three static blocks.

It is config overrides in a second HTML file, not a second copy of the experiment, so it cannot
drift out of step with the real one. Files are saved with a `pursuit-only` prefix so a
diagnostic run is never mistaken for a session. Fullscreen stays on: the ellipse is sized from
the viewport, so testing it in a window tests different geometry.

## Deploying on Prolific

GitHub Pages serves the experiment straight from `main`. `lib/` is committed for that reason:
the page loads the saccade.js bundles and the 20 MB model from this repo. Before opening a
study:

1. **Sync saccade.js from a released commit.** Build `jspsych/saccadejs` on `main`, run
   `scripts/sync-saccade.sh`, and commit `lib/`. Every session records
   `saccade_build_version`, `saccade_build_commit` and `saccade_build_dirty`, so a session
   built from a feature branch or uncommitted changes can be identified afterwards.
2. **Fill in `js/config.js`:** `prolific.completion_code`, and `prolific.screenout_code` if the
   study pays screen-outs. DataPipe is already set up (`save.datapipe_experiment_id`).
   Someone arriving with a `PROLIFIC_PID` is turned away before anything starts until the first
   three are set, so a half-configured study cannot collect data it then loses.
3. **Point Prolific at the page with its IDs:**
   `https://jspsych.github.io/eyetracking-cookoff-exp/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
   `PROLIFIC_PID` becomes `participant_id`.
4. **Restrict devices to desktop** in Prolific, and require a webcam. The keyboard check stops
   anyone who gets through anyway.

What a participant can hit, and where it goes:

| | |
|---|---|
| no key press within 20 s | screened out (`no_keyboard`) |
| no WebGL in the browser (hardware acceleration off, GPU blocklisted) | screened out at once (`no_webgl`), before any model downloads: the face tracker cannot run without it |
| camera refused or broken, three tries | screened out (`no_camera`) |
| no face found within 60 s of the camera preview | screened out (`no_face`) |
| upload fails after four tries | offered a download of their data, then completed as normal |
| finishes | redirected to the completion code |

A screened-out participant is sent to `screenout_code`, or asked to return the study if it is
empty. What they got through is saved under a `screenout_` prefix: only the screening record,
with `screened_out` saying which check stopped them.

Uploads are gzipped. DataPipe refuses any request over 32 MB before its own code runs, and a
session is about 5 MB. A slow machine that writes raw eye crops can be several times that, and
gzip keeps it under the limit.

## The session

About ten minutes. The participant-facing copy says what to do and nothing else; the reasons
live here and in the code.

1. **Screening** — a browser record (`@jspsych/plugin-browser-check`), a space-bar press, and
   one question about the camera. All before the camera or the 20 MB model are touched, so a
   participant who cannot do the task finds out before downloading anything.

   The keyboard press is the gate, and it is behavioral on purpose: the task reports the color
   change with a key, so a device that cannot send one cannot produce the attention data. A
   user-agent test would be cheaper and wrong in both directions. `is_mobile_ua` is recorded
   beside `keyboard_present` so the two can be compared.

2. **Fullscreen** — `@jspsych/plugin-fullscreen`, entered before anything measures the viewport.
   Target positions are fractions of `window.innerWidth` and both trackers are fitted against
   them, so a viewport that changes size part-way through moves the screen out from under a
   calibration taken before it. `fullscreen: false` in `js/config.js` for piloting.
3. **Camera setup** — `saccade-preview`: camera permission, the model download behind a
   progress bar, and the eye strip the model actually sees.
4. **Warming the pans** — `saccade-performance`: runs saccade.js's *real* tracker on this camera
   for five seconds and reports the frame rate it achieved. The session never predicts gaze, so
   this is the only thing that answers "could saccade.js have tracked this participant live, and
   how fast?" — see [Could this machine have tracked live?](#could-this-machine-have-tracked-live).
5. **One measurement first** — `saccade-time-sync`: the screen steps between black and white
   while the camera watches, which measures the display + camera lag that JavaScript cannot see.
   The result is `timing_lag_ms`, the constant that puts a camera frame on the same clock as a
   stimulus onset. Measured in the wild, on the participant's own machine.
6. **Getting ready** — loads the recorder's face mesh and runs it on a live camera for
   three seconds, which is the only honest frame-rate check: the recorder does more work per
   frame than saccade.js's own tracker, so the tracker's performance trial would report a rate
   this session never sees.
7. **A practice block**, then three blocks of stations.
8. **The moving-target block**, off in the main study (`pursuit.repeats: 0`); see
   [What the timing measurement is worth](#what-the-timing-measurement-is-worth).

A trial is one fixation point. A target appears, settles to its final size, and the recorder
holds a capture window open while the participant looks at its center.

### The color-change probe

Nothing here predicts gaze in real time, so nothing can tell afterwards whether a participant
was actually looking at the point. That is what the attention probe is for.

On some stations the small dot at the center of the target — six pixels across — turns orange,
and the participant presses <kbd>space</kbd>. A change that small
cannot be caught peripherally, so a **hit is direct evidence the fovea was on the target**, and
a miss marks a capture window an analysis may want to drop. Hits, misses, false alarms and
reaction times are all recorded per station.

Decide whether you are dropping misses *before* you look at the accuracy numbers. It is exactly
the kind of filter that can be tuned, after the fact, to favor one tracker.

### The target

The capture-phase target is the concentric bullseye from Thaler et al. (2013) — a filled disc,
a cross cut through it, a small central dot — which people fixate more stably than a plain dot,
and which gives the probe somewhere to live that does not move. The themed emoji appears only
while the target is landing, and is gone before the first frame is kept.

## Could this machine have tracked live?

Every session records enough to answer this without guessing from the user agent.

**Was the hardware there.** `webgpu_available`, `webgpu_adapter`, `webgpu_vendor`,
`webgpu_architecture`, and `webgpu_fallback_adapter` — that last one matters, because a software
fallback adapter reports as WebGPU and performs like wasm, and would otherwise sit in your data
as a baffling outlier. Probed before the camera or the model are touched.

Kept deliberately separate from what ONNX Runtime *did* with it (`encoder_provider`,
`saccade_live_backend`). ORT tries WebGPU with graph capture, then WebGPU without, then wasm,
and takes the first that survives. A machine with `webgpu_available: true` and a wasm backend is
a real case — a GPU the runtime could not use — and it is invisible unless both are recorded.

**What it ran at.** The `saccade-performance` trial gives `saccade_live_fps_median` (the
headline), `saccade_live_fps_p10` (the slow tail — a machine that stalls periodically has a
healthy median and a poor p10), and `saccade_live_embed_ms_median`, which separates a slow model
from a camera that is not delivering frames.

Do not use `warmup_fps` for this. That is *this recorder's* rate, and the recorder does strictly
more per frame than live tracking: it also cuts WebGazer's patches, and on a slow machine it
moves the encoder to a worker, which live tracking cannot do. Both are recorded; the gap between
them is the price of carrying the second tracker.

**Where the time goes.** `ridge_predict_us` is the measured cost of the per-frame ridge
prediction — the dot product of the embedding with the `2 x d` kernel. It comes in around
**1 µs**, against tens of milliseconds for the encoder on WebGPU and hundreds on wasm. So yes:
the frame rate is the encoder and the face mesh, and the regression is four decimal places
below either. It is measured per machine rather than asserted, so the claim is checkable instead
of folklore.

`replay.py` prints all of this at the top of a session summary:

```
machine       no WebGPU, 4 cores
live tracking saccade.js managed 4.6 fps (p10 3.2) on wasm, 198.7 ms/frame in the model
              the ridge prediction itself: 0.37 us/frame
```

There is no exclusion rule. `saccade-performance` takes an `inclusion_function` if you want one,
but the deferred encoder exists precisely so that slow machines stay in the sample — they are
the participants where the two trackers are most likely to differ.

## Slow machines

ONNX Runtime falls back to single-threaded wasm without WebGPU, and the encoder goes from about
20 ms to one to several hundred. Run that inline and it does not merely slow the task down: it
stalls the frame loop, so MediaPipe misses frames and the capture windows come back half empty.
The participant's data ends up worse than if the encoder had never run.

So on those machines the encoder moves to a worker (`js/embed-worker.js`) and the recorder
stops waiting for it entirely. Crops are queued, the worker answers when it can, and the
results are written back into the finished trials by `station_id`.

**Every frame is queued** (from version 1.3.1), the 600 ms arrival included, so saccade.js's
saccade latency can be timed on these machines too. The worker is a pool of single-threaded
workers, each encoding its own frames (up to four; an M3 manages about 22 frames a second with
four, close to a 24 fps camera). A pool that keeps up embeds everything, catching up in the
station gaps and breaks. One that falls behind works through the queue in order, and whatever it
has not reached by the end is saved as raw crops. The analysis times saccade.js only on stations
whose whole arrival was encoded.

Before 1.3.1 the worker had a per-station budget instead: the warm-up measured an encode and
queued an evenly spaced subset of the fixation frames, four to a dozen a station on a typical
wasm machine, and saccade.js contributed no latency on those sessions.

Three things keep that from quietly becoming a bias:

- WebGazer's features are cut for **every** frame regardless — they cost almost nothing. So the
  two streams have different lengths, `embedded_frames` records which frames have both, and a
  head-to-head comparison runs on the intersection. `analysis/replay.py` does this; scoring one
  tracker on eight frames and the other on thirty would compare sample sizes, not trackers.
- Nothing is silently dropped. Frames still queued when the drain deadline expires have their
  **raw crops** written into the data instead — twenty times the bytes, identical information.
  Up to 5000 crops a session (`encoder.max_crops_saved`: crops barely gzip, and DataPipe has a 32 MB upload
  limit); past that they are thinned evenly and the dropped frames counted.
  `python analysis/replay.py session.json --encode-crops lib/saccade/eye_embedding.onnx` folds
  them back in.
- The measurement and the outcome are recorded (`encoder_mode`, `encoder_workers`,
  `encoder_worker_ms`, `frames_embedded`, `frames_left_as_crops`), so a slow session is visible
  in the data rather than inferred from the user agent afterwards.

The queue drains in the background during the task, in the fixed 10-second break between blocks
(the same length for everyone, so the task's timing never depends on the machine), and behind a
progress bar before saving. The progress bar is skipped instantly on a machine that encoded inline.

Why not just store every frame's crop and encode them all offline? Because the crops do not
compress: CLAHE amplifies sensor noise into high-entropy detail, gzip gets a ratio of about 1.1,
and a session would be roughly 12 MB instead of 1. The embedding is a 20x compression that
happens to be the thing the tracker actually uses.

Knobs are under `encoder` in `js/config.js`, including `mode: "deferred"` to force the worker
path on a machine that does not need it, which is how to exercise it without finding slow
hardware.

## Saccade latency, for free

Every station is already a step task: the target appears somewhere new while the eye is on
the previous target. With `design.record_settle` on (the default) the recorder opens when the
target *appears* rather than after it has settled, so each station captures the whole
arrival and the session yields ~60 saccade latencies at no cost in task time.

```bash
python analysis/saccades.py session.json
```

It costs about 60% more data — a station records 1600 ms instead of 1000 — and
`record_settle: false` in `js/config.js` returns to fixation-only recording.

Two details that decide whether the number means anything:

- **The onset is a step.** The target appears at full opacity on the first frame and only
  its scale springs. A stimulus that fades in over 180 ms produces a longer latency by an
  amount nothing downstream can recover.
- **The method is a position crossing, not a velocity threshold.** At 24 fps a 40 ms saccade
  usually falls *between* two samples, so a velocity criterion measures the sampling grid.
  Gaze is projected onto the line between the two targets and the halfway crossing is
  interpolated between the bracketing samples. On a synthetic session with a planted 205 ms
  latency that recovers the crossing to 1 ms; snapping to the nearest sample biased it +39 ms
  with an interquartile range of exactly zero.

The raw figure includes the display and camera lag, so the loopback correction is what turns it
into a number comparable with the ~200 ms in the literature — the same argument as the moving
target below, on a different measurement.

## What the timing measurement is worth

The static blocks cannot show it. A gaze sample that is 110 ms stale still lands on a target
that has not moved, which makes it easy to conclude the measurement changes nothing at all.

The **moving-target block** shows the consequence. A target slides across the screen at a known
speed and the participant tracks it. If gaze timestamps are wrong by `dt` and the target moves
at `v`, lining gaze up against the target introduces `v x dt` pixels of error that have nothing
to do with how well the tracker sees the eye. On one session's 110 ms lag that was 18 px at
250 px/s and **86 px at 900 px/s — more than saccade.js's entire 34 px static median error.**

```bash
python analysis/pursuit.py session.json
```

The main study leaves this block out (`pursuit.repeats: 0` in `js/config.js`): the pilot
sessions made the demonstration, more participants would not strengthen it, and the saccade
latencies already show what the loopback correction is worth, against a known physiological
range. `pursuit-only.html` still runs it. When it is on: three runs, about half a minute, after
the static blocks — so calibration still comes from stationary targets and the moving target is
purely a test set.

### Why an ellipse

On a straight sweep, a timing lag, a pursuit gain deficit and a plain miscalibration all leave
the eye trailing the target and nothing separates them. Going round a closed path they point in
different directions:

| error | direction | reads as |
|---|---|---|
| timing lag | **tangential** — behind along the path, whichever way it is going | `speed x lag` |
| gain deficit | **radial** — inward, the eye tracing a smaller ellipse | independent of direction |
| miscalibration | fixed in screen coordinates, so it rotates relative to the path | averages out tangentially and radially, survives in the mean error |

Verified on a synthetic session with all three planted at once — 150 ms of lag, gain 0.92, a
(25, −15) px offset. The decomposition returned the offset as (24, −15), (25, −15), (26, −15) at
the three speeds, and the gain as ~32 px inward, which is what 8% of that ellipse's mean radius
is. A straight sweep cannot do that at all.

The ellipse is matched to the viewport's aspect, so instantaneous speed varies continuously
around each lap — fastest across the top and bottom, slowest at the sides — which gives a range
of speeds inside every run rather than one number per run.

### What this block cannot do

It cannot *measure* the lag, and the output is written to say so. One observable — gaze-to-target
distance — carries the hardware lag, the eye's own pursuit latency, pursuit gain, catch-up
saccades, and a tracker whose accuracy is itself speed-dependent. Three speeds cannot identify
five unknowns.

The ellipse improves this and does not fix it. Offset and gain now come out cleanly, and the
tangential term is a far better-identified lag estimate than the old savings slope — but on the
same synthetic session it read 64, 25 and 47 ms for a planted 40, because at 300 px/s a 40 ms
lag is only 12 px of tangential displacement against a comparable amount of gaze noise. The
scatter is down from the old method's 102 ms range to about 39 ms; it is not down to a few ms.

The lag also still mixes the instrument with the participant: the eye's own pursuit latency
trails the target too, and no amount of geometry separates instrument-plus-eye from instrument.
That needs a second instrument.

### How to validate the loopback properly

Run a **laboratory eye tracker alongside the webcam**, on the same participant, at the same time.

Both instruments watch the same eyeball make the same movement, so pursuit latency, gain,
catch-up saccades and compliance are common to both and **cancel in the difference**. What
survives is instrumental: the webcam pipeline's lag minus the lab tracker's own, which is small
and documented. An unidentifiable inference problem becomes a subtraction.

The awkward part of that design — putting two clocks on the same timebase — is already solved in
[`tobii-integration`](https://github.com/jspsych/tobii-integration): `toLocalTime()` maps Tobii
device timestamps onto `performance.now()`, and `getTimeSyncStatus()` reports `bcStdDev` /
`bcMin` / `bcMax` so the alignment uncertainty is a measured quantity rather than an assumption.

Two things that make it cheaper than it sounds:

- **The static blocks are enough.** Cross-correlating the two gaze traces through the 129 station
  transitions estimates the lag directly; the moving-target block is not needed for it.
- **It needs sessions, not a sample.** The quantity is a property of the hardware, so the
  variance that matters is between setups, not between people.

The binding limit is the webcam's own sampling: at 23 fps a single event is located to about
43 ms, so the precision comes from averaging over many transitions — roughly 4 ms of standard
error over 129 of them, which is ample against a 110 ms figure.

## The data

One session is a JSON object with `trials` and `interactions` — jsPsych's focus/blur/fullscreen
log, which `data.get().json()` omits and which is otherwise thrown away. Roughly 2.5 MB.

Three per-trial flags mark a trial that has stopped describing the participant, and they are not
the same event: `page_hidden` (tab backgrounded, timers throttled, frames stop), `window_blurred`
(**page still visible, camera still running, frames look healthy — participant looking at
another window**), and `fullscreen_exited` (viewport resized out from under the calibration).
`replay.py` drops all three and reports the counts. Each station is one row carrying the
per-frame feature streams as base64 over typed arrays: float16 for the embeddings, **uint8 for
WebGazer's patches**, which is lossless — WebGazer equalizes each patch against its own 60
pixels and nothing else, so the stored bytes reproduce its feature vector exactly, at a third
the size of storing the equalized floats.

`docs/DATA.md` is the full dictionary. `analysis/replay.py` is a working offline replay: it
fits each tracker's own calibration and reports held-out error for both, two ways.

- **Cross-validated across blocks:** fit on two blocks, test on the third. This is the most
  data per fit, and it is not how anyone deploys a tracker.
- **Forward only:** calibrate on one block's grid, test on the test points of that block and
  every later one, for each grid size (9, 13, 25). This matches a live study, which calibrates
  once and then predicts what comes after. Block 0 → block 2 is the drift a single calibration
  suffers. Recalibrating on block 2 and testing there is what a second calibration would buy,
  and the output prints that gain directly.

```bash
python analysis/replay.py session.json              # needs numpy
python analysis/replay.py session.json --drop-misses
python analysis/replay.py session.json --lambdas 0.1,0.3,1,3   # saccade.js's ridge penalty
```

### Calibration size is a subset, not a condition

Every block collects a 5x5 grid at 5/27.5/50/72.5/95%. That grid contains both of saccade.js's
own grids exactly: its 3x3 at 5/50/95 and its default 13, which is the 3x3 plus the four points
at 27.5/72.5. Each grid station records `grid_level`, the smallest of those grids it belongs to,
so "a 13-point calibration" is `grid_level <= 13`. Everyone does the superset and the analysis
subsets it. Every comparison of calibration size is then within-subject, and no participant
sees a different task.

The test points follow the grid in each block: saccade.js's 9-point validation grid (which lets
an analysis emulate the validate plugin's score) and nine fresh random positions, one per cell
of a 3x3 over the screen. Across blocks and participants, the test set covers the whole screen
instead of the same 22 locations.

### The ridge penalty

saccade.js weights each calibration row by the model's own quality score, which averages about
0.4. That shrinks the data term and leaves the penalty alone, so the fit behaves as if lambda
were about 2.5 times its nominal value. On the three pilot sessions, which all came from one
person on one machine, the shipped defaults cost a lot. With the row weights normalized to mean
1, the 3x3 grid did as well as 13 points, both at about 26 px on block 0's calibration. As
shipped, with lambda 3 for nine points or fewer, the 3x3 was about 100 px.
`--lambdas` prints that comparison per session. The offline data put the optimum at lambda
1-3 (`eye-tracking/EXPERIMENT_LOG.md`). Whether that holds live is one of the questions this
study answers.

Data goes to DataPipe (`save.mode: "datapipe"` in `js/config.js`); `"download"` saves a file in
the browser instead. A session opened without a `PROLIFIC_PID` is a pilot. It still uploads, so
the whole path gets exercised, but under a `pilot_` filename prefix and with `pilot: true` on every
row, so a test run cannot be mistaken for a participant.

## Releasing the data

Collect everything, publish the numbers. `privacy.save_eye_crops` is on, so a slow participant's
un-encoded frames keep their raw crops in the recorded session — and the sharing decision is
made at release time instead:

```bash
python analysis/redact.py session.json --out public/session.json \
    --encode-crops lib/saccade/eye_embedding.onnx
```

The order matters, and it is why this is a script rather than deleting a field. `--encode-crops`
puts the crops through the eye model **first**, so every frame they covered leaves as a 128-number
embedding instead of being dropped. The public copy then loses no frames — only the pixels those
frames were made of. On a synthetic session with 78 crop frames the redacted copy reproduced the
original analysis exactly, to the decimal, at 0.42 MB instead of 0.93.

What goes, what stays:

| | |
|---|---|
| **removed** | `saccade_crop`, `crop_frames` — the raw 144x36 strip across both eyes, ~50x36 px each, a recognizable picture |
| **kept** | `webgazer_gray_*` — 10x6 per eye, iris about three pixels across, below any recognition threshold and the actual input to WebGazer's regression |
| **kept** | `saccade_embedding` — 128 numbers, the actual input to saccade.js's regression |
| **optional** | `--strip head_landmarks head_pose user_agent webgpu_description technical_problems interruptions` |

`head_landmarks` is the judgement call, and it is kept by default. Nine facial points is weak
face geometry but real head-position data, and head position explains a lot of the accuracy
variance — it is what showed that one session's improvement was *not* better head stability.
Strip it if your sharing agreement calls for it; you lose a useful covariate.

**This is not anonymisation, and the script says so.** Gaze is behavioral, and the patches carry
enough per-pixel statistics to link two sessions of the same person. What it removes is the
recognizable imagery. The rest is a data-use agreement's job, not arithmetic's.

## What else the recording measures

Every frame already describes the conditions it was collected under, without a single extra
measurement. `analysis/observers.py` reads them out:

| from | |
|---|---|
| the eye patches | illumination and contrast of the eye region, and the left/right asymmetry that reveals side lighting |
| the eye boxes | head position in the camera frame, head roll, and interocular distance — which stands in for viewing distance, since the eyes subtend fewer pixels the further away they are |
| the landmarks | nose position and head geometry |
| the aspect ratios | eye openness and blinks |
| the model's own weight | its per-frame opinion of the image |

On one session that is a sanity check; across participants it is the covariate set that explains
why one is twice as accurate as another, which is otherwise the most frustrating thing about a
webcam comparison. Each measure is correlated against per-frame error in the output, because a
covariate nobody relates to the outcome is a curiosity.

Two conventions the script states and gets right, both easy to invert: **"left" is the subject's
left** (WebGazer's naming), which in an unmirrored frame sits at *higher* image x — take head
roll the other way round and it lands near 180 degrees with a meaningless standard deviation.
And **blinks are a relative drop**, tested against that observer's own median, because an
absolute aspect-ratio cutoff that finds blinks in one face finds none in the next.

## The one approximation

WebGazer's eye boxes are computed here from saccade.js's MediaPipe face mesh
(`face_landmarker.task`) rather than from the legacy `face_mesh` solution WebGazer runs. Same
landmark indices, same topology, slightly different weights — so a box can differ from the one
real WebGazer would draw, usually by about a pixel.

That substitution *is* the optimization: one face mesh per frame instead of two. Everything
downstream of the box is byte-identical to WebGazer's own code, and every frame's eye box is
recorded so the size of the remaining difference can be measured against a real WebGazer
session rather than assumed.

## Checks

Three pages, no build step. Serve the repo and open them.

| page | what it settles |
|---|---|
| `test/port-check.html` | runs `js/webgazer-features.js` and webgazer 3.5.3's own functions over the same pixels and demands identical bytes, including eye boxes that run off the edge of the frame |
| `test/pipeline-check.html` | loads the real ONNX encoder and runs a synthetic frame through both feature cuts and the packing, with no camera; reports the per-frame cost on this machine |
| `test/trial-check.html` | drives the capture trial on a virtual clock: probe timing, response scoring, and the capture window stretching to cover a late color change |
| `test/pursuit-check.html` | the moving-target trial on a virtual clock (`requestAnimationFrame` included): the velocity profile, the turnarounds, and that the logged path and the gaze frames share a clock |
| `test/deferred-check.html` | the no-WebGPU path with no camera: the real worker and wasm session, queueing every frame, the crop cap, the deliberately-missed deadline, and the write-back into a jsPsych dataset |

`port-check` is the one that matters most, and it has already earned its keep: it caught that
passing `willReadFrequently` to the destination canvas changes Chrome's downsampling filter
(shifting the feature vector by up to 23 gray levels), and that reusing the resize canvases
across frames — an obvious optimization — changes the pixels again for any eye box carrying
transparency. WebGazer allocates fresh canvases per eye per frame; so does this, and the test
is what keeps it that way.

`trial-check` runs on a virtual clock rather than the wall clock because Chrome throttles a
backgrounded tab's timers to about 1 Hz, which would make the results meaningless without
making them look wrong. The experiment records `page_hidden` per station for the same reason.

## Layout

```
index.html                      the experiment
js/config.js                    grids, durations, probe rate, where the data goes
js/capability.js                WebGPU probe, hardware facts, the ridge microbenchmark
js/dual-recorder.js             the shared per-frame loop: one face mesh, two feature cuts
js/deferred-embedder.js         the worker pool's queue, crop cap and write-back
js/embed-worker.js              the ONNX encoder on its own thread
js/webgazer-features.js         WebGazer's eye boxes, patches and equalization, ported
js/plugin-cookoff-capture.js    one station: land, settle, capture, color change, score
js/plugin-cookoff-pursuit.js    the moving target, with its painted path logged
js/encoding.js                  float16 / uint8 packing
js/experiment.js                the timeline
js/save.js                      download or DataPipe
analysis/replay.py              offline calibration and accuracy for both trackers
analysis/pursuit.py             what the timing measurement is worth, from the moving target
analysis/saccades.py            saccade latency, from the arrival at each station
analysis/redact.py              make the public copy: numbers in, pictures out
analysis/observers.py           illumination, head position, distance, eye state per session
docs/DATA.md                    the data dictionary
scripts/sync-saccade.sh         copy saccade.js builds + model into lib/
scripts/serve.py                localhost dev server with caching off
```

## Tuning it

Everything worth changing is in `js/config.js`: the number of blocks, the probe rate, the
settle and capture durations, the points (the calibration grid size, whether to include the
validation grid, and how many random test points), and the save destination.

The defaults are 25 grid points, 9 validation points and 9 random points: 43 stations and about
84 seconds per block. Three blocks is 129 stations and roughly two megabytes. More blocks buy
more calibrate-then-test pairs, and a longer view of drift, at a linear cost in both.
