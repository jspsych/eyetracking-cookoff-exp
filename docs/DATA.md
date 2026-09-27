# The data

One session is a JSON object:

```json
{ "format": "cookoff/1",
  "trials": [ ... ],        // the jsPsych dataset, one object per trial
  "interactions": [ ... ] } // focus, blur and fullscreen changes, with timestamps
```

Roughly 2.5 MB, most of it base64. The analysis loaders also accept a bare array of trials,
which is what sessions recorded before this format carry.

`interactions` is jsPsych's own log: `{event, trial, time}`, where `event` is one of `blur`,
`focus`, `fullscreenenter` or `fullscreenexit`, `trial` is the trial index it fell in, and `time`
is ms from the start of the experiment. It is saved because `jsPsych.data.get().json()` does not
include it and it is otherwise discarded. The per-trial flags below are the summary of the same
events; this log is where the timings are.

Every station — one fixation point — is one row with `trial_type: "cookoff-capture"`. Everything
below is on that row unless it says otherwise.

## Reading the binary fields

Several fields hold per-frame streams, packed as base64 over a flat little-endian typed array
and laid out row-major: frame 0's values, then frame 1's, and so on.

WebGazer's streams, the head landmarks and the eye aspect ratios always have exactly `n_frames`
rows, and row `k` of each describes the camera frame taken at `frame_t[k]`. **saccade.js's
streams may be shorter**; see *Frames with both trackers* below.

| field | type | stride | what one row is |
|---|---|---|---|
| `saccade_embedding` | float16 | `embedding_dim` (128) | saccade.js's eye embedding |
| `saccade_weight` | float16 | 1 | the model's own quality score for the frame, in [0, 1], or NaN |
| `saccade_crop` | uint8 | 5184 | a raw 144x36 eye crop, for frames the encoder never reached |
| `webgazer_gray_left` | uint8 | 60 | the left eye's 10x6 grayscale patch, row-major |
| `webgazer_gray_right` | uint8 | 60 | the right eye's patch |
| `webgazer_eye_box` | int16 | 8 | `[lx, ly, lw, lh, rx, ry, rw, rh]`, in camera pixels |
| `head_landmarks` | float16 | 27 | nine landmarks as `(x, y, z)`, normalized |
| `head_pose` | float16 | 16 | MediaPipe's 4x4 head-pose matrix, column-major; NaN when it gave none |
| `eye_aspect_ratio` | float16 | 2 | left then right; low means a closing eye |
| `eye_image_stats` | float16 | 3 | the eye strip's image quality: sharpness (SD of the Laplacian), mean luminance (0-255), and noise (Immerkaer's estimate of the sensor noise SD). NaN where there was no strip. See below |

In Python:

```python
import base64, numpy as np

def unpack(b64, dtype, stride):
    if b64 is None:
        return np.zeros((0, stride), dtype=dtype)
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, stride)

embeddings = unpack(row["saccade_embedding"], "f2", 128).astype(np.float32)  # (n_frames, 128)
left       = unpack(row["webgazer_gray_left"], "u1", 60)                     # (n_frames, 60)
```

The nine `head_landmarks` are MediaPipe indices `[1, 10, 152, 234, 454, 33, 133, 362, 263]` —
nose tip, forehead, chin, both face edges, and the four eye corners.

`head_pose` is the rigid transform from MediaPipe's canonical face model to the face in that
frame: rotation in the upper-left 3x3, translation in elements 12–14 in MediaPipe's model units
(about centimeters, under the camera field of view MediaPipe assumes, so treat the distance as
relative). `analysis/observers.py` turns it into yaw, pitch and roll with the same decomposition as
saccade.js's `headPoseFromMatrix`. Which direction counts as positive follows MediaPipe's axes;
check the sign against a deliberate head turn before interpreting it. Sessions recorded before
saccade.js core 0.3.0 have no `head_pose`, and `head_pose_recorded` is false.

`saccade_weight` is NaN for every frame when the loaded model has no second output for it. That
is a model-wide fact, not a per-frame one, so check the first value and treat the column as
absent rather than filtering frame by frame. The shipped `eye-embedding@1.0.0` does emit it.

## Frames with both trackers

On a machine with WebGPU the encoder runs inside the frame loop, every frame gets an embedding,
and `embedded_frames` is `null`, meaning "all of them".

Without WebGPU the encoder costs one to several hundred milliseconds on single-threaded wasm —
longer than a camera frame. It moves to a Web Worker so it cannot stall the recording, and it
gets a per-station budget, because backgrounding does not make it fast: at 200 ms an encode, a
station's thirty frames need six seconds and a station lasts about two. So an evenly spaced
subset is encoded and the rest of the frames have no embedding. WebGazer's features are cut for
every frame either way, since they cost almost nothing.

That leaves two streams of different lengths, and three fields to line them up:

- **`embedded_frames`** — indices into `frame_t`, one per row of `saccade_embedding` and
  `saccade_weight`. `null` means every frame.
- **`crop_frames`** and **`saccade_crop`** — frames the worker did not reach before the drain
  deadline. Their raw 144x36 crops are in the data instead, twenty times the bytes and exactly
  the same information; run them through the model offline. Usually `null`.
- **`station_id`** — the station's serial number, which is how the worker's results found this
  row after the trial had already finished.

**A head-to-head comparison has to run on the frames both trackers have.** Scoring saccade.js on
its eight frames and WebGazer on all thirty is not a comparison of trackers, it is a comparison
of sample sizes. `analysis/replay.py` intersects on `embedded_frames` for both fitting and
scoring, and `--encode-crops <model.onnx>` folds the leftover crops back in.

```python
embedded = row["embedded_frames"]
if embedded is None:
    embedded = list(range(row["n_frames"]))
matched_webgazer_features = features[embedded]   # same frames as the embeddings
```

The budget is recorded per session as `encoder_budget_per_station`, so a session collected at
eight frames a station is distinguishable from one collected at thirty.

## Rebuilding WebGazer's features

The stored patches are grayscale *before* histogram equalization, which is lossless and a third
the size. WebGazer equalizes each patch over its own 60 pixels and nothing else, so its feature
vector is a pure function of what is stored:

```python
def equalize_histogram(patch, step=5):
    """webgazer 3.5.3, util.mjs. Note the sampling: 12 of the 60 pixels build the histogram."""
    hist = np.bincount(patch[::step], minlength=256).astype(np.float64)
    hist = np.cumsum(hist) * (255.0 * step / len(patch))
    return hist[patch]

features = np.concatenate([equalize_histogram(left_row), equalize_histogram(right_row)])  # 120
```

`analysis/replay.py` has this, and `test/port-check.html` checks the browser-side half of it
against webgazer's own code on the same pixels.

## Per-station fields

**Where the target was.** `target_x`, `target_y` in viewport pixels; `target_x_frac`,
`target_y_frac` as fractions; `viewport_width`, `viewport_height`, `device_pixel_ratio`. Store
the fractions, predict in whichever you prefer — but note the two trackers' own conventions
differ: saccade.js fits to viewport fractions centered on 0.5, WebGazer fits to raw pixels.

**Which condition.** `block` (0-based; one block is one full pass over every point), `rep` (the
same number, kept separate because a design with several blocks per pass would need both),
`point_set` (`"grid"` for the calibration grid, `"validation"` for saccade.js's 9-point
validation grid, `"random"` for the block's fresh random positions), `grid_level` (for a grid
point, the smallest of saccade.js's nested grids it belongs to: `9` for the 3x3 at 5/50/95%,
`13` for the four points that make its default 13, `25` for the rest of the 5x5; `null` for a
test point. An n-point calibration is `grid_level <= n`), `practice` (true for the practice
block; exclude it). Within a block the grid comes first, then the test points; `index_in_block`
gives the order.

Sessions recorded before the 5x5 grid use `"train"` for the 13-point grid and `"test"` for the
validation grid, and have no `grid_level`. `replay.py` recovers the level from the position.

**Timing.** `capture_onset` is `performance.now()` when the window opened. `frame_t` is each
kept frame's camera timestamp in ms from that onset, *uncorrected*. `clock_source` says which
clock those came from: `captureTime` is the camera's own and is the one you want,
`receiveTime` is the renderer's, `callback` means the browser offered nothing and the numbers
describe JavaScript rather than the camera.

To put a frame on the same clock as the stimulus, subtract the session-level `timing_lag_ms`
(measured by `saccade-time-sync`, see below). Check `timing_verdict` first: only `"OK"` means
the estimate is well constrained.

**Frame accounting.** `n_frames` is what was kept — a face found, an eye strip cut, both patches
built. `n_face_missing`, `n_camera_dropped` (frames the browser says the camera dropped) and
`n_skipped` (a face, but no usable crop) are what was not. `fps` is the recorder's rate at the
end of the window.

**The attention probe.** `probe` (did this station color change), `probe_onset` (ms from
`capture_onset`), `response_time` (ms from `capture_onset`), `probe_rt` (ms from the color change),
`probe_outcome` (`hit` / `miss` / `false_alarm` / `correct_rejection`), `points`,
`running_score`, `streak_after`.

A `hit` is direct evidence the fovea was on the target: the color change is six pixels across and
cannot be caught peripherally. A `miss` is the one signal in the dataset that says a capture
window may not describe a fixation on the point. **Deciding whether to drop misses is a real
analysis choice — make it before you look at the accuracy numbers**, since it is exactly the
kind of filter that can be tuned to favor one tracker.

**`fullscreen_exited`.** True if the window left fullscreen during this trial. That resizes the
viewport, which moves every target relative to a calibration collected at the old size — so the
positions before and after are not on the same screen. `replay.py` drops these trials and counts
them, and warns if the kept trials still carry more than one viewport size (a plain window
resize does the same damage without firing a fullscreen event). `fullscreen_requested` on every
row says whether the session asked for fullscreen at all.

**`window_blurred`.** True if the window lost focus during the trial. This is the exclusion that
leaves no other trace: the page stays *visible*, so nothing throttles, the camera keeps
delivering and the frames look perfectly healthy — while the participant is looking at another
window. A second monitor, a notification, an alt-tab. `replay.py` drops these trials.

**`page_hidden`.** True if the tab was hidden at any moment during the station. A hidden tab has
its timers throttled to about 1 Hz and stops receiving camera frames. Always exclude these.

## Session-level properties

On every row: `participant_id`, `experiment`, `version` (also in the file name, as
`cookoff_v<version>_<participant>_<timestamp>.json`; the first pilot's files, version 1.0.0,
have no version in the name), `user_agent`, `screen_width`,
`screen_height`.

From the screening trials, before the camera or the model are touched:

| field | |
|---|---|
| `browser`, `browser_version`, `os` | from `@jspsych/plugin-browser-check` |
| `is_mobile_ua` | what the user-agent string claims. Recorded, never acted on — see `keyboard_present` |
| `has_webcam`, `vsync_rate` | a camera was enumerable; the display's refresh rate, which is part of the display lag the loopback measures |
| `window_width`, `window_height` | window size *before* fullscreen |
| `keyboard_present` | a space bar was actually pressed within the timeout. This is the gate: a participant who fails it is aborted, so in a completed session it is always true |
| `camera_type` | `built-in`, `separate` or `unsure`, self-reported. Camera position relative to the screen is the largest thing that varies between setups and cannot be recovered from the video |
| `fullscreen_requested` | whether the session asked for fullscreen |

`is_mobile_ua` and `keyboard_present` answer different questions and are both worth having. The
user-agent string can lie in both directions — a tablet with a keyboard attached is perfectly
usable, and a desktop browser in mobile emulation is not a phone — so the gate tests the thing
the task needs rather than the string.

From the end questionnaire, asked after the eye tracking so nothing in it moves the participant
between calibration and test. Every question can be declined; a question switched off in
`config.questions` records `null`.

| field | |
|---|---|
| `eyewear` | `glasses`, `contacts`, `neither` or `prefer_not` |
| `technical_problems` | the answer to "Did you encounter any technical problems?", or `null` if left blank |
| `interruptions` | the answer to "Did anything interrupt your ability to keep your eyes on the targets?", or `null` if left blank |

Both free-text answers are optional and can name people, places or devices;
`redact.py --strip technical_problems interruptions` removes them from the public copy.

Demographics are not asked. Prolific's demographic export carries age, sex and the rest, keyed on
the Prolific ID, which is `participant_id` here.

These answers are the only record of *why* a session went badly from the participant's side. The
covariates can show a head that drifted or lighting that changed; this is what says someone
walked in. Blank does not mean nothing went wrong. Decide how it will be used before looking at
the accuracy numbers, since it is easy to read selectively in favor of one tracker.

From the timing measurement: `timing_lag_ms`, `timing_verdict` (`OK` / `INCONCLUSIVE` /
`UNRELIABLE`), `timing_plateau_width_ms` (the uncertainty on the lag — one camera frame or less
is expected), `timing_camera_period_ms`.

About the encoder: `encoder_mode` (`inline`, `strided`, `deferred`, or `inline-fallback` when
the worker would not start), `encoder_inline_ms` and `encoder_worker_ms` (measured on this machine, which
is what the budget was computed from), `encoder_provider`, `encoder_budget_per_station`,
`frames_embedded`, `frames_left_as_crops`, and `embed_queue_timed_out`.

`strided` (from version 1.2.0) is WebGPU that cannot encode every frame: the encoder stays on the
GPU and embeds a frame whenever it is free, so a capture trial's `embedded_frames` lists the
frames that got an embedding, in order, with `saccade_embedding` and `saccade_weight` holding one
row each. Unlike `deferred`, the embeddings are there when the row is written, arrival frames are
encoded too (so saccade.js latency can be timed), and no frame is ever left as a raw crop.

The frames the GPU had no time for during a station are kept in memory and encoded in the
downtime: during the fixed break between blocks (`break_ms`, 10 s), during the end questions,
and on the "finishing up" screen before saving (up to `drain_ms_at_end`). Before saving, those embeddings are merged into each
station's row, so `embedded_frames` covers both, in frame order; `backfilled_frames` lists the
frames that were encoded afterwards (empty if none), and `n_backfilled` counts them. The frames
in `embedded_frames` but not in `backfilled_frames` are the ones live tracking had. Rows from
inline and deferred sessions carry `backfilled_frames: null`. On a session row, `frames_embedded` counts the
backfilled frames and `frames_dropped_unencoded` the ones the deadline left without an embedding
(their crops are discarded, never saved). What live tracking on that machine would have produced
is the live part: `embedded_frames` minus `backfilled_frames`.

`encoder_forced` is the path a pilot forced from the URL (`?encoder=strided`, `inline` or
`deferred`), and null when the machine chose its own. Never set for a Prolific participant.

What the mode was decided from: `camera_fps`, the camera's delivered frame rate measured on the
video element itself before anything was timed (null if the browser could not report it);
`encoder_inline_ms`, now the *median* of `encoder_inline_samples` inferences; and
`encoder_inline_limit_ms`, the limit it was compared against. On WebGPU the limit is 85% of one
camera frame, between 30 and 80 ms; on wasm, or with no `camera_fps`, it is 30 ms. Sessions
recorded before these fields existed used a flat 30 ms and the mean of five inferences.

What the camera was doing: `camera_settings`, what the browser reports the camera's track is
set to, and `camera_capabilities`, the ranges and modes it offers. Both are allow-listed and
never include the device ID: resolution, `frameRate` (the configured rate; `camera_fps` is the
delivered one), exposure, focus and white-balance modes and values, image controls, and
`backgroundBlur` (the operating system's own camera effect, which softens the image before
either tracker sees it). Chrome reports exposure and focus for many cameras, other browsers
rarely; keys the browser does not report are absent.

**Image quality, per frame (`eye_image_stats`).** Three numbers from the 144x36 eye strip
saccade.js encodes, the same region WebGazer's patches come from. *Sharpness* falls when the
image is blurred, by focus or by the eye moving during a long exposure. It also rises with sensor
noise and scales with contrast, so read it next to the other two: a dim, noisy frame can score
as "sharp" as a crisp one. *Noise* is an estimate of the sensor noise's standard deviation, in
grey levels, which is what separates the two cases.

From the warm-up: `embedding_dim`, `embedding_model` (a string like `eye-embedding@1.0.0`, which
means the loaded bytes hash-matched that published release; anything else appears as
`sha256:...`), `execution_provider` (`webgpu` or `wasm`), `landmarker_delegate` (`GPU` or `CPU`),
`video_width`, `video_height`, `warmup_fps`, and the recorder's totals for the whole session
(`recorder_frames`, `recorder_face_missing`, `recorder_camera_dropped`, `recorder_errors`).

`warmup_fps` below about 20 means this recorder could not keep up and the capture windows will
be thin. In sessions recorded before `camera_fps` existed, a `warmup_fps` of 0 (or a
`warmup_frames` of 1 or 2) is not a slow camera: MediaPipe's one-time setup ran on the warm-up's
first frame and blocked the page for the whole window. Use the capture trials' `frame_t`
spacing for those sessions' real rate. The setup now runs while the recorder loads.

Decide the exclusion threshold before collecting. For what *live* tracking would have
managed, use `saccade_live_fps_median` instead — see the next section.

## Could this machine have tracked live?

The session never predicts gaze, so nothing in the capture trials answers the question you will
be asked about any webcam study: could saccade.js have tracked this participant in real time,
and how fast? Three groups of fields, recorded on every row.

**Whether the hardware was there.** `webgpu_available` (does `navigator.gpu` exist),
`webgpu_adapter` (did `requestAdapter()` return one), `webgpu_fallback_adapter` (true means a
*software* adapter — WebGPU by the letter, wasm-like speeds in practice, and worth excluding
from any "had WebGPU" grouping), `webgpu_vendor`, `webgpu_architecture`, `webgpu_description`,
`webgpu_error`. Plus `hardware_concurrency` and `device_memory_gb` as covariates.

These describe the *browser's* capability and are deliberately separate from what ONNX Runtime
actually did with it: ORT tries WebGPU with graph capture, then WebGPU without, then wasm, and
takes the first that survives. `encoder_provider` and `saccade_live_backend` say which it
landed on. A machine with `webgpu_available: true` and `encoder_provider: "wasm"` is a real and
interesting case — a GPU the runtime could not use — and only shows up if both are recorded.

**What it actually ran at.** From the `saccade-performance` trial, which runs saccade.js's real
tracker on this camera for five seconds after a one-second warm-up:

| field | |
|---|---|
| `saccade_live_fps_median` | **the headline number**: the rate live tracking achieved |
| `saccade_live_fps_mean`, `saccade_live_fps_p10` | the mean, and the slow tail — a machine that stalls periodically has a healthy median and a poor p10 |
| `saccade_live_embed_ms_median` | time in the model per frame; separates a slow model from a camera that is not delivering frames |
| `saccade_live_frames`, `saccade_live_frames_without_face`, `saccade_live_dropped` | what the measurement was computed from |
| `saccade_live_backend` | `webgpu` or `wasm` |

That trial's own row (`trial_type: "saccade-performance"`) carries the same fields under their
plugin names.

**Do not substitute `warmup_fps` for this.** `warmup_fps` is *this recorder's* rate, and the
recorder does strictly more work per frame than live tracking — it also cuts WebGazer's patches
— and on a slow machine it moves the encoder to a worker, which live tracking cannot do. The
gap between `saccade_live_fps_median` and `warmup_fps` is the price of carrying the second
tracker, which is a real quantity, just not the one the question asks for.

**Where the time goes.** `ridge_predict_us` is the measured cost of saccade.js's per-frame ridge
prediction — the dot product of the embedding with the `2 x d` kernel that turns an embedding
into a gaze point. It lands around **1 µs**, against tens of milliseconds for the encoder on
WebGPU and hundreds on wasm. The frame rate is the encoder and the face mesh; the regression is
four decimal places below either. Recorded rather than asserted, so the claim can be checked per
machine instead of taken on faith. `encoder_inline_ms` and `encoder_worker_ms` are the other
side of that comparison.

## The arrival: saccade latency

With `design.record_settle` on (the default), recording opens when the target *appears*
rather than after it has settled, so a station contains the whole arrival — the eye still on the
previous target, the saccade across, the fixation that follows. Three fields make that usable:

| field | |
|---|---|
| `settle_duration` | the settle length in ms, or `null` when the settle was not recorded |
| `fixation_from` | index of the first frame at or after the settle ended. **Frames before it are the eye on its way here, not steady fixation** |
| `previous_target_x/y`, `previous_station_id` | where the saccade started. `null` for the first station after any break screen, because after one of those the eye was on a button |

**An accuracy analysis must drop frames before `fixation_from`.** Letting them into the fit
calibrates partly on where the participant used to be looking. `replay.py` does this, and
reports the count as `skipped["arrival_frames"]`.

`frame_t` stays relative to `capture_onset`, which with `record_settle` is the moment the
target appeared. `probe_onset` and `response_time` are on that same clock.

The target appears at **full opacity on the first frame**, and only its scale springs. That
is deliberate: saccade latency to a stimulus that fades in over 180 ms is longer than to one
that simply appears, by an amount nothing downstream can recover.

`analysis/saccades.py` measures it. The method is a position crossing, not a velocity threshold
— at 24 fps a 40 ms saccade usually falls *between* two samples, so any velocity criterion
measures the sampling grid. Gaze is projected onto the line between the two targets and the 0.5
crossing is interpolated between the samples either side. On a synthetic session with a known
answer that recovers the crossing to 1 ms; snapping to the nearest sample instead biased it
by +39 ms.

What it returns is the saccade's **midpoint**; onset is roughly 15-25 ms earlier. And the raw
figure includes `timing_lag_ms`, because a sample stamped at T shows the eye as it was that much
earlier — so the corrected number is the one to compare against the ~200 ms in the literature.

## The moving-target trials

`trial_type: "cookoff-pursuit"` rows are the moving-target block, and they are the only trials
where `timing_lag_ms` changes an answer. On a stationary target a stale gaze timestamp costs
nothing; on a target moving at `v` it costs `v x lag` pixels of pure bookkeeping error.

They carry every field a capture station carries, plus:

| field | |
|---|---|
| `target_path` | where the target was, per animation frame: base64 little-endian **int16**, stride 3, `[t_ms, x_px, y_px]`, `t_ms` relative to `capture_onset`. Whole pixels and whole milliseconds. |
| `path_frames`, `raf_period_ms` | animation frames painted and the mean interval (16.7 ms on a 60 Hz display) |
| `speed`, `cycles`, `condition` | *mean* tangential speed in px/s, circuits of the ellipse, and the speed label |
| `semi_axis_x/y`, `center_x/y`, `direction` | the ellipse itself, and which way round it went |

`t_ms` is the `requestAnimationFrame` timestamp — when the browser began painting that position.
That is the same clock a stimulus onset is on, and the clock `timing_lag_ms` corrects gaze
timestamps *to*. So the correction is: `gaze_time - timing_lag_ms`, then interpolate
`target_path` at that time.

The path is an ellipse matched to the viewport aspect, entered by easing the *phase* rate rather
than the amplitude — so the shape stays exactly elliptical while the target accelerates into it.
Instantaneous speed therefore varies around each lap, fastest across the top and bottom.

**The eased ends are not flagged**, deliberately: differentiate the path and keep the frames
above half the run's median speed. That keeps the analysis correct if the motion changes shape
again, and `analysis/pursuit.py` does exactly this.

A closed path is what lets the three error types come apart — tangential for timing, radial for
gain, and a screen-fixed offset that averages out of both. Sessions recorded before this used
straight sweeps; `pursuit.py` detects a path that encloses no area and prints `n/a` for the
radial term rather than a meaningless number.

Direction alternates across runs. A real lag always trails the motion and therefore flips sign
with it; a constant spatial bias does not.

```python
raw = np.frombuffer(base64.b64decode(row["target_path"]), dtype="<i2").reshape(-1, 3)
t_ms, x, y = raw[:, 0], raw[:, 1], raw[:, 2]
corrected = np.asarray(row["frame_t"]) - session["timing_lag_ms"]
target_x = np.interp(corrected, t_ms, x, left=np.nan, right=np.nan)
```

**Three speeds, not one.** A single speed cannot separate a timing lag from a constant spatial
offset, because at one velocity both produce the same displacement. Across speeds the lag's
contribution grows linearly and the offset's does not, so the slope of error-against-speed
identifies the lag — and that slope is an estimate of the same quantity the loopback measured
optically, from completely independent evidence.

## What is identifiable in here

Two image streams, and they are not comparable.

**WebGazer's patches are not a biometric.** `webgazer_gray_left` / `_right` are 10x6 grayscale —
sixty pixels an eye, with the iris spanning about three of them. Iris recognition wants an iris
diameter upwards of 120 px and periocular recognition wants tens of pixels of skin texture;
neither survives that downsample. Upscaled, a patch is a dark blob beside a bright blob. It
carries where the eye was pointing, which is its job, and not who it belongs to.

What they *can* do, weakly, is link two sessions of the same person: the per-pixel statistics
carry skin tone and lighting. That is a re-identification risk, not a reconstruction one, and it
is worth stating in a data-sharing plan rather than ignoring.

**`saccade_crop` is the real exposure**, and it is off by default. It is the raw 144x36 strip
across *both* eyes — roughly 50 x 36 px each, which is enough to be a recognizable picture. It
is only ever written when the encoder was deferred to a worker and the worker missed the drain
deadline, so it is empty on any machine with WebGPU. `privacy.save_eye_crops` in `js/config.js`
gates it; with it off those frames are dropped and counted in `frames_dropped_unencoded`, and
`eye_crops_saved` on every row records which way the session ran.

**`head_landmarks`** is nine facial points per frame. Sparse, but inter-landmark distances carry
some face geometry, so treat it as a linkage risk rather than an anonymous covariate.

### On obfuscating the patch pixels

WebGazer's ridge regression is a plain dot product over the 120 features, so it does not care
what order they are in: permuting the features and permuting the coefficients the same way gives
identical predictions (verified — the largest change in any prediction was 0.1 px, which is
solver conditioning). A per-session random permutation is therefore free, analytically.

Three things to know before relying on it:

1. **It has to be applied after equalization, not before.** `equalizeHistogram` builds its
   histogram from every 5th pixel, so which pixels land on those indices changes the output.
   Permuting first and equalizing offline changed features by up to 106 gray levels.
2. **It costs nothing to store.** An equalized feature takes one of only 13 distinct values
   (spacing 21.25), so the equalized-and-permuted vector fits in the same 120 bytes the raw
   patch uses today.
3. **It is only partly effective.** Adjacent pixels correlate at 0.51, and that structure
   survives shuffling: recovering a layout from the correlation matrix of 2950 shuffled frames
   got horizontal position back at r = 0.68 (vertical, at 0.05, did not). A random *orthogonal
   rotation* mixes every pixel into every feature and preserves ridge exactly — the L2 penalty
   is rotation-invariant — which is strictly stronger for the same analytic cost.

None of these reduce the information in the data; they are invertible linear maps that raise the
cost of casual misuse. Given the patches are already below any recognition threshold, the
honest ordering is: drop the crops, keep the patches as they are, and put access control rather
than arithmetic in front of anything richer.

## The one approximation

WebGazer's eye boxes here are computed from saccade.js's MediaPipe face mesh
(`face_landmarker.task`, via @mediapipe/tasks-vision), not from the legacy `face_mesh` solution
WebGazer runs through @tensorflow-models/face-landmarks-detection. Same landmark indices, same
topology, different weights — so a box can differ from the one real WebGazer would have drawn,
usually by about a pixel.

This is the whole optimization: one face mesh per frame instead of two. Everything downstream of
the box — the crop, the resize, the grayscale, the equalization — is byte-identical to
WebGazer's own code, and `test/port-check.html` is what keeps it that way.

`webgazer_eye_box` is recorded per frame so the size of the remaining difference can be
*measured* rather than assumed: run a short session with real WebGazer alongside, and compare
the boxes it draws against the ones recorded here.
