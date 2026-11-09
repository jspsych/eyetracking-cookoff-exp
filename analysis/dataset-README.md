# saccade.js validation dataset

Webcam recordings from an online study comparing two browser-based eye trackers,
[saccade.js](https://github.com/jspsych/saccadejs) and [WebGazer](https://webgazer.cs.brown.edu/), on
identical camera frames. 203 participants recruited through Prolific each looked at 129 targets
on their own screen while their webcam recorded. Neither tracker predicted gaze during the
session. Instead, the session saved what each tracker needs as input, frame by frame, so both
can be calibrated and tested offline on the same data.

No video is included. Each frame is stored as numbers: saccade.js's eye embedding, WebGazer's
low-resolution eye patches, head position, and image-quality measures.

## Contents

| Path | What it is |
|---|---|
| `sessions/S###.json.gz` | One gzipped JSON file per participant (203 files) |
| `participants.csv` | One row per participant: session-level facts and four demographics |
| `README.md` | This file |

Each session is labeled with a code such as `S104`. The codes are arbitrary labels, not
consecutive numbers.

## The task

Each participant completed three blocks of 43 targets, after a short practice block.

- **Targets.** Each block shows a 25-point calibration grid (a 5×5 grid at 5, 27.5, 50, 72.5
  and 95% of the screen width and height), then 18 test points: a 9-point grid at 15, 50 and 85%,
  and 9 random positions. Positions are fractions of the browser window, which was fullscreen.
- **Timing.** Each target appears abruptly and stays for 1600 ms: a 600 ms settle period, then a
  1000 ms fixation window. Both are recorded, so the saccade to each target is in the data.
- **Attention check.** On 13 targets per block the center dot turns orange 300–700 ms into the
  fixation window, and the participant presses the space bar. A press within 1000 ms counts as a
  hit. On those targets, recording continues until the 1000 ms response window closes.

## Session files

Each file is a JSON object:

```json
{ "format": "cookoff/1",
  "trials": [ ... ],         // one object per jsPsych trial
  "interactions": [ ... ] }  // focus, blur and fullscreen events, with timestamps
```

`trials` is the jsPsych dataset. Each target is one row with `trial_type: "cookoff-capture"`;
the other rows are the setup screens, checks and questionnaire. Session-level fields (described
below) are repeated on every row.

`interactions` is jsPsych's event log: `{event, trial, time}`, where `event` is `blur`, `focus`,
`fullscreenenter` or `fullscreenexit` and `time` is ms from the start of the session.

### Per-frame streams

Several fields hold one row per camera frame, packed as base64 over a little-endian typed array,
row after row.

| Field | Type | Values per frame | What one row is |
|---|---|---|---|
| `saccade_embedding` | float16 | 128 | saccade.js's eye embedding |
| `saccade_weight` | float16 | 1 | the model's own 0–1 estimate of how usable the frame is |
| `webgazer_gray_left` | uint8 | 60 | the left eye's 10×6 grayscale patch, before histogram equalization |
| `webgazer_gray_right` | uint8 | 60 | the right eye's patch |
| `webgazer_eye_box` | int16 | 8 | `[lx, ly, lw, lh, rx, ry, rw, rh]`, the patches' source boxes in camera pixels |
| `head_landmarks` | float16 | 27 | nine facial landmarks as `(x, y, z)` |
| `head_pose` | float16 | 16 | MediaPipe's 4×4 head-pose matrix, column-major; NaN when none was given |
| `eye_aspect_ratio` | float16 | 2 | left then right eye; low values mean a closing eye |
| `eye_image_stats` | float16 | 3 | image quality of the eye region: sharpness (SD of the Laplacian), mean brightness (0–255), sensor noise (Immerkær's estimate, in gray levels) |

```python
import base64, gzip, json
import numpy as np

def unpack(b64, dtype, width):
    if b64 is None:
        return np.zeros((0, width), dtype=dtype)
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, width)

session = json.load(gzip.open("sessions/S104.json.gz"))
row = next(r for r in session["trials"] if r["trial_type"] == "cookoff-capture")
embeddings = unpack(row["saccade_embedding"], "f2", 128).astype(np.float32)
left_eye = unpack(row["webgazer_gray_left"], "u1", 60)
```

**Frame alignment.** WebGazer's streams, the landmarks, head pose, eye aspect ratios and image
statistics always have `n_frames` rows. saccade.js's streams can have fewer: `embedded_frames`
lists the frame index (into `frame_t`) of each embedding row, and `null` means every frame. A
comparison between the trackers should use the frames both have.

On machines without WebGPU, the eye model ran in background CPU workers during the session, and
frames the workers did not reach were saved as raw eye images. Those images were run through the
same model after the session and then deleted. In the 133 sessions where this happened,
`offline_frames` lists the frames encoded this way; the field is absent from the other sessions.
`frames_dropped_unencoded` counts frames that were never saved at all; they have WebGazer data
but no embedding (0.9% of frames in the main blocks).

**Landmarks.** The nine `head_landmarks` are MediaPipe face-mesh indices
`[1, 10, 152, 234, 454, 33, 133, 362, 263]`: the nose tip, the top and bottom of the face
outline, its left and right edges at cheekbone height, and the outer and inner corners of each
eye. x and y are fractions of the camera image; z is MediaPipe's relative depth.

**Head pose.** Rotation is in the upper-left 3×3 of `head_pose`, and translation in elements
12–14 in MediaPipe's model units (roughly centimeters, but treat distance as relative).

### Per-target fields

- **Target position:** `target_x`, `target_y` (pixels); `target_x_frac`, `target_y_frac`
  (fractions of the window); `viewport_width`, `viewport_height`, `device_pixel_ratio`.
- **Condition:** `block` (0–2), `point_set` (`grid`, `validation` or `random`), `grid_level`
  (for grid points, the smallest calibration grid the point belongs to: 9 for the 3×3 at 5, 50 and
  95%, 13 for the four points that complete saccade.js's default 13-point grid, 25 for the rest;
  `null` for test points), `practice` (true for practice rows, which should be excluded), and
  `practice_round`.
- **Timing:** `capture_onset` is when recording opened, at target onset. `frame_t` is each
  frame's camera timestamp in ms from that onset. `clock_source` says which clock it came from
  (`captureTime`, the camera's own, in nearly every row). `settle_duration` is 600 ms, and
  `fixation_from` is the index of the first frame after the settle.
- **The saccade:** `previous_target_x`, `previous_target_y` and `previous_station_id` give the
  target the eye started from, or `null` for the first target after a break screen.
- **Frame counts:** `n_frames` (kept), `n_face_missing`, `n_camera_dropped`, `n_skipped` (a face,
  but no usable eye region), and `fps`.
- **Attention check:** `probe` (did the dot change color), `probe_onset` and `response_time` (ms
  from `capture_onset`), `probe_rt` (ms from the color change), and `probe_outcome` (`hit`,
  `miss`, `false_alarm` or `correct_rejection`).
- **Distractions:** `page_hidden` (the tab was hidden), `window_blurred` (the window lost focus)
  and `fullscreen_exited` (the window left fullscreen) during the target.

### Session-level fields

Repeated on every row.

- **Setup and screening:** `browser`, `browser_version`, `os`, `user_agent`, `is_mobile_ua`,
  `has_webcam`, `vsync_rate`, `screen_width`, `screen_height`, `window_width` and `window_height`
  (before fullscreen), `keyboard_present`, and `camera_type` (self-reported: `built-in`,
  `separate` or `unsure`).
- **Hardware:** `hardware_concurrency`, `device_memory_gb`, `webgl_version`, `webgl_renderer`,
  `webgpu_available`, `webgpu_adapter`, `webgpu_fallback_adapter` (a software adapter),
  `webgpu_vendor`, `webgpu_architecture`, `webgpu_description`.
- **Camera:** `video_width`, `video_height`, `camera_fps` (measured), `camera_settings` and
  `camera_capabilities` (as reported by the browser; no device IDs).
- **Live-tracking performance:** saccade.js's own tracker was run for 5 s on each machine.
  `saccade_live_fps_median`, `saccade_live_fps_mean` and `saccade_live_fps_p10` are the frame
  rates it achieved, with `saccade_live_embed_ms_median`, `saccade_live_backend` and frame counts.
  `ridge_predict_us` is the measured cost of saccade.js's per-frame regression step.
- **Encoder:** `encoder_mode` is `inline` (the eye model ran on the GPU for every frame) or
  `deferred` (it ran in background CPU workers). The decision used `camera_fps`,
  `encoder_inline_ms` (the median time per frame, over `encoder_inline_samples` runs) and
  `encoder_inline_limit_ms`. Also recorded: `encoder_provider`, `encoder_workers`,
  `encoder_worker_ms`, `frames_embedded`, `embedding_model`, `embedding_dim`.
- **Camera timing:** a 15 s screen-flash measurement estimated the delay between the screen and
  the camera. `timing_lag_ms` is that delay; subtract it from `frame_t` to put frames on the same
  clock as the stimuli. `timing_verdict` is `OK` or `UNRELIABLE`, and `timing_plateau_width_ms`
  is its uncertainty.
- **Recorder totals:** `warmup_fps`, `recorder_frames`, `recorder_face_missing`,
  `recorder_camera_dropped`, `recorder_errors`.
- **End questionnaire:** `eyewear` (`glasses`, `contacts`, `neither` or `prefer_not`), and two
  optional free-text answers, `technical_problems` and `interruptions` (`null` if left blank).
- **Software:** `version` is the build of the experiment that recorded the session. The dataset
  contains two builds that record identical data; the later one only changed how a failed upload
  is retried. `saccade_build_version` and `saccade_build_commit` identify the saccade.js build.

Some fields describe features the experiment supports but these sessions never used, and can be
ignored: `encoder_forced`, `encoder_budget_per_station`, `backfilled_frames`, `pilot`,
`eye_crops_saved`, `frames_left_as_crops`. `eye_crop` holds the
eye region's dimensions (`[144, 36]`), not an image.

## participants.csv

One row per session.

| Column | Description |
|---|---|
| `session` | Session code, matching the file name |
| `version` | Experiment build (see above) |
| `n_stations`, `n_probes` | Targets (129) and color changes (39) in the main blocks |
| `probe_hit_rate` | Share of color changes caught |
| `encoder_mode`, `execution_provider` | How the eye model ran |
| `camera_type`, `eyewear` | Self-reported |
| `camera_fps`, `saccade_live_fps_median` | Measured camera and live-tracking frame rates |
| `timing_lag_ms`, `timing_verdict` | Camera timing measurement |
| `browser`, `os`, `screen_width`, `screen_height` | Machine |
| `age`, `sex`, `ethnicity`, `country_of_residence` | From the participant's Prolific profile; `ethnicity` is Prolific's simplified category |

## Notes for analysis

- **Practice rows** (`practice: true`) are not part of the design and should be dropped.
- **Settle frames:** for accuracy, drop frames before `fixation_from`; they show the eye moving
  to the target.
- **Attention:** in our analyses, a session was excluded if the participant caught fewer than 85%
  of the color changes (`probe_hit_rate < 0.85`, 3 sessions).
- **Distractions:** we dropped targets with `page_hidden`, `window_blurred` or
  `fullscreen_exited`.
- **WebGazer's features** can be rebuilt exactly from the stored patches. WebGazer equalizes
  each 60-pixel patch with its own histogram, sampling every fifth pixel:

  ```python
  def equalize_histogram(patch, step=5):
      hist = np.bincount(patch[::step], minlength=256).astype(np.float64)
      hist = np.cumsum(hist) * (255.0 * step / len(patch))
      return hist[patch]

  features = np.concatenate([equalize_histogram(left), equalize_histogram(right)])  # 120 values
  ```

  WebGazer's eye boxes here come from the same MediaPipe face mesh saccade.js uses, rather than
  the older one WebGazer ships with. Everything after the box is identical to WebGazer's code.

## Privacy

The dataset contains no video and no eye images other than WebGazer's 10×6 patches. Raw eye
images captured on slower machines were converted to embeddings and deleted. Prolific IDs were
replaced with session codes, and demographics are limited to the four columns in
`participants.csv`.
