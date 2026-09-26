/**
 * Everything worth changing without reading the rest of the code.
 */
window.CookoffConfig = {
  /**
   * The experiment's version: on every row as `version`, and in the file name
   * (`cookoff_v1.1.0_<participant>_<timestamp>.json`), so sessions from different versions sort
   * apart without opening them. Bump it whenever a change alters what gets recorded or how.
   *
   * 1.0.0  the first Prolific pilot (file names without a version).
   * 1.1.0  MediaPipe set up before the warm-up, so `warmup_fps` no longer reads 0 on slow
   *        machines; the camera's own frame rate recorded as `camera_fps`; the inline-encoder
   *        limit a share of one camera frame, not a flat 30 ms; per-frame image quality of the
   *        eye strip (`eye_image_stats`) and the camera's settings (`camera_settings`).
   * 1.2.0  "strided" encoder mode: on WebGPU too slow for every frame, the encoder stays on
   *        the GPU and embeds whichever frames it has time for, instead of moving to the CPU
   *        worker. The worker is now only for machines without WebGPU.
   */
  version: "1.2.0",

  /**
   * Where the heavy assets live. The ONNX encoder is served from this repo because saccade.js
   * is not on npm yet — `scripts/sync-saccade.sh` puts it there. MediaPipe and ONNX Runtime
   * come from saccade.js's own pinned CDN defaults; set `mediapipeWasmUrl`, `faceLandmarkerUrl`
   * and `ortWasmUrl` here to self-host those too.
   */
  assets: {
    modelUrl: "lib/saccade/eye_embedding.onnx",
  },

  /** Camera resolution to ask for. Both trackers were built around 640x480. */
  video: { width: 640, height: 480 },

  /**
   * Run the session in fullscreen.
   *
   * Not cosmetic. Every target position is a fraction of `window.innerWidth`, and both trackers
   * are fitted against those positions — saccade.js in viewport fractions, WebGazer in raw
   * pixels. A viewport that changes size mid-session therefore moves the screen out from under
   * a calibration collected before it, and neither fit survives that. Fullscreen fixes the
   * viewport for the duration and takes the browser chrome out of the picture.
   *
   * It has to be entered before anything measures the viewport or the participant positions
   * themselves, which is why the trial sits immediately after the welcome screen and before the
   * camera preview.
   *
   * Turn it off for piloting, where being trapped in fullscreen while debugging is worse than
   * the viewport moving.
   */
  fullscreen: true,

  /** Run saccade.js's live frame-rate check. Off for short diagnostic builds. */
  performance_check: true,

  /** What the welcome screen tells the participant to expect. */
  intro_minutes: 10,

  /**
   * Screening, before the 20 MB model downloads.
   *
   * Two different jobs. `browser_check` is a *record*: what browser, OS, screen and refresh
   * rate each participant had, so a result can be read against the hardware it came from. It
   * excludes nobody.
   *
   * The keyboard gate is a *gate*, and it is behavioral rather than a user-agent guess. The
   * task needs a key press to report the color change, so a participant who cannot press one
   * cannot produce the attention data — and on a tablet or phone there is no keyboard, no
   * stable viewing distance and usually no usable front camera geometry. Asking for a key press
   * tests the thing that actually matters instead of parsing a string that can lie, and it
   * costs nothing for anyone who can do the task.
   */
  screening: {
    browser_check: true,
    keyboard_gate: true,
    /** How long to wait for the key before concluding there is no keyboard, in ms. */
    keyboard_timeout: 20000,
    /** Ask which camera they are using. Position relative to the screen is the main thing
     *  that varies between setups and it cannot be measured from the video. */
    camera_question: true,
  },

  /**
   * What to do about the eye encoder on a machine with no WebGPU.
   *
   * On WebGPU an encode costs about 20 ms and runs inside the frame loop. On single-threaded
   * wasm it costs one to several hundred milliseconds — longer than a camera frame — so it
   * moves to a worker and stops blocking the recording. Backgrounding alone is not enough
   * though: at 200 ms an encode, a station's thirty frames need six seconds and a station
   * lasts about two. So the worker also gets a budget, and encodes an evenly spaced subset.
   *
   * WebGazer's features are still cut for every frame, so the two trackers' streams have
   * different lengths on these machines. `embedded_frames` records which frames have both, and
   * a head-to-head comparison has to run on those.
   */
  encoder: {
    /** Force a path, for testing: "auto", "inline", "strided" or "deferred". "strided" on a
     *  machine without WebGPU would run the CPU encoder on the page itself; test it on WebGPU. */
    mode: "auto",
    /**
     * On WebGPU, the encoder runs inline if its measured cost fits in this share of one camera
     * frame, with the frame rate measured on the camera itself: about 57 ms on a 15 fps webcam,
     * 28 ms at 30 fps. See `inlineLimitMs` in experiment.js for why the share is this high.
     */
    inline_frame_share: 0.85,
    /** Never inline above this, however slow the camera: a lower frame rate is often a
     *  measurement that came up short, not a camera. */
    inline_ceiling_ms: 80,
    /** The limit on wasm, where the encode runs on the main thread, and whenever the camera's
     *  frame rate could not be measured. Also the floor of the WebGPU limit. */
    inline_max_ms: 30,
    /** Timing the inline encode: the median of at least `min_samples` and at most `max_samples`
     *  inferences, stopping after `max_sampling_ms` once the minimum is met. */
    min_samples: 5,
    max_samples: 15,
    max_sampling_ms: 1500,
    /** Fraction of a station's wall-clock time the worker is assumed to get. Below 1 because
     *  the machine is also running the camera, MediaPipe and the page. */
    worker_share: 0.85,
    /** Never queue fewer than this per station, however slow the machine is: below about four
     *  frames a station there is not enough left to average an embedding over. Frames the
     *  worker cannot reach are written out as raw crops and encoded offline. */
    min_frames_per_station: 4,
    /** How long the participant may be held on a progress bar while the queue drains, in ms.
     *  Split between the breaks between blocks and the end of the session. */
    drain_ms_between_blocks: 20000,
    drain_ms_at_end: 45000,
  },

  design: {
    /**
     * Each block is one pass over the calibration grid and a test set; see `points`. Blocks
     * are what the analysis calibrates and tests across: calibrate on block 0 and test on the
     * later blocks (forward-only, as a live study would), or refit on a later block to measure
     * what recalibrating buys. Drift -- the participant settling, slumping, drifting out of
     * frame -- is therefore part of what is measured.
     */
    n_blocks: 3,
    /** Fraction of targets whose center changes color. */
    probe_rate: 0.3,
    /** Targets in the practice block, which is not analyzed. */
    n_practice: 5,
    settle_duration: 600,
    /**
     * Whether the settle phase is recorded as well as the capture window.
     *
     * Off, the recorder opens only after the target has stopped moving and the eye has
     * arrived, so every frame is steady fixation — which is what the accuracy comparison
     * wants, and which throws away the one moment in a station that carries a saccade.
     *
     * On, recording starts when the target appears. The trial then contains the whole
     * arrival: the eye still on the *previous* target, the saccade across, and the fixation.
     * That is a saccade-latency measurement per station, about sixty per session, at no cost
     * in task time. It costs roughly 60% more data, because a station records 1600 ms instead
     * of 1000.
     *
     * `frame_t` stays relative to the moment recording opened either way; `settle_duration`
     * and `fixation_from` on each row say where the fixation window starts, and an analysis
     * that only wants steady fixation drops everything before it.
     */
    record_settle: true,
    capture_duration: 1000,
    probe_window: [300, 700],
    response_deadline: 1000,
    response_key: " ",
    /**
     * Blank time between one target leaving and the next appearing, in ms.
     *
     * Zero, so each station is a plain step: the new target appears while the eye is still on
     * the old one. A blank before the target shortens saccade latencies (the gap effect), which
     * would make the latencies `saccades.py` measures incomparable with step-task norms.
     */
    station_gap: 0,
  },

  /**
   * The moving-target block, which is the only part of the session where the timing
   * measurement changes an answer.
   *
   * A stale gaze timestamp costs nothing on a stationary target and costs `speed x lag` pixels
   * on a moving one, so speed is the manipulation.
   *
   * The path is an ellipse rather than a straight sweep, because on a closed 2-D path the three
   * things that make an eye trail its target stop being the same measurement: a timing lag
   * leaves a *tangential* error, a pursuit gain deficit leaves a *radial* one, and a constant
   * miscalibration is fixed in screen coordinates and averages out around a cycle. A straight
   * line confounds all three.
   *
   * Cycle counts roughly equalize time on task across the speeds rather than the number of
   * circuits, so the slow condition gets one lap and the fast ones two.
   */
  pursuit: {
    /** `speed` is the *mean* tangential speed; instantaneous speed varies around it, fastest
     *  across the top and bottom of the ellipse and slowest at the sides. */
    runs: [
      { condition: "slow", speed: 300, cycles: 1 },
      { condition: "medium", speed: 550, cycles: 2 },
      { condition: "fast", speed: 850, cycles: 2 },
    ],
    /** Fraction of the viewport left clear at the sides, and at top and bottom. */
    margin: 0.1,
    margin_y: 0.12,
    /** Cosine ease at each end of a sweep, in ms: no instantaneous jerk to fail to follow. */
    ramp_duration: 400,
    /** Still moment before the target sets off, so the first sweep is pursuit and not a search. */
    lead_in: 700,
    /** Fraction of runs with a color change. Keeps the eye on the target rather than ahead of it. */
    probe_rate: 0.67,
    /**
     * How many times the whole set of speeds is run. Zero leaves the block out.
     *
     * Zero in the main study. The block demonstrates the *consequence* of a timing error and
     * cannot measure the lag (too many unknowns share one observable; see the note at the top of
     * `analysis/pursuit.py`), and the static stations' saccade latencies already show what the
     * loopback correction is worth, against a known physiological range. Two pilot sessions
     * made the demonstration; more participants would not strengthen it, and it cost about
     * half a minute of each session. `pursuit-only.html` sets this to 1.
     */
    repeats: 0,
  },

  /**
   * Where the targets go, as fractions of the viewport. Every block collects the same
   * calibration grid, then a test set, so each block has the shape of a real deployment:
   * calibrate, then validate or run the task.
   *
   * The grid is 5x5 at 5/27.5/50/72.5/95%, which *nests* saccade.js's grids exactly: its
   * 3x3 (5/50/95) and its default 13 (those nine plus the four at 27.5/72.5) are both subsets.
   * Each grid station records `grid_level` -- 9, 13 or 25, the smallest of those grids it
   * belongs to -- so an analysis fits "a 13-point calibration" by taking `grid_level <= 13`.
   * That is why the calibration size is not randomized between participants: collecting the
   * superset from everyone makes every comparison of calibration size and layout within-subject,
   * and nobody sees a different task.
   *
   * The test set is saccade.js's 9-point validation grid (15/50/85%), which lets an analysis
   * emulate the validate plugin's score, plus `n_random` fresh positions per block. Those are
   * drawn one per cell of a 3x3 over the screen, so they cover it, and kept away from every
   * fixed point, so across blocks and participants the test set samples the whole screen
   * rather than the same 22 locations.
   */
  points: {
    /** The largest calibration grid collected: 9, 13 or 25. */
    grid: 25,
    /** Include saccade.js's 9-point validation grid in each block's test set. */
    validation: true,
    /** Fresh random test positions per block. */
    n_random: 9,
    /** Random positions stay inside [margin, 1 - margin] on both axes. */
    random_margin: 0.05,
    /** ...and at least this far, in viewport fractions, from every grid and validation point. */
    random_min_distance: 0.04,
    /**
     * Run the grid before the test points in each block rather than interleaving them. Off
     * interleaves everything, which removes the calibrate-then-test structure the forward-only
     * analysis in `replay.py` is modeled on.
     */
    grid_first: true,
  },

  time_sync: {
    /** Seconds of screen-to-webcam loopback. Longer gives more edges and a tighter estimate. */
    duration: 15000,
    /** "full" is black/white and gives the strongest signal; "reduced" is gentler and noisier. */
    contrast: "full",
  },

  /**
   * What identifiable image data the session is allowed to ship.
   *
   * Worth being exact about what is and is not in the data, because the two image streams are
   * very different and only one of them is a real exposure.
   *
   * WebGazer's stored patches are 10x6 grayscale per eye — sixty pixels, with the iris spanning
   * about three of them. Iris recognition wants an iris diameter of 120 px or more and
   * periocular recognition wants tens of pixels of skin texture; neither survives here, and an
   * upscaled patch is a dark blob beside a bright blob. It carries gaze, which is the point,
   * and not identity.
   *
   * `saccade_crop` is a different matter. It is the raw 144x36 strip across *both* eyes, so
   * roughly 50 x 36 px each — enough resolution to be a recognizable picture of someone's eyes.
   * It is only written when the encoder was deferred to a worker and the worker could not
   * finish before the drain deadline, which never happens on a machine with WebGPU and can
   * happen on a slow one. For a remotely deployed study that ships its data publicly, that is
   * the field to turn off.
   */
  privacy: {
    /**
     * Write raw eye crops for frames the worker could not encode.
     *
     * On, because the crops are worth keeping and the sharing decision is better made at
     * release time than at collection time: `analysis/redact.py` produces the public copy, and
     * it can encode the crops into embeddings on the way out — so the published dataset keeps
     * those frames as 128 numbers rather than losing them, and no recognizable eye image
     * leaves the lab.
     *
     * Turn this off for a study with no secure place to hold the raw session, where the
     * collected file *is* the published file. Then the frames are dropped instead.
     */
    save_eye_crops: true,
  },

  /**
   * Where the data goes. "download" writes a file in the browser, which is what you want while
   * piloting. "datapipe" uploads it, gzipped, and falls back to offering the participant a
   * download if the upload keeps failing, so a session is never lost to a network error.
   */
  save: {
    mode: "datapipe",
    datapipe_experiment_id: "0ixlhzrhiGQn",
    filename_prefix: "cookoff",
    /** Upload attempts before falling back to a download. Only network errors and server
     *  errors are retried; a 400 is DataPipe refusing the data, and resending will not help. */
    attempts: 4,
  },

  /**
   * Prolific. Participant, study and session IDs are read from the URL Prolific opens the study
   * with (`?PROLIFIC_PID=...&STUDY_ID=...&SESSION_ID=...`), and `PROLIFIC_PID` becomes the
   * `participant_id`. Without it, `?participant=` or a random ID is used, as when piloting.
   *
   * The codes are the ones set up in the study's Prolific settings. With a code empty, the
   * participant is told to go back to Prolific instead of being redirected.
   */
  prolific: {
    /** Sent to once the data is saved. */
    completion_code: "C3GCQAIQ",
    /**
     * Sent to by a participant who is screened out: no keyboard, or no face found by the
     * camera. Set up in Prolific as a screen-out code with a fixed payment, or leave empty to
     * ask them to return the study.
     */
    screenout_code: "",
    /** Give up on finding a face after this long, in ms, and screen the participant out. */
    face_timeout: 60000,
  },

  /** Questions asked at the end, after the eye tracking and before saving. */
  questions: {
    /** Glasses or contact lenses during the session. */
    eyewear: true,
    /** Two optional free-text questions: technical problems, and interruptions to looking at
     *  the targets. Demographics are not asked; Prolific supplies them. */
    self_report: true,
  },
};
