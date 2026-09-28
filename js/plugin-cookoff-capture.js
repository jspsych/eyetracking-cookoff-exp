/**
 * jsPsych plugin: one fixation target.
 *
 * A target appears at a point on the screen and the recorder holds a capture window open while
 * the participant looks at it. On some trials the small dot at the center changes color, and
 * the participant presses the response key.
 *
 * The probe is not decoration. Nothing in this experiment predicts gaze in real time, so
 * nothing can tell afterwards whether a participant was looking where the data says they
 * were. A change confined to a few pixels at the center of the target cannot be caught
 * peripherally: a hit is direct evidence that the fovea was on the target, and a miss marks a
 * capture window an analysis may want to drop.
 *
 * The capture-phase target is the concentric bullseye from Thaler et al. (2013) — a filled
 * disc, a cross cut through it, a small central dot — which people fixate more stably than a
 * plain dot, and which gives the probe somewhere to live that does not move.
 */
(function (global) {
  "use strict";

  const ParameterType = global.jsPsychModule.ParameterType;

  /**
   * Stations are numbered across the whole session, and the pursuit plugin draws from the same
   * counter: the deferred encoder finds a row by `station_id`, so the two trial types must
   * never mint the same one.
   */
  let stationCounter = 0;
  const nextStationId = () => stationCounter++;

  /**
   * Watch for the three ways a trial can stop describing the participant.
   *
   * Each sets a flag on the trial's own data and returns a teardown. All three are distinct and
   * all three matter:
   *
   *   - **hidden** — the tab went to the background. Timers throttle to about 1 Hz and camera
   *     frames stop arriving, so the capture window is mostly empty.
   *   - **blurred** — the window lost focus while still *visible*. Nothing throttles, the camera
   *     keeps delivering and the frames look perfectly healthy; the participant is simply
   *     looking at another window. This is the one that leaves no trace anywhere else, which is
   *     exactly why it is worth a flag.
   *   - **fullscreen exited** — the viewport resized, so this trial's target positions are on a
   *     different screen from the calibration collected before it.
   *
   * jsPsych records the same events globally, with timestamps, and the session saves that log
   * too. These flags are the per-trial summary of it, so the common case — "drop the trials
   * where something happened" — needs no cross-referencing.
   */
  function watchDistractions(data) {
    const onVisibility = () => {
      if (document.hidden) data.page_hidden = true;
    };
    const onBlur = () => {
      data.window_blurred = true;
    };
    const onFullscreen = () => {
      if (!document.fullscreenElement) data.fullscreen_exited = true;
    };
    document.addEventListener("visibilitychange", onVisibility);
    document.addEventListener("fullscreenchange", onFullscreen);
    window.addEventListener("blur", onBlur);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      document.removeEventListener("fullscreenchange", onFullscreen);
      window.removeEventListener("blur", onBlur);
    };
  }

  /**
   * Where the last station's target was, and which station it belonged to.
   *
   * A saccade needs somewhere to have come *from*. Cleared by `forgetPreviousTarget()`, which
   * the timeline calls after anything that breaks the run of stations — a break screen, a
   * block intro — because after one of those the eye was on a button, not on the last target,
   * and the "saccade" would be measured from a place it never was.
   */
  let previousTarget = null;
  const forgetPreviousTarget = () => {
    previousTarget = null;
  };

  const info = {
    name: "cookoff-capture",
    version: "1.0.0",
    parameters: {
      /** Horizontal position of the target, as a fraction of the viewport width. */
      x: { type: ParameterType.FLOAT, default: undefined },
      /** Vertical position of the target, as a fraction of the viewport height. */
      y: { type: ParameterType.FLOAT, default: undefined },
      /** How long the target takes to appear and settle, in ms. */
      settle_duration: { type: ParameterType.INT, default: 600 },
      /** Whether frames during the settle are recorded too — the arrival saccade lives there. */
      record_settle: { type: ParameterType.BOOL, default: false },
      /** How long frames are recorded for, in ms, measured from the end of the settle. */
      capture_duration: { type: ParameterType.INT, default: 1000 },
      /** Whether this target's center changes color during the capture window. */
      probe: { type: ParameterType.BOOL, default: false },
      /** Earliest and latest color-change onset, in ms from the start of the capture window. */
      probe_window: { type: ParameterType.INT, array: true, default: [300, 700] },
      /** How long after the color change a key press still counts, in ms. Recording
       *  continues to the end of this window even if it outlasts `capture_duration`, so a
       *  late change is not followed by a truncated capture. */
      response_deadline: { type: ParameterType.INT, default: 1000 },
      /** Key the participant presses when the center changes color. */
      response_key: { type: ParameterType.STRING, default: " " },
      /** Which block this target belongs to. Recorded as-is. */
      block: { type: ParameterType.INT, default: 0 },
      /** Which pass over the grid this is. Recorded as-is; this is what held-out splits use. */
      rep: { type: ParameterType.INT, default: 0 },
      /** Label for the point set this target came from, e.g. "train" or "test". */
      point_set: { type: ParameterType.STRING, default: "" },
      /** Position of this target within its block, and the block length, for the progress readout. */
      index_in_block: { type: ParameterType.INT, default: 0 },
      block_length: { type: ParameterType.INT, default: 0 },
    },
    data: {
      /** Target center in viewport pixels, and as a fraction of the viewport. */
      target_x: { type: ParameterType.INT },
      target_y: { type: ParameterType.INT },
      target_x_frac: { type: ParameterType.FLOAT },
      target_y_frac: { type: ParameterType.FLOAT },
      /** Viewport size at the moment of capture, needed to turn fractions back into pixels. */
      viewport_width: { type: ParameterType.INT },
      viewport_height: { type: ParameterType.INT },
      device_pixel_ratio: { type: ParameterType.FLOAT },
      /** Block, pass over the grid, and which point set this target came from. */
      block: { type: ParameterType.INT },
      rep: { type: ParameterType.INT },
      point_set: { type: ParameterType.STRING },
      /** `performance.now()` when recording opened, which `frame_t` is relative to. With
       *  `record_settle` that is when the target appeared; without it, when the steady
       *  capture window began. */
      capture_onset: { type: ParameterType.FLOAT },
      /** The settle length, and the index of the first frame at or after the end of it. Frames
       *  before `fixation_from` cover the eye arriving and are not steady fixation; an accuracy
       *  analysis drops them. Null when the settle was not recorded, meaning every frame is
       *  fixation. */
      settle_duration: { type: ParameterType.INT },
      fixation_from: { type: ParameterType.INT },
      /** Where the previous station's target was, and its id — the start of the arrival
       *  saccade. Null for the first station after any break in the run. */
      previous_target_x: { type: ParameterType.INT },
      previous_target_y: { type: ParameterType.INT },
      previous_station_id: { type: ParameterType.INT },
      /** Camera frames kept: a face found, an eye crop cut, and both feature sets built. */
      n_frames: { type: ParameterType.INT },
      /** Capture time of each kept frame, in ms from `capture_onset`, before any lag correction. */
      frame_t: { type: ParameterType.FLOAT, array: true },
      /** Which camera clock `frame_t` came from: captureTime, receiveTime, or callback. */
      clock_source: { type: ParameterType.STRING },
      /** Frames in the window with no face, and frames the browser reported the camera dropping. */
      n_face_missing: { type: ParameterType.INT },
      n_camera_dropped: { type: ParameterType.INT },
      /** Frames with a face where an eye crop or an eye patch could not be cut. */
      n_skipped: { type: ParameterType.INT },
      /** saccade.js eye embeddings: base64 little-endian float16, `n_frames` x `embedding_dim`. */
      saccade_embedding: { type: ParameterType.STRING },
      /** The model's per-frame quality weight: base64 float16, one per frame. */
      saccade_weight: { type: ParameterType.STRING },
      /** WebGazer's 10x6 grayscale eye patches: base64 uint8, 60 bytes per eye per frame.
       *  Equalize each with `equalizeHistogram(patch, 5)` and concatenate left then right to
       *  get the 120-number vector WebGazer's ridge regression actually sees. */
      webgazer_gray_left: { type: ParameterType.STRING },
      webgazer_gray_right: { type: ParameterType.STRING },
      /** The eye boxes those patches were cut from: base64 int16, `[lx, ly, lw, lh, rx, ry, rw, rh]`. */
      webgazer_eye_box: { type: ParameterType.STRING },
      /** Nine landmarks as (x, y, z): base64 float16, a compact record of head position. */
      head_landmarks: { type: ParameterType.STRING },
      /** Eye aspect ratio, left then right: base64 float16. Low values mean a closing eye. */
      eye_aspect_ratio: { type: ParameterType.STRING },
      /** Strided mode: the frames whose embeddings were encoded after the station, in the
       *  downtime. The rest of `embedded_frames` were encoded live. Null on the other paths. */
      backfilled_frames: { type: ParameterType.INT, array: true },
      /** The eye strip's sharpness, luminance and noise per frame: base64 float16, stride 3. */
      eye_image_stats: { type: ParameterType.STRING },
      /** Recorder frame rate at the end of the window. */
      fps: { type: ParameterType.FLOAT },
      /** Whether this target's center changed color, and when, in ms from `capture_onset`. */
      probe: { type: ParameterType.BOOL },
      probe_onset: { type: ParameterType.FLOAT },
      /** Key press time in ms from `capture_onset`, or null. */
      response_time: { type: ParameterType.FLOAT },
      /** Time from the color change to the key press, in ms, or null. */
      probe_rt: { type: ParameterType.FLOAT },
      /** "hit", "miss", "false_alarm" or "correct_rejection". */
      probe_outcome: { type: ParameterType.STRING },
      /** Serial number of this station within the session. When the encoder is deferred to a
       *  worker, this is how its results find their way back into this row after the fact. */
      station_id: { type: ParameterType.INT },
      /** Indices into `frame_t` of the frames that have an embedding, or null meaning all of
       *  them. A subset when the encoder was deferred and fell behind (the rest are in
       *  `crop_frames`), or budgeted (before 1.3.1). Line the two trackers
       *  up on these frames: WebGazer's features are present for every frame, saccade.js's
       *  embedding is not. */
      embedded_frames: { type: ParameterType.INT, array: true },
      /** Frames whose crops are in `saccade_crop` because the worker never reached them, and
       *  the raw 144x36 crops themselves, base64 uint8. Encode them offline. Usually null. */
      crop_frames: { type: ParameterType.INT, array: true },
      saccade_crop: { type: ParameterType.STRING },
      /** Whether the page was hidden at any point during this station. A hidden tab has its
       *  timers throttled to about 1 Hz and stops receiving camera frames, so a station with
       *  this set describes a browser, not a participant. Exclude it. */
      page_hidden: { type: ParameterType.BOOL },
      /** Whether the window left fullscreen during this trial. Leaving fullscreen resizes the
       *  viewport, which moves every target relative to a calibration collected at the old size
       *  — so this is not a cosmetic event, it is the point after which positions from before
       *  and after are not comparable. */
      fullscreen_exited: { type: ParameterType.BOOL },
      /** Whether the window lost focus during this trial. Unlike `page_hidden` this leaves the
       *  page visible and the camera running, so the frames look healthy while the participant
       *  is looking at another window. Exclude it. */
      window_blurred: { type: ParameterType.BOOL },
    },
  };

  class CookoffCapturePlugin {
    static info = info;

    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial, on_load) {
      const recorder = global.Cookoff.recorder;
      if (!recorder) throw new Error("cookoff-capture: the recorder has not been started");

      const stationId = nextStationId();
      const width = window.innerWidth;
      const height = window.innerHeight;
      const px = Math.round(trial.x * width);
      const py = Math.round(trial.y * height);

      display_element.innerHTML = `
        <div class="ck-stage">
          <div class="ck-hud">
            <div class="ck-hud-item"><span class="ck-hud-label">Progress</span><span class="ck-hud-value">${trial.index_in_block + 1}/${trial.block_length}</span></div>
          </div>
          <div class="ck-target ck-target-landing" id="ck-target" style="left:${px}px; top:${py}px;">
            <span class="ck-bullseye" aria-hidden="true">
              <span class="ck-bullseye-cross"></span>
              <span class="ck-bullseye-core" id="ck-core"></span>
            </span>
          </div>
        </div>`;

      const target = display_element.querySelector("#ck-target");
      const core = display_element.querySelector("#ck-core");

      const data = {
        station_id: stationId,
        target_x: px,
        target_y: py,
        target_x_frac: trial.x,
        target_y_frac: trial.y,
        viewport_width: width,
        viewport_height: height,
        device_pixel_ratio: window.devicePixelRatio || 1,
        settle_duration: trial.record_settle ? trial.settle_duration : null,
        fixation_from: null,
        previous_target_x: previousTarget ? previousTarget.x : null,
        previous_target_y: previousTarget ? previousTarget.y : null,
        previous_station_id: previousTarget ? previousTarget.station_id : null,
        block: trial.block,
        rep: trial.rep,
        point_set: trial.point_set,
        probe: trial.probe,
        probe_onset: null,
        response_time: null,
        probe_rt: null,
        probe_outcome: null,
        page_hidden: document.hidden,
        fullscreen_exited: false,
        window_blurred: !document.hasFocus(),
      };

      // A backgrounded tab throttles setTimeout to roughly 1 Hz and stops delivering camera
      // frames, which stretches the capture window and empties it. Nothing here can prevent
      // that; marking the station is what lets the analysis throw it away instead of treating
      // three frames over four seconds as a fixation.
      const stopWatching = watchDistractions(data);

      let captureOnset = null;
      let settleEndedAt = null;
      /** Offset, in ms from `captureOnset`, at which the steady fixation window begins. */
      const fixationStart = () => (trial.record_settle ? (settleEndedAt ?? trial.settle_duration) : 0);
      let probeOnset = null;
      let responded = false;
      let finished = false;
      let endTimer = null;
      const timers = [];
      const later = (fn, delay) => {
        const id = setTimeout(fn, delay);
        timers.push(id);
        return id;
      };

      const endAt = (msFromCaptureOnset) => {
        if (endTimer != null) clearTimeout(endTimer);
        const delay = Math.max(0, captureOnset + msFromCaptureOnset - performance.now());
        endTimer = later(() => void finish(), delay);
      };

      const onResponse = (response) => {
        if (responded || captureOnset == null) return;
        responded = true;
        /* `rt_method: "performance"` measures from when the listener was registered, which is
           the start of the *fixation* window. `frame_t` and `probe_onset` are measured from
           when *recording* opened, which with `record_settle` is one settle earlier. Shift the
           response onto the recording clock so a row never carries two origins. */
        const rt = response.rt + fixationStart();
        data.response_time = Math.round(rt * 10) / 10;
        if (probeOnset != null && rt - probeOnset <= trial.response_deadline) {
          data.probe_rt = Math.round((rt - probeOnset) * 10) / 10;
          data.probe_outcome = "hit";
          core.classList.remove("ck-core-probe");
          target.classList.add("ck-hit");
        } else {
          if (probeOnset != null) data.probe_rt = Math.round((rt - probeOnset) * 10) / 10;
          data.probe_outcome = "false_alarm";
          target.classList.add("ck-miss");
        }
      };

      const finish = async () => {
        if (finished) return;
        finished = true;
        for (const id of timers) clearTimeout(id);
        stopWatching();
        this.jsPsych.pluginAPI.cancelAllKeyboardResponses();

        await recorder.flush();
        const buffer = recorder.endCapture();
        Object.assign(data, recorder.pack(buffer, captureOnset));
        data.capture_onset = captureOnset;
        data.fps = Math.round(recorder.fps * 10) / 10;

        if (trial.record_settle) {
          // The first frame at or after the settle ended. Everything before it is the eye on
          // its way here; everything from it on is steady fixation.
          const boundary = fixationStart();
          const index = data.frame_t.findIndex((t) => t >= boundary);
          data.fixation_from = index === -1 ? data.n_frames : index;
        }
        previousTarget = { x: px, y: py, station_id: stationId };

        if (data.probe_outcome === null) {
          data.probe_outcome = trial.probe ? "miss" : "correct_rejection";
        }
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(data);
      };

      // The target appears at full size-change speed and settles. The luminance onset is
      // abrupt — that is what the saccade-latency measurement needs — and only the scale
      // animates, which never moves the center.
      later(() => {
        target.classList.remove("ck-target-landing");
        target.classList.add("ck-target-settled");
      }, Math.round(trial.settle_duration * 0.45));

      /**
       * Open the recorder.
       *
       * With `record_settle` this runs at trial start, so the station captures the eye leaving
       * the previous target, the saccade across, and the fixation that follows. Without it the
       * recorder opens after the settle and sees only steady fixation.
       */
      const openRecorder = () => {
        captureOnset = performance.now();
        recorder.beginCapture({ stationId });
      };

      if (trial.record_settle) openRecorder();

      later(() => {
        if (!trial.record_settle) openRecorder();
        // The boundary `fixation_from` is computed from.
        else settleEndedAt = performance.now() - captureOnset;
        target.classList.add("ck-target-capturing");

        this.jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: onResponse,
          valid_responses: [trial.response_key],
          rt_method: "performance",
          persist: false,
          allow_held_key: false,
        });

        endAt(fixationStart() + trial.capture_duration);

        if (trial.probe) {
          const [lo, hi] = trial.probe_window;
          later(
            () => {
              probeOnset = performance.now() - captureOnset;
              data.probe_onset = Math.round(probeOnset * 10) / 10;
              core.classList.add("ck-core-probe");
              // Recording runs to the end of the response window, so a late change still has
              // a full capture behind it rather than a truncated one.
              endAt(probeOnset + trial.response_deadline);
            },
            lo + Math.random() * Math.max(0, hi - lo),
          );
        }
      }, trial.settle_duration);

      if (on_load) on_load();
    }
  }

  global.jsPsychCookoffCapture = CookoffCapturePlugin;
  global.jsPsychCookoffCapture.nextStationId = nextStationId;
  global.jsPsychCookoffCapture.watchDistractions = watchDistractions;
  global.jsPsychCookoffCapture.forgetPreviousTarget = forgetPreviousTarget;
})(window);
