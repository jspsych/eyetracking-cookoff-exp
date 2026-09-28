/**
 * jsPsych plugin: a target moves across the screen and the participant follows it.
 *
 * Every other trial in this experiment holds its target still, which is exactly the condition
 * in which the timing measurement does not matter: a gaze sample that is 110 ms stale still
 * lands on a target that has not moved. Nothing in a static session can show what
 * `saccade-time-sync` bought.
 *
 * A moving target can. If gaze timestamps are wrong by dt and the target moves at v, then
 * lining gaze up against the target's position introduces a spatial error of v * dt — pure
 * artefact, nothing to do with the tracker's accuracy. At 700 px/s a 110 ms lag is 77 px,
 * which on the session this was built from is larger than saccade.js's entire static median
 * error. The error is invisible until something moves, and then it dominates.
 *
 * The path is an **ellipse**, and that is the whole design rather than a decoration. On a
 * straight sweep every kind of error looks the same: a timing lag, a pursuit gain deficit and a
 * plain miscalibration all leave the eye trailing the target, and nothing in the data separates
 * them. On a closed 2-D path they point in different directions and come apart geometrically:
 *
 *   - a **timing lag** puts the eye at an earlier phase, so its error is *tangential* — along
 *     the path, behind the target, whichever way the target happens to be going;
 *   - a **gain deficit** means the eye traces a smaller ellipse, so its error is *radial*,
 *     pointing in toward the center;
 *   - a **constant offset** is fixed in screen coordinates, so it rotates relative to the path
 *     and averages to zero around a full cycle while leaving a DC term in the raw error.
 *
 * One run therefore identifies three things that a straight line confounds. The ellipse is also
 * matched to the viewport's aspect, so speed varies continuously around the cycle — fastest
 * across the top and bottom, slowest at the sides — which gives a range of speeds inside every
 * run instead of one number per run.
 *
 * What the analysis gets from this trial:
 *   - `target_path`: where the target was, per animation frame, on the same clock as the gaze
 *     timestamps. Position at any gaze time is an interpolation away.
 *   - the usual per-frame gaze streams for both trackers, recorded continuously through the
 *     sweep rather than in a settle-then-capture window.
 *
 * The velocity profile is trapezoidal with cosine-eased ends, so there is no instantaneous jerk
 * for the eye to fail to follow. The eased ends are not marked: the analysis differentiates the
 * logged path and keeps the steady stretch, which needs no cooperation from this file.
 */
(function (global) {
  "use strict";

  const ParameterType = global.jsPsychModule.ParameterType;

  const info = {
    name: "cookoff-pursuit",
    version: "1.0.0",
    parameters: {
      /** Mean tangential speed around the ellipse, in CSS px per second. Instantaneous speed
       *  varies around it — fastest across the top and bottom, slowest at the sides. */
      speed: { type: ParameterType.FLOAT, default: 500 },
      /** How many complete circuits of the ellipse. */
      cycles: { type: ParameterType.FLOAT, default: 2 },
      /** Fraction of the viewport left clear at each side, and at top and bottom. */
      margin: { type: ParameterType.FLOAT, default: 0.1 },
      margin_y: { type: ParameterType.FLOAT, default: 0.12 },
      /** Cosine ease at each end of a sweep, in ms. Keeps the turnaround pursuable. */
      ramp_duration: { type: ParameterType.INT, default: 400 },
      /** 1 goes clockwise from the top, -1 counter-clockwise. */
      direction: { type: ParameterType.INT, default: 1 },
      /** How long the target sits still before it sets off, in ms. Not recorded. */
      lead_in: { type: ParameterType.INT, default: 600 },
      /** Whether the target's center changes color mid-run, and the response key for it. */
      probe: { type: ParameterType.BOOL, default: false },
      response_deadline: { type: ParameterType.INT, default: 1000 },
      response_key: { type: ParameterType.STRING, default: " " },
      /** Recorded as-is. */
      block: { type: ParameterType.INT, default: 0 },
      condition: { type: ParameterType.STRING, default: "" },
      index_in_block: { type: ParameterType.INT, default: 0 },
      block_length: { type: ParameterType.INT, default: 0 },
    },
    data: {
      /** Serial number, shared with the capture trials; the deferred encoder writes back by it. */
      station_id: { type: ParameterType.INT },
      /** Mean tangential speed in px/s, circuits, and the label for this speed condition. */
      speed: { type: ParameterType.FLOAT },
      cycles: { type: ParameterType.FLOAT },
      condition: { type: ParameterType.STRING },
      /** The ellipse the target ran on, in viewport pixels, and which way round it went. The
       *  logged path already carries it; these make the geometry readable without differencing. */
      semi_axis_x: { type: ParameterType.INT },
      semi_axis_y: { type: ParameterType.INT },
      center_x: { type: ParameterType.INT },
      center_y: { type: ParameterType.INT },
      direction: { type: ParameterType.INT },
      block: { type: ParameterType.INT },
      viewport_width: { type: ParameterType.INT },
      viewport_height: { type: ParameterType.INT },
      device_pixel_ratio: { type: ParameterType.FLOAT },
      /** `performance.now()` when the target started moving. Everything below is relative to it. */
      capture_onset: { type: ParameterType.FLOAT },
      /**
       * Where the target was, one row per animation frame: base64 little-endian int16, stride 3,
       * `[t_ms, x_px, y_px]`. `t_ms` is the `requestAnimationFrame` timestamp relative to
       * `capture_onset` — the moment the browser began painting that position, which is the
       * same clock a stimulus onset is on and the clock `timing_lag_ms` corrects *to*.
       */
      target_path: { type: ParameterType.STRING },
      /** Animation frames painted, and the mean interval between them. */
      path_frames: { type: ParameterType.INT },
      raf_period_ms: { type: ParameterType.FLOAT },
      /** The gaze streams, identical in shape to a `cookoff-capture` station. */
      n_frames: { type: ParameterType.INT },
      frame_t: { type: ParameterType.FLOAT, array: true },
      clock_source: { type: ParameterType.STRING },
      n_face_missing: { type: ParameterType.INT },
      n_camera_dropped: { type: ParameterType.INT },
      n_skipped: { type: ParameterType.INT },
      embedded_frames: { type: ParameterType.INT, array: true },
      saccade_embedding: { type: ParameterType.STRING },
      saccade_weight: { type: ParameterType.STRING },
      crop_frames: { type: ParameterType.INT, array: true },
      saccade_crop: { type: ParameterType.STRING },
      webgazer_gray_left: { type: ParameterType.STRING },
      webgazer_gray_right: { type: ParameterType.STRING },
      webgazer_eye_box: { type: ParameterType.STRING },
      head_landmarks: { type: ParameterType.STRING },
      eye_aspect_ratio: { type: ParameterType.STRING },
      eye_image_stats: { type: ParameterType.STRING },
      fps: { type: ParameterType.FLOAT },
      /** The color-change probe, as on a static target. */
      probe: { type: ParameterType.BOOL },
      probe_onset: { type: ParameterType.FLOAT },
      response_time: { type: ParameterType.FLOAT },
      probe_rt: { type: ParameterType.FLOAT },
      probe_outcome: { type: ParameterType.STRING },
      page_hidden: { type: ParameterType.BOOL },
      /** Whether the window left fullscreen during this run. Leaving fullscreen resizes the
       *  viewport, which moves the target relative to a calibration taken at the old size. */
      fullscreen_exited: { type: ParameterType.BOOL },
      /** Whether the window lost focus during this run. The page stays visible and the camera
       *  keeps delivering, so the frames look healthy while the participant is elsewhere. */
      window_blurred: { type: ParameterType.BOOL },
    },
  };

  /**
   * Advance a quantity at a rate, with cosine-eased ends.
   *
   * Used for *phase* around the ellipse rather than distance along a line, which is what keeps
   * the path exactly elliptical while the target speeds up and slows down: easing the amplitude
   * instead would deform the shape, and starting at full speed would ask the eye to follow a
   * step in velocity it cannot.
   *
   * Under a cosine ease `rate(t) = rate * (1 - cos(pi t / ramp)) / 2`, whose integral over a
   * whole ramp is `rate * ramp / 2` — so the two ramps together cover `rate * ramp` and the rest
   * runs at full rate.
   */
  function rampedProfile(total, rate, rampMs) {
    const ramp = Math.min(rampMs, (total / rate) * 1000 * 0.5);
    const rampAmount = (rate * ramp) / 1000; // per ramp: rate * ramp / 2, twice
    const cruiseMs = (Math.max(0, total - rampAmount) / rate) * 1000;
    const duration = 2 * ramp + cruiseMs;

    const advanced = (t) => {
      if (t <= 0) return 0;
      if (t >= duration) return total;
      if (t < ramp) {
        const phase = (Math.PI * t) / ramp;
        return ((rate / 1000) * (t - (ramp / Math.PI) * Math.sin(phase))) / 2;
      }
      if (t < ramp + cruiseMs) {
        return rampAmount / 2 + (rate / 1000) * (t - ramp);
      }
      const back = duration - t;
      const phase = (Math.PI * back) / ramp;
      return total - ((rate / 1000) * (back - (ramp / Math.PI) * Math.sin(phase))) / 2;
    };
    return { advanced, duration, ramp, cruiseMs };
  }

  /**
   * Ramanujan's second approximation to the ellipse perimeter.
   *
   * Needed because the runs are specified by mean tangential speed, which is the quantity that
   * means something to a reader, and converting it to an angular rate needs the circumference.
   * Accurate to better than one part in a million at these eccentricities — far inside the pixel
   * rounding the path is stored with.
   */
  function ellipsePerimeter(a, b) {
    return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
  }

  class CookoffPursuitPlugin {
    static info = info;

    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial, on_load) {
      const recorder = global.Cookoff.recorder;
      if (!recorder) throw new Error("cookoff-pursuit: the recorder has not been started");

      const stationId = global.jsPsychCookoffCapture.nextStationId();
      const width = window.innerWidth;
      const height = window.innerHeight;

      // Semi-axes matched to the viewport, so the ellipse uses the shape of the screen rather
      // than being squeezed into the circle its short side would allow.
      const a = Math.round(((1 - 2 * trial.margin) * width) / 2);
      const b = Math.round(((1 - 2 * trial.margin_y) * height) / 2);
      const cx = Math.round(width / 2);
      const cy = Math.round(height / 2);

      // Runs are specified by mean tangential speed because that is the number that means
      // something; the angular rate follows from the circumference.
      const perimeter = ellipsePerimeter(a, b);
      const omega = (2 * Math.PI * trial.speed) / perimeter; // rad/s
      const profile = rampedProfile(2 * Math.PI * trial.cycles, omega, trial.ramp_duration);
      const totalMs = profile.duration;

      display_element.innerHTML = `
        <div class="ck-stage">
          <div class="ck-hud">
            <div class="ck-hud-item"><span class="ck-hud-label">Progress</span><span class="ck-hud-value">${trial.index_in_block + 1}/${trial.block_length}</span></div>
          </div>
          <div class="ck-target ck-target-capturing ck-moving" id="ck-target" style="left:0px; top:0px;">
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
        speed: trial.speed,
        cycles: trial.cycles,
        condition: trial.condition,
        semi_axis_x: a,
        semi_axis_y: b,
        center_x: cx,
        center_y: cy,
        direction: trial.direction,
        block: trial.block,
        viewport_width: width,
        viewport_height: height,
        device_pixel_ratio: window.devicePixelRatio || 1,
        probe: trial.probe,
        probe_onset: null,
        response_time: null,
        probe_rt: null,
        probe_outcome: null,
        page_hidden: document.hidden,
        fullscreen_exited: false,
        window_blurred: !document.hasFocus(),
      };

      const stopWatching = global.jsPsychCookoffCapture.watchDistractions(data);

      /**
       * Where the target is, `t` ms into the run.
       *
       * Phase zero is the top of the ellipse, which is where it waits during the lead-in, and
       * `direction` decides which way round it goes. Screen y grows downward, hence the minus.
       */
      const positionAt = (t) => {
        const phase = trial.direction * profile.advanced(t);
        return { x: cx + a * Math.sin(phase), y: cy - b * Math.cos(phase) };
      };

      let captureOnset = null;
      let rafHandle = null;
      let finished = false;
      let responded = false;
      let probeOnset = null;
      const path = []; // [t_ms, x, y] per painted frame
      const timers = [];
      const later = (fn, delay) => {
        const id = setTimeout(fn, delay);
        timers.push(id);
        return id;
      };

      const onResponse = (response) => {
        if (responded || captureOnset == null) return;
        responded = true;
        data.response_time = Math.round(response.rt * 10) / 10;
        if (probeOnset != null && response.rt - probeOnset <= trial.response_deadline) {
          data.probe_rt = Math.round((response.rt - probeOnset) * 10) / 10;
          data.probe_outcome = "hit";
          core.classList.remove("ck-core-probe");
          target.classList.add("ck-hit");
        } else {
          if (probeOnset != null) data.probe_rt = Math.round((response.rt - probeOnset) * 10) / 10;
          data.probe_outcome = "false_alarm";
          target.classList.add("ck-miss");
        }
      };

      const finish = async () => {
        if (finished) return;
        finished = true;
        if (rafHandle != null) cancelAnimationFrame(rafHandle);
        for (const id of timers) clearTimeout(id);
        stopWatching();
        this.jsPsych.pluginAPI.cancelAllKeyboardResponses();

        await recorder.flush();
        const buffer = recorder.endCapture();
        Object.assign(data, recorder.pack(buffer, captureOnset));
        data.capture_onset = captureOnset;
        data.fps = Math.round(recorder.fps * 10) / 10;

        const rows = path.map((p) => [Math.round(p[0]), Math.round(p[1]), Math.round(p[2])]);
        data.target_path = global.CookoffEncoding.encodeI16Rows(rows, 3);
        data.path_frames = rows.length;
        data.raf_period_ms =
          rows.length > 1
            ? Math.round(((rows[rows.length - 1][0] - rows[0][0]) / (rows.length - 1)) * 100) / 100
            : null;

        if (data.probe_outcome === null) {
          data.probe_outcome = trial.probe ? "miss" : "correct_rejection";
        }
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(data);
      };

      const step = (now) => {
        const t = now - captureOnset;
        const p = positionAt(t);
        // translate3d rather than `left`/`top`: the position is set on the compositor, so the
        // paint this timestamp refers to is the one the participant actually sees.
        target.style.transform = `translate3d(${p.x}px, ${p.y}px, 0)`;
        path.push([t, p.x, p.y]);
        if (t >= totalMs) {
          void finish();
          return;
        }
        rafHandle = requestAnimationFrame(step);
      };

      // A beat to find the target before it sets off, so the first sweep is pursuit and not a
      // search. Nothing is recorded during it.
      const start = positionAt(0);
      target.style.transform = `translate3d(${start.x}px, ${start.y}px, 0)`;
      later(() => {
        captureOnset = performance.now();
        // The target has been sitting at its start position all through the lead-in, and the
        // first animation frame is still one frame away. Without this sample the path begins
        // ~17 ms late and every gaze frame before it interpolates to nothing.
        const origin = positionAt(0);
        path.push([0, origin.x, origin.y]);
        recorder.beginCapture({ stationId });

        this.jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: onResponse,
          valid_responses: [trial.response_key],
          rt_method: "performance",
          persist: false,
          allow_held_key: false,
        });

        if (trial.probe) {
          // Somewhere in the middle half of the run, so it never lands on a turnaround.
          const delay = totalMs * (0.25 + Math.random() * 0.5);
          later(() => {
            probeOnset = performance.now() - captureOnset;
            data.probe_onset = Math.round(probeOnset * 10) / 10;
            core.classList.add("ck-core-probe");
            later(() => core.classList.remove("ck-core-probe"), trial.response_deadline);
          }, delay);
        }

        rafHandle = requestAnimationFrame(step);
      }, trial.lead_in);

      if (on_load) on_load();
    }
  }

  global.jsPsychCookoffPursuit = CookoffPursuitPlugin;
  global.jsPsychCookoffPursuit.rampedProfile = rampedProfile;
  global.jsPsychCookoffPursuit.ellipsePerimeter = ellipsePerimeter;
})(window);
