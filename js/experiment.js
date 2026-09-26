/**
 * A fixation session that records what saccade.js and WebGazer would each have seen, without
 * either of them predicting anything.
 *
 * Structure: camera setup, a live frame-rate check, the screen-to-webcam timing measurement, a
 * warm-up that confirms the recorder keeps up on this machine, a practice block, one block per
 * pass over the point grid, then the moving-target block if `pursuit.repeats` asks for one.
 *
 * On the participant-facing copy: it says what to do and stops. The reasons any of this is
 * built the way it is live in the code and the docs, where they are useful. A participant
 * reading three paragraphs of rationale before each phase is a participant who has stopped
 * reading, and the instructions that matter — hold still, look at the center, press space —
 * are the ones that get skimmed.
 */
(function (global) {
  "use strict";

  const config = global.CookoffConfig;

  /** How long the recorder runs on a live camera before its frame rate is believed. */
  const WARMUP_MS = 3000;
  /** How long the camera's frame rate is measured for, before the encoder is timed. */
  const CAMERA_MEASURE_MS = 1500;

  /** Page-level state the plugins and the scoreboards read. */
  const Cookoff = (global.Cookoff = {
    recorder: null,
    tracker: null,
    model: null,
    extension: null,
    embedder: null,
    /** Mirror of jsPsych's interaction log, for anything that wants it mid-session. */
    interactions: [],
  });

  // ---------------------------------------------------------------------------- setup

  /*
   * One camera and one encoder for the whole page, built here and handed to the extension
   * rather than letting it build its own.
   *
   * The setup trials (`saccade-preview`, `saccade-time-sync`) drive saccade.js's tracker; the
   * capture trials drive this experiment's recorder. If those were two trackers, the timing
   * lag would be measured on one camera stream and applied to frames from another, and the
   * 20 MB encoder would be resident twice.
   *
   * Sharing the *model* has one condition: `OrtEmbeddingModel` keeps a single reusable input
   * tensor, so two inferences must never be in flight at once. They never are — the recorder
   * stops the tracker's frame loop before starting its own, and `runLoopback` stops it for the
   * duration of the timing measurement.
   */
  Cookoff.model = new Saccade.OrtEmbeddingModel({ assets: config.assets });
  Cookoff.tracker = new Saccade.SaccadeTracker({
    assets: config.assets,
    video: config.video,
    model: Cookoff.model,
  });

  const jsPsych = initJsPsych({
    extensions: [{ type: jsPsychExtensionSaccade, params: { tracker: Cookoff.tracker } }],
    display_element: "jspsych-target",
    // Focus, blur and fullscreen changes, with timestamps and the trial they fell in. Saved
    // with the session; the per-trial flags on each row are the summary of the same events.
    on_interaction_data_update: (event) => {
      Cookoff.interactions.push(event);
    },
    // Task trials paint the task field over the whole viewport, and jsPsych empties the display
    // for `post_trial_gap` between them. Without this the page background shows through in
    // every gap: a darker flash on each of ~130 stations, which is a luminance step for the
    // camera's auto-exposure and the participant's pupils as well as a distraction. The page
    // takes the field's color for task trials and the gaps after them, and returns to the
    // darker ground for the screens between blocks.
    on_trial_start: (trial) => {
      const task = trial.type === jsPsychCookoffCapture || trial.type === jsPsychCookoffPursuit;
      document.documentElement.classList.toggle("ck-task", task);
    },
  });
  // Exposed for the console: checking what the tracker and the recorder are actually doing
  // mid-session is most of debugging a webcam experiment.
  Cookoff.jsPsych = jsPsych;

  /*
   * Who this is. Prolific opens the study with its own IDs in the URL; `PROLIFIC_PID` is the one
   * the payment and any follow-up are keyed on, so it becomes `participant_id`. `?participant=`
   * and a random ID are for piloting.
   */
  const params = new URLSearchParams(window.location.search);
  const prolific = {
    prolific_pid: params.get("PROLIFIC_PID"),
    prolific_study_id: params.get("STUDY_ID"),
    prolific_session_id: params.get("SESSION_ID"),
  };
  const participantId =
    prolific.prolific_pid || params.get("participant") || jsPsych.randomization.randomID(10);

  /*
   * A participant who arrived from Prolific against a config still set up for piloting would do
   * the whole session and then have their data downloaded to their own machine, or go nowhere.
   * Refuse before they start instead. Piloting without Prolific IDs is unaffected.
   */
  if (prolific.prolific_pid) {
    const missing = [
      config.save.mode !== "datapipe" && 'save.mode is not "datapipe"',
      config.save.mode === "datapipe" && !config.save.datapipe_experiment_id &&
        "save.datapipe_experiment_id is empty",
      !config.prolific.completion_code && "prolific.completion_code is empty",
    ].filter(Boolean);
    if (missing.length) {
      console.error("cook-off: not configured for collection:", missing);
      document.getElementById("jspsych-target").innerHTML = `<div class="ck-card">
        <h1>This study is not open yet</h1>
        <p>Sorry for the trouble. Please return the study on Prolific. Nothing has been
        recorded.</p></div>`;
      return;
    }
  }

  // A session that did not arrive from Prolific is a pilot: it still uploads, so the whole path
  // gets exercised, but under its own filename prefix and flagged in the data, so it cannot be
  // mistaken for a participant.
  const pilot = !prolific.prolific_pid;
  if (pilot) config.save.filename_prefix = `pilot_${config.save.filename_prefix || "cookoff"}`;

  jsPsych.data.addProperties({
    participant_id: participantId,
    pilot,
    ...prolific,
    experiment: "eyetracking-cookoff",
    version: config.version,
    user_agent: navigator.userAgent,
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    fullscreen_requested: config.fullscreen,
    ...CookoffCapability.hardware(),
  });

  // Which saccade.js build this session ran, from the record `scripts/sync-saccade.sh` writes.
  fetch("lib/saccade/SYNCED.json", { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : null))
    .then((build) => {
      if (build) {
        jsPsych.data.addProperties({
          saccade_build_version: build.core_version ?? null,
          saccade_build_commit: build.commit ?? null,
          saccade_build_dirty: build.dirty ?? null,
        });
      }
    })
    .catch(() => {});

  // Whether WebGPU exists here, which is a different question from whether ONNX Runtime
  // managed to use it. Probed up front, before the camera or the model are involved, so the
  // answer is about the machine and not about anything this experiment did to it.
  CookoffCapability.probeWebGPU().then((webgpu) => jsPsych.data.addProperties(webgpu));

  // ---------------------------------------------------------------------------- content

  const screen = (title, body, button) => ({
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="ck-card"><h1>${title}</h1>${body}</div>`,
    choices: [button],
    button_html: (choice) => `<button class="ck-btn">${choice}</button>`,
  });

  /** Whether this session has the moving-target block; `pursuit.repeats: 0` leaves it out. */
  const hasPursuit = config.pursuit.runs.length > 0 && config.pursuit.repeats > 0;

  /** How many blocks the participant will be asked to do, counting the moving-target one. */
  const totalBlocks = () => config.design.n_blocks + (hasPursuit ? 1 : 0);

  /**
   * A screen advanced with the space bar rather than a button.
   *
   * Used for everything that sits *between* blocks, so the hand never leaves the space bar: the
   * task already reports the color change with it, and making people find a mouse between every
   * block invites them to shift position, which is the one thing the calibration cannot absorb.
   *
   * `title` and `body` may be functions, because the between-block screens report counts that do
   * not exist until the block has run.
   */
  const keyboardScreen = (title, body) => ({
    type: jsPsychHtmlKeyboardResponse,
    choices: [" "],
    stimulus: () => `
      <div class="ck-card">
        <h1>${typeof title === "function" ? title() : title}</h1>
        ${typeof body === "function" ? body() : body}
        <p class="ck-advance">Press <kbd>space</kbd> to continue.</p>
      </div>`,
  });

  // ---------------------------------------------------------------------------- Prolific

  const prolificUrl = (code) => `https://app.prolific.com/submissions/complete?cc=${code}`;

  /**
   * End the study for someone who cannot do it, and send them back to Prolific.
   *
   * Called from an `on_finish`, which jsPsych does not wait for, so the experiment is aborted
   * synchronously and the save runs afterwards, behind the message. The way back to Prolific
   * appears once the save has finished, so a quick click cannot navigate away mid-upload.
   *
   * What gets saved is only the screening record (browser, hardware, which check failed), under
   * its own prefix. It is how many people each check stops gets counted. A failed save is not
   * worth holding anyone for, so it is tried once and not turned into a download.
   */
  function screenOut(reason, heading, explanation) {
    jsPsych.data.addProperties({ screened_out: reason });
    Cookoff.tracker.dispose(); // the camera light should go off with the study
    jsPsych.abortExperiment(`<div class="ck-card"><h1>${heading}</h1>${explanation}
      <div id="ck-screenout-next"><p class="ck-status">Saving…</p></div></div>`);

    const code = config.prolific.screenout_code;
    const next = code
      ? `<p><a class="ck-btn" href="${prolificUrl(code)}">Return to Prolific</a></p>`
      : `<p>Please return the study on Prolific. There is nothing else you need to do.</p>`;
    CookoffSave.saveSession(jsPsych, { ...config.save, attempts: 1 }, { prefix: pilot ? "pilot_screenout" : "screenout" })
      .finally(() => {
        const slot = document.getElementById("ck-screenout-next");
        if (slot) slot.innerHTML = next;
      });
  }

  const welcome = screen(
    "Eye tracking study",
    `<p>In this study, targets will appear on the screen and we will measure where you look.
     It takes about ${config.intro_minutes} minutes.</p>
     <p>Your webcam stays on throughout. No video is recorded or sent — only numerical
     measurements taken from each frame.</p>
     <p>Before you begin, please find a comfortable position you can hold for a while. During
     the study you will only need the space bar, so you can leave the mouse alone.</p>`,
    "Begin",
  );

  /**
   * A record of the machine, not a filter.
   *
   * Excludes nobody: the point is to be able to read a result against the hardware it came
   * from. `vsync_rate` is the one worth having beyond the obvious — a 120 Hz display presents a
   * frame every 8 ms rather than every 17, which is part of the display lag the timing
   * measurement picks up.
   */
  const browserCheck = {
    type: jsPsychBrowserCheck,
    features: ["width", "height", "browser", "browser_version", "os", "mobile", "webcam",
               "vsync_rate"],
    skip_features: ["microphone", "webaudio"],
    on_finish: (data) => {
      jsPsych.data.addProperties({
        browser: data.browser,
        browser_version: data.browser_version,
        os: data.os,
        is_mobile_ua: data.mobile,
        has_webcam: data.webcam,
        vsync_rate: data.vsync_rate,
        window_width: data.width,
        window_height: data.height,
      });
    },
  };

  /**
   * A behavioral check that there is a keyboard, run before the 20 MB model downloads.
   *
   * The task reports the color change with a key press, so a participant who cannot press one
   * cannot produce the attention data that says whether they were looking at the target. A
   * phone or tablet also has no stable viewing distance and a front camera at an angle the
   * calibration never sees.
   *
   * Deliberately a key press rather than a user-agent test. The string can lie in both
   * directions — a tablet with a keyboard is perfectly usable, a desktop browser in a mobile
   * emulation mode is not a phone — and this asks the question the task actually cares about.
   * `is_mobile_ua` is recorded next to it so the two can be compared later.
   */
  const keyboardGate = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<div class="ck-card">
      <h1>A quick check</h1>
      <p>This study uses a physical keyboard, so it can't be done on a phone or tablet.</p>
      <p><strong>Press the space bar to continue.</strong></p>
    </div>`,
    choices: [" "],
    trial_duration: config.screening.keyboard_timeout,
    data: { screening: "keyboard_gate" },
    on_finish: (data) => {
      data.keyboard_present = data.rt !== null;
      jsPsych.data.addProperties({ keyboard_present: data.keyboard_present });
      if (!data.keyboard_present) {
        return screenOut(
          "no_keyboard",
          "Sorry — this study needs a keyboard",
          `<p>We didn't detect a key press, so this device may not be able to run the study.</p>
           <p>If you do have a keyboard and just missed the prompt, reload the page to try
           again.</p>`,
        );
      }
    },
  };

  /**
   * WebGL, checked before the camera or the 32 MB of models are touched.
   *
   * The face tracker cannot run without it, even on its CPU path, so a browser with no WebGL
   * would download everything, sit on "Looking for your face…" until the face timeout, and be
   * told its face could not be found. This stops that participant at once, with the real reason.
   * A trial of zero duration: nothing is shown unless the check fails.
   */
  const webglCheck = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "",
    choices: "NO_KEYS",
    trial_duration: 0,
    data: { screening: "webgl" },
    on_finish: (data) => {
      const webgl = CookoffCapability.probeWebGL();
      Object.assign(data, webgl);
      jsPsych.data.addProperties(webgl);
      if (!webgl.webgl_version) {
        return screenOut(
          "no_webgl",
          "Sorry — this computer can't run the study",
          `<p>The eye tracker needs your browser's graphics features (WebGL), and they are turned
           off or unavailable here.</p>
           <p>This can happen when hardware acceleration is switched off in the browser's
           settings.</p>`,
        );
      }
    },
  };

  /*
   * Camera illustrations.
   *
   * Drawn here rather than photographed: line art survives both themes, needs no network and no
   * licence, and the distinction being asked about — where the camera sits relative to the
   * screen — is clearer as a diagram than in a photo of somebody's desk.
   */
  const CAMERA_ART = {
    integrated: `
      <svg viewBox="0 0 120 92" width="120" height="92" aria-hidden="true">
        <rect x="14" y="10" width="92" height="58" rx="4" fill="none"
              stroke="currentColor" stroke-width="2.5"/>
        <rect x="20" y="16" width="80" height="46" rx="2" fill="currentColor" opacity="0.12"/>
        <circle cx="60" cy="14" r="2.6" fill="var(--ck-brass)"/>
        <path d="M4 82 h112 l-8 -14 H12 Z" fill="none" stroke="currentColor" stroke-width="2.5"
              stroke-linejoin="round"/>
      </svg>`,
    external: `
      <svg viewBox="0 0 120 92" width="120" height="92" aria-hidden="true">
        <rect x="14" y="20" width="92" height="52" rx="4" fill="none"
              stroke="currentColor" stroke-width="2.5"/>
        <rect x="20" y="26" width="80" height="40" rx="2" fill="currentColor" opacity="0.12"/>
        <path d="M52 20 h16 v-6 h-16 Z" fill="none" stroke="var(--ck-brass)" stroke-width="2.5"
              stroke-linejoin="round"/>
        <rect x="48" y="4" width="24" height="11" rx="3" fill="none" stroke="var(--ck-brass)"
              stroke-width="2.5"/>
        <circle cx="60" cy="9.5" r="2.6" fill="var(--ck-brass)"/>
        <path d="M46 84 h28 M60 72 v12" fill="none" stroke="currentColor" stroke-width="2.5"
              stroke-linecap="round"/>
      </svg>`,
  };

  /**
   * Which camera, and where it sits.
   *
   * Camera position relative to the screen is the largest thing that varies between setups and
   * it cannot be recovered from the video: a camera built into a laptop lid is a known geometry
   * above the display, and a separate one could be anywhere. It plausibly shows up in the
   * accuracy comparison, and asking costs one screen.
   */
  const cameraQuestion = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="ck-card">
      <h1>Your camera</h1>
      <p>Which of these is closest to the camera you are using right now?</p>
    </div>`,
    choices: ["built-in", "separate", "unsure"],
    button_html: (choice) => {
      const art = { "built-in": CAMERA_ART.integrated, separate: CAMERA_ART.external, unsure: "" };
      const label = {
        "built-in": "Built into my laptop or monitor",
        separate: "A separate camera, on or near the screen",
        unsure: "I'm not sure",
      };
      return `<button class="ck-choice${choice === "unsure" ? " ck-choice-plain" : ""}">
        ${art[choice]}<span>${label[choice]}</span></button>`;
    },
    data: { screening: "camera_type" },
    on_finish: (data) => {
      const answer = ["built-in", "separate", "unsure"][data.response];
      data.camera_type = answer;
      jsPsych.data.addProperties({ camera_type: answer });
    },
  };

  /**
   * Fullscreen, entered before anything measures the viewport.
   *
   * Target positions are fractions of `window.innerWidth` and both fits are tied to them, so a
   * viewport that changes size part-way through moves the screen out from under a calibration
   * taken before it. This pins it for the session.
   *
   * The browser only grants fullscreen from a user gesture, which is why it is a trial with a
   * button rather than something the page does on load.
   */
  const enterFullscreen = {
    type: jsPsychFullscreen,
    fullscreen_mode: true,
    message: `<div class="ck-card"><h1>Fullscreen</h1>
      <p>The study runs in fullscreen, so the screen stays the same size from start to
      finish.</p>
      <p>Please stay in fullscreen until the end. If you leave it by accident, press
      <kbd>F11</kbd> (or <kbd>&#8963;&#8984;F</kbd> on a Mac) to go back.</p></div>`,
    button_label: "Enter fullscreen",
    delay_after: 500, // let the window finish resizing before anything reads innerWidth
  };

  const exitFullscreen = { type: jsPsychFullscreen, fullscreen_mode: false, delay_after: 0 };

  /**
   * Camera permission, asked for on its own before anything loads.
   *
   * The preview trial asks too, but when permission is refused or the camera is busy it stops
   * on an error with no way forward. Here a refusal can be retried, and a camera that never
   * works screens the participant out instead of stranding them.
   */
  const cameraPermission = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="ck-card"><h1>Camera</h1>
      <p>When you continue, your browser will ask for permission to use your camera. Please
      choose <strong>Allow</strong>.</p>
      <p class="ck-status" id="ck-camera-status"></p></div>`,
    choices: ["Continue"],
    button_html: (choice) => `<button class="ck-btn" id="ck-camera-go">${choice}</button>`,
    data: { screening: "camera_permission" },
    on_load: () => {
      const button = document.getElementById("ck-camera-go");
      const status = document.getElementById("ck-camera-status");
      let failures = 0;
      // The plugin's own click handler would end the trial before the camera answers, so the
      // button is taken over: capture phase, and the trial ends from here.
      button.addEventListener(
        "click",
        async (event) => {
          event.stopImmediatePropagation();
          button.disabled = true;
          status.textContent = "Waiting for the camera…";
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach((track) => track.stop());
            jsPsych.finishTrial({ camera_ok: true, camera_failures: failures });
          } catch (err) {
            failures++;
            const name = err && err.name;
            if (failures >= 3) {
              jsPsych.finishTrial({ camera_ok: false, camera_failures: failures, camera_error: name });
              return;
            }
            status.innerHTML =
              name === "NotAllowedError"
                ? `The camera was blocked. Click the camera icon in the address bar, choose
                   <strong>Allow</strong>, then try again.`
                : `The camera could not start (${name}). Close any other app that might be using
                   it, then try again.`;
            button.textContent = "Try again";
            button.disabled = false;
          }
        },
        { capture: true },
      );
    },
    on_finish: (data) => {
      if (!data.camera_ok) {
        return screenOut(
          "no_camera",
          "Sorry — the camera did not start",
          `<p>The study needs a working camera, and we couldn't start yours.</p>`,
        );
      }
    },
  };

  const preview = {
    type: jsPsychSaccadePreview,
    // After this long the continue button unlocks even with no face, and `on_finish` screens
    // the participant out: the recorder cannot collect anything from a camera that never
    // finds a face, and without a limit that participant is stuck on this screen.
    face_timeout: config.prolific.face_timeout,
    on_finish: (data) => {
      if (!data.face_detected) {
        return screenOut(
          "no_face",
          "Sorry — the camera could not find your face",
          `<p>The eye tracker needs a clear view of your face, and it couldn't find one.</p>
           <p>Light from in front of you often helps. If you'd like to try again, reload the
           page.</p>`,
        );
      }
    },
    instructions: `
      <h2>Camera setup</h2>
      <p>Sit facing the screen at your usual distance. Light from in front of you works best;
      a bright window behind you makes your face hard to see.</p>
      <p>When the label under the video says <strong>Face found</strong>, you're set. Try to
      stay in about that position for the rest of the study.</p>`,
    button_text: "Continue",
  };

  /**
   * What saccade.js would run at if it were tracking live.
   *
   * This experiment never predicts gaze, so nothing else in the session answers the obvious
   * question about a participant's machine: could saccade.js have tracked them in real time,
   * and at what rate? `saccade-performance` is saccade.js's own answer — it runs the real
   * tracker, with the real model, on this camera, and reports the frame rate it achieved.
   *
   * Two reasons not to substitute the recorder's own rate for this. The recorder does strictly
   * more work per frame (it also cuts WebGazer's patches), so it runs slower than live
   * tracking would; and on a machine without WebGPU it moves the encoder to a worker, which
   * live tracking cannot do. Both numbers are recorded, and the difference between them is the
   * price of carrying the second tracker.
   *
   * Placed before the fifteen-second timing measurement so that an exclusion rule, if one is
   * ever added through `inclusion_function`, fires before the participant has spent that time.
   * There is no rule here: the whole point of the deferred encoder is to keep slow machines in
   * the sample rather than drop them.
   */
  const performance_check = {
    type: jsPsychSaccadePerformance,
    measurement_duration: 5000,
    warmup_duration: 1000,
    stimulus: `
      <div class="ck-card">
        <h1>Checking your setup</h1>
        <p>Please look at the dot and hold still for a few seconds.</p>
        <div class="ck-center-dot"></div>
      </div>`,
    on_finish: (data) => {
      // Recorded on every row as well as its own: this is one fact about the machine, and an
      // analysis that filters by frame rate should not have to join across trials to get it.
      jsPsych.data.addProperties({
        saccade_live_fps_median: data.fps_median,
        saccade_live_fps_mean: data.fps_mean,
        saccade_live_fps_p10: data.fps_p10,
        saccade_live_embed_ms_median: data.embed_ms_median,
        saccade_live_frames: data.frames,
        saccade_live_frames_without_face: data.frames_without_face,
        saccade_live_dropped: data.dropped,
        saccade_live_backend: data.backend,
      });
    },
  };

  const timeSync = {
    type: jsPsychSaccadeTimeSync,
    duration: config.time_sync.duration,
    contrast: config.time_sync.contrast,
    apply_offset: true,
    instructions: `
      <h2>Camera timing check</h2>
      <p>Next we'll measure how quickly your webcam responds to changes on the screen. This
      differs from computer to computer, so we measure it on yours.</p>
      <p>For about fifteen seconds, the whole screen will switch between dark and light at
      random moments while the camera watches.</p>
      <p><strong>This is not a flashing display.</strong> The screen changes at most twice a
      second.</p>
      <p>Please sit still and keep looking at the screen until it finishes.</p>`,
    button_text: "Continue",
    on_finish: (data) => {
      // Recorded once at the top level too: every frame time in the session is on the camera
      // clock, and this is the constant that puts it on the same clock as the display.
      jsPsych.data.addProperties({
        timing_lag_ms: data.lag_ms,
        timing_verdict: data.verdict,
        timing_plateau_width_ms: data.plateau_width_ms,
        timing_camera_period_ms: data.camera_period_ms,
      });
    },
  };

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

  /**
   * The camera's delivered frame rate, from the video element's own frame counter.
   *
   * Independent of any frame loop: `presentedFrames` and `mediaTime` count what the camera
   * delivered, so a page that stalls for a moment does not read as a slow camera. Null when the
   * browser has no `requestVideoFrameCallback` (Firefox before 132), or too few frames arrived.
   */
  function measureCameraFps(video, durationMs) {
    if (!video || !video.requestVideoFrameCallback) return Promise.resolve(null);
    return new Promise((resolve) => {
      let first = null;
      let last = null;
      let finished = false;
      const onFrame = (now, meta) => {
        if (finished) return;
        if (!first) first = meta;
        last = meta;
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
      setTimeout(() => {
        finished = true;
        const frames = first && last ? last.presentedFrames - first.presentedFrames : 0;
        const seconds = first && last ? last.mediaTime - first.mediaTime : 0;
        resolve(frames >= 3 && seconds > 0 ? frames / seconds : null);
      }, durationMs);
    });
  }

  /** Camera settings worth keeping. Never `deviceId` or `groupId`: those identify the device. */
  const CAMERA_SETTING_KEYS = [
    "width", "height", "frameRate", "aspectRatio", "resizeMode", "facingMode",
    "exposureMode", "exposureTime", "exposureCompensation", "iso",
    "focusMode", "focusDistance", "whiteBalanceMode", "colorTemperature",
    "brightness", "contrast", "saturation", "sharpness", "backgroundBlur",
  ];

  /**
   * What the browser says the camera is doing, and what it could do.
   *
   * The frame rate here is the one the camera was configured for, which can differ from what it
   * delivers (`camera_fps`): many webcams halve their rate in dim light to lengthen the
   * exposure, and a long exposure blurs a moving eye. Exposure and focus modes are reported by
   * Chrome for many cameras and by other browsers rarely; absent keys are simply not recorded.
   * `backgroundBlur` is the operating system's camera effect, which would soften the image
   * before either tracker sees it.
   */
  function cameraSettings(video) {
    const stream = video && video.srcObject;
    const track = stream && stream.getVideoTracks ? stream.getVideoTracks()[0] : null;
    if (!track) return { camera_settings: null, camera_capabilities: null };
    const pick = (source) => {
      if (!source) return null;
      const out = {};
      for (const key of CAMERA_SETTING_KEYS) if (source[key] !== undefined) out[key] = source[key];
      return out;
    };
    let capabilities = null;
    try {
      capabilities = track.getCapabilities ? pick(track.getCapabilities()) : null;
    } catch (err) {
      capabilities = null;
    }
    return { camera_settings: pick(track.getSettings()), camera_capabilities: capabilities };
  }

  /**
   * The longest encode that can run inline on this machine.
   *
   * An inline encode has to keep up with the camera, so the limit is a share of one camera
   * frame, not a fixed number: 85% of 67 ms on a 15 fps webcam, which is most of them, and
   * 85% of 33 ms at 30 fps. The share is high because on WebGPU the encode runs on the GPU,
   * pipelined against the frame loop's CPU work. The first pilot bears it out: five sessions
   * that encoded in 30-54 ms were deferred under a flat 30 ms limit, yet saccade.js's own live
   * tracker ran at the camera's full 15.6 fps on every one of them.
   *
   * On wasm the encode runs on the main thread instead, where a slow one delays everything else
   * on the page, key presses included; that keeps the flat limit. So does an unknown camera rate.
   */
  function inlineLimitMs(provider, cameraFps) {
    const settings = config.encoder;
    if (provider !== "webgpu" || !cameraFps) return settings.inline_max_ms;
    return Math.min(
      settings.inline_ceiling_ms,
      Math.max(settings.inline_max_ms, (settings.inline_frame_share * 1000) / cameraFps),
    );
  }

  /**
   * Decide where the eye encoder runs, and how much of it there is room for.
   *
   * Returns the numbers to record, so a session's frame budget is visible in the data rather
   * than inferred from the user agent afterwards.
   */
  async function chooseEncoder(recorder, status) {
    const settings = config.encoder;
    const provider = Cookoff.model.provider;

    /*
     * Stop the tracker's frame loop before timing anything.
     *
     * The preview trial left it running, and it calls `model.embed` on every camera frame.
     * `OrtEmbeddingModel` fills one reusable input tensor and runs one session, so a
     * measurement taken alongside it would be two concurrent inferences through one buffer:
     * a wrong number, and corrupted frames on the way to getting it. The pause afterwards is
     * for the inference that was already in flight when the loop stopped.
     */
    Cookoff.tracker.stop();
    await new Promise((resolve) => setTimeout(resolve, 250));

    // How fast the camera delivers frames, measured on the video itself while nothing else runs.
    // This is the budget an inline encode has to fit into.
    const cameraFps = await measureCameraFps(recorder.tracker.video, CAMERA_MEASURE_MS);

    // The inline path is already pipelined against the frame loop; time it the same way
    // saccade.js's own tracker would experience it. The median of up to `max_samples`, within a
    // time cap: five samples averaged let one slow inference decide the path, and in the first
    // pilot two sessions measured 30.0 and 30.2 ms against a 30 ms threshold.
    const crop = new Uint8Array(144 * 36);
    for (let i = 0; i < crop.length; i++) crop[i] = (i * 37) % 256;
    for (let i = 0; i < 5; i++) await Cookoff.model.embed(crop); // WebGPU compiles shaders lazily
    const samples = [];
    const started = performance.now();
    while (
      samples.length < settings.min_samples ||
      (samples.length < settings.max_samples &&
        performance.now() - started < settings.max_sampling_ms)
    ) {
      const t0 = performance.now();
      await Cookoff.model.embed(crop);
      samples.push(performance.now() - t0);
    }
    const inlineMs = median(samples);
    const limitMs = inlineLimitMs(provider, cameraFps);

    const wantsDeferred =
      settings.mode === "deferred" || (settings.mode === "auto" && inlineMs > limitMs);

    const common = {
      camera_fps: cameraFps == null ? null : Math.round(cameraFps * 10) / 10,
      encoder_inline_ms: Math.round(inlineMs * 10) / 10,
      encoder_inline_samples: samples.length,
      encoder_inline_limit_ms: Math.round(limitMs * 10) / 10,
    };

    if (!wantsDeferred) {
      return {
        ...common,
        encoder_mode: "inline",
        encoder_provider: provider,
        encoder_budget_per_station: null,
      };
    }

    status.textContent = "Your computer needs a little longer to get ready. This can take a minute…";
    const embedder = new DeferredEmbedder({
      workerUrl: "js/embed-worker.js",
      modelUrl: new URL(config.assets.modelUrl, window.location.href).href,
      ortUrl: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${Saccade.ORT_VERSION}/dist/ort.bundle.min.mjs`,
      wasmDir: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${Saccade.ORT_VERSION}/dist/`,
    });

    try {
      await embedder.init();
      const workerMs = await embedder.measure(3);
      const stationMs =
        config.design.settle_duration + config.design.capture_duration + config.design.station_gap;
      const budget = embedder.setBudget(
        stationMs,
        settings.worker_share,
        settings.min_frames_per_station,
        Number.MAX_SAFE_INTEGER,
      );
      recorder.embedder = embedder;
      Cookoff.embedder = embedder;
      return {
        ...common,
        encoder_mode: "deferred",
        encoder_worker_ms: Math.round(workerMs * 10) / 10,
        encoder_provider: "wasm-worker",
        encoder_budget_per_station: budget,
      };
    } catch (error) {
      // A worker that will not start is not a reason to lose the participant: fall back to
      // encoding inline, which is slow but correct, and say so in the data.
      console.error("cook-off: the embed worker would not start; staying inline.", error);
      embedder.dispose();
      return {
        ...common,
        encoder_mode: "inline-fallback",
        encoder_provider: provider,
        encoder_budget_per_station: null,
        encoder_worker_error: String((error && error.message) || error),
      };
    }
  }

  /**
   * A screen that waits for the encoder's queue to empty.
   *
   * Only ever shown when the encoder was deferred and is behind. The deadline is what keeps it
   * from being an open-ended wait: whatever is still queued when it expires has its raw crops
   * written into the data instead, which costs bytes rather than frames.
   */
  function drainScreen(deadlineMs, heading) {
    return {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
        <div class="ck-card">
          <h1>${heading}</h1>
          <p class="ck-status" id="ck-drain-status">Finishing up…</p>
          <div class="ck-bar"><div id="ck-drain-fill"></div></div>
        </div>`,
      choices: "NO_KEYS",
      on_load: async () => {
        const embedder = Cookoff.embedder;
        if (!embedder || embedder.pending() === 0) return jsPsych.finishTrial({ drained: 0 });

        const status = document.getElementById("ck-drain-status");
        const fill = document.getElementById("ck-drain-fill");
        const result = await embedder.drain(deadlineMs, (left, total) => {
          if (fill) fill.style.width = `${Math.round((1 - left / Math.max(1, total)) * 100)}%`;
          if (status) status.textContent = `Finishing up… (${left} left)`;
        });
        jsPsych.finishTrial({ drained: result.left === 0, left_queued: result.left });
      },
    };
  }

  /**
   * Load the recorder's own face mesh and run it for a moment on a live camera.
   *
   * This is the machine check. The recorder does strictly more work per frame than saccade.js's
   * tracker does — one face mesh, then two sets of features off it — so the tracker's own
   * performance trial would report a frame rate this session never sees. Measuring the thing
   * that will actually run is the only honest version of that check.
   */
  const warmUp = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div class="ck-card">
        <h1>Getting ready</h1>
        <p class="ck-status" id="ck-warmup-status">Starting up. This takes a few seconds…</p>
      </div>`,
    choices: "NO_KEYS",
    trial_duration: null,
    on_load: async () => {
      const status = document.getElementById("ck-warmup-status");
      try {
        const extension = jsPsych.extensions.saccade;
        Cookoff.extension = extension;

        // Idempotent, and already done by the preview trial. Here so that dropping the preview
        // from the timeline — to re-run a participant who has already granted the camera, say —
        // does not leave the recorder with no camera and no encoder.
        await extension.start();
        extension.hideVideo();

        const recorder = new DualRecorder({
          tracker: Cookoff.tracker,
          model: Cookoff.model,
          assets: config.assets,
        });
        status.textContent = "Loading…";
        await recorder.init();

        /*
         * Where the encoder runs is decided here, from what it actually costs on this machine.
         *
         * The provider alone is not the question. WebGPU almost always means fast enough to
         * run inline, but "wasm" covers everything from tolerable to hopeless, and the
         * difference decides whether a station yields thirty embeddings or four. So the cost
         * is measured — on the real model, after a warm-up inference — and the arithmetic is
         * done against the station cycle.
         */
        const measured = await chooseEncoder(recorder, status);

        /*
         * The other half of the per-frame cost, for the record.
         *
         * Everything above is about the encoder because the encoder is where the time goes —
         * but "the ridge is free" is an assumption, and an assumption in a methods section
         * should be a number. This measures it: a dot product of the embedding with a 2 x d
         * kernel, which is what saccade.js does per frame to turn an embedding into a gaze
         * point. It comes in around a microsecond, against tens of milliseconds for the
         * encoder on WebGPU and hundreds on wasm.
         */
        const ridgeUs = CookoffCapability.measureRidgePredict(recorder.embeddingDim);

        status.textContent = "Almost ready…";
        Cookoff.recorder = recorder;
        recorder.start();
        await new Promise((resolve) => setTimeout(resolve, WARMUP_MS));

        jsPsych.data.addProperties({
          ...measured,
          ...recorder.describe(),
          ridge_predict_us: Math.round(ridgeUs * 1000) / 1000,
          ...cameraSettings(recorder.tracker.video),
          warmup_fps: Math.round(recorder.fps * 10) / 10,
          warmup_frames: recorder.stats.frames,
          warmup_face_missing: recorder.stats.faceMissing,
          warmup_camera_dropped: recorder.stats.dropped,
        });

        status.textContent = "Ready.";
        setTimeout(() => jsPsych.finishTrial({}), 700);
      } catch (error) {
        // Without a recorder every station afterwards would throw one at a time, leaving the
        // participant clicking through a broken experiment. Stop here and say so.
        console.error(error);
        status.innerHTML = `<span class="ck-ember">Something went wrong while getting
          ready.</span><br><br>
          Please check that the camera is allowed, then reload the page. If this keeps happening,
          please send the researcher a message through Prolific.<br><br>
          <small>${error && error.message ? error.message : error}</small>`;
        jsPsych.data.addProperties({ warmup_error: String(error && error.message) });
      }
    },
  };

  const howToPlay = screen(
    "What to do",
    `<p>A target will appear on the screen. <strong>Look at its center</strong> and keep your
     eyes there. When it jumps to a new place, follow it with your eyes.</p>
     <p>Now and then, the small dot in the center will turn <span class="ck-ember">orange</span>.
     When it does, press <kbd>space</kbd>.</p>
     <p>Please keep your head as still as you can, and move only your eyes.</p>
     <p>You'll start with a short practice round.</p>`,
    "Start practice",
  );

  // ---------------------------------------------------------------------------- blocks

  /**
   * The calibration grid, nested: 5x5 at 5/27.5/50/72.5/95%, each point labeled with the
   * smallest of saccade.js's grids it belongs to. `maxLevel` 13 gives saccade.js's default
   * grid, 9 its 3x3.
   *
   * Checked against saccade.js's own 13-point grid rather than trusted, because the analysis
   * reports "a 13-point calibration" on the strength of this labeling.
   */
  function calibrationGrid(maxLevel) {
    const outer = [0.05, 0.5, 0.95];
    const inner = [0.275, 0.725];
    const steps = [0.05, 0.275, 0.5, 0.725, 0.95];
    const grid = [];
    for (const y of steps) {
      for (const x of steps) {
        const level =
          outer.includes(x) && outer.includes(y) ? 9 : inner.includes(x) && inner.includes(y) ? 13 : 25;
        grid.push({ x, y, set: "grid", grid_level: level });
      }
    }
    const same = (a, b) => Math.abs(a.x - b.x) < 1e-9 && Math.abs(a.y - b.y) < 1e-9;
    const g13 = window.Saccade.defaultGrid13();
    const ours = grid.filter((p) => p.grid_level <= 13);
    if (ours.length !== g13.length || !g13.every((p) => ours.some((q) => same(p, q)))) {
      throw new Error("calibrationGrid: the 13-point subset no longer matches saccade.js's defaultGrid13");
    }
    return grid.filter((p) => p.grid_level <= maxLevel);
  }

  /**
   * Fresh test positions for one block: one per cell of a 3x3 over the screen, cycling through
   * the cells in random order if more than nine are asked for, and resampled while too close to
   * any fixed point.
   */
  function randomPoints(n, fixed) {
    const { random_margin: m, random_min_distance: minD } = config.points;
    const cell = (1 - 2 * m) / 3;
    const cells = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cells.push([i, j]);
    const order = [];
    while (order.length < n) order.push(...jsPsych.randomization.shuffle(cells));

    const out = [];
    for (const [i, j] of order.slice(0, n)) {
      let p;
      for (let attempt = 0; attempt < 200; attempt++) {
        p = { x: m + (i + Math.random()) * cell, y: m + (j + Math.random()) * cell };
        if ([...fixed, ...out].every((q) => Math.hypot(p.x - q.x, p.y - q.y) >= minD)) break;
      }
      out.push({ x: p.x, y: p.y, set: "random", grid_level: null });
    }
    return out;
  }

  const grid = calibrationGrid(config.points.grid);
  const validation = config.points.validation
    ? window.Saccade.validationGrid9().map((p) => ({ ...p, set: "validation", grid_level: null }))
    : [];

  /** One block's targets, in presentation order: the grid, then the test points. */
  function blockPoints() {
    const test = [...validation, ...randomPoints(config.points.n_random, [...grid, ...validation])];
    return config.points.grid_first
      ? [...jsPsych.randomization.shuffle(grid), ...jsPsych.randomization.shuffle(test)]
      : jsPsych.randomization.shuffle([...grid, ...test]);
  }

  /**
   * One block's stations, in the order given, with a random subset probed.
   *
   * Which points are probed is drawn without replacement rather than per-point at random, so
   * every participant gets the same number of probes and the attention measure has the same
   * precision for all of them.
   */
  function buildStations(order, rep, probeRate) {
    const nProbes = Math.round(order.length * probeRate);
    const probes = jsPsych.randomization
      .shuffle(order.map((_, i) => i))
      .slice(0, nProbes)
      .reduce((set, i) => set.add(i), new Set());

    return order.map((point, i) => ({
      x: point.x,
      y: point.y,
      point_set: point.set,
      grid_level: point.grid_level,
      rep,
      block: rep,
      probe: probes.has(i),
      index_in_block: i,
      block_length: order.length,
    }));
  }

  const station = {
    type: jsPsychCookoffCapture,
    x: jsPsych.timelineVariable("x"),
    y: jsPsych.timelineVariable("y"),
    probe: jsPsych.timelineVariable("probe"),
    point_set: jsPsych.timelineVariable("point_set"),
    rep: jsPsych.timelineVariable("rep"),
    block: jsPsych.timelineVariable("block"),
    index_in_block: jsPsych.timelineVariable("index_in_block"),
    block_length: jsPsych.timelineVariable("block_length"),
    settle_duration: config.design.settle_duration,
    record_settle: config.design.record_settle,
    capture_duration: config.design.capture_duration,
    probe_window: config.design.probe_window,
    response_deadline: config.design.response_deadline,
    response_key: config.design.response_key,
    post_trial_gap: config.design.station_gap,
    // 9, 13 or 25 for a grid point: the smallest of saccade.js's grids it belongs to. Null for a
    // test point. The analysis builds an n-point calibration from `grid_level <= n`.
    data: { grid_level: jsPsych.timelineVariable("grid_level") },
  };

  const practice = {
    timeline: [station],
    timeline_variables: buildStations(
      jsPsych.randomization.shuffle(grid).slice(0, config.design.n_practice),
      -1,
      0.5,
    ),
    data: { practice: true },
  };

  // Drawn now rather than when each block starts, so the block intro can say how long it is.
  const blockStations = Array.from({ length: config.design.n_blocks }, (_, block) =>
    buildStations(blockPoints(), block, config.design.probe_rate),
  );

  const staticBlock = (block) => ({
    timeline: [station],
    timeline_variables: blockStations[block],
    data: { practice: false },
  });

  /**
   * The moving-target block: the only trials where the timing measurement changes an answer.
   *
   * Placed after the static blocks so the calibration data is already collected from a
   * stationary target — which is how both trackers are meant to be calibrated — and the moving
   * target is purely a test set.
   */
  function movingTargetBlock() {
    const settings = config.pursuit;
    const runs = [];
    for (let repeat = 0; repeat < settings.repeats; repeat++) {
      for (const [index, run] of settings.runs.entries()) {
        runs.push({ ...run, repeat, index: repeat * settings.runs.length + index });
      }
    }
    const probes = jsPsych.randomization
      .shuffle(runs.map((_, i) => i))
      .slice(0, Math.round(runs.length * settings.probe_rate))
      .reduce((set, i) => set.add(i), new Set());

    return runs.map((run) => ({
      type: jsPsychCookoffPursuit,
      speed: run.speed,
      cycles: run.cycles,
      condition: run.condition,
      margin: settings.margin,
      margin_y: settings.margin_y,
      ramp_duration: settings.ramp_duration,
      lead_in: settings.lead_in,
      // Direction alternates across runs. A real lag always trails the motion and so flips with
      // it; a spatial bias is fixed in screen coordinates and does not.
      direction: run.index % 2 === 0 ? 1 : -1,
      probe: probes.has(run.index),
      response_deadline: config.design.response_deadline,
      response_key: config.design.response_key,
      block: 100 + run.repeat,
      index_in_block: run.index,
      block_length: runs.length,
      post_trial_gap: 400,
      data: { practice: false, trial_kind: "pursuit" },
    }));
  }

  const movingTargetIntro = keyboardScreen(
    () => `Block ${config.design.n_blocks + 1} of ${totalBlocks()}`,
    `<p>This time, one target will <strong>move in a loop around the screen</strong>. Follow it
     with your eyes the whole way around.</p>
     <p>Try to follow it smoothly, staying with the target rather than looking ahead of it.</p>
     <p>As before, press <kbd>space</kbd> if its center turns
     <span class="ck-ember">orange</span>.</p>
     <p>This is the last block, and it takes about half a minute.</p>`,
  );

  /** Between blocks: how the color-change task went. Advanced with the space bar. */
  function blockReport(blockIndex) {
    return keyboardScreen(
      () => `Block ${blockIndex + 1} of ${totalBlocks()} complete`,
      () => {
        const rows = jsPsych.data
          .get()
          .filter({ trial_type: "cookoff-capture", block: blockIndex, practice: false });
        const probes = rows.filter({ probe: true });
        const hits = probes.filter({ probe_outcome: "hit" }).count();
        return `<p>You spotted <strong>${hits} of ${probes.count()}</strong> color changes.</p>
          <p>Please stay in the same position, and continue when you're ready for the next
          block.</p>`;
      },
    );
  }

  // ---------------------------------------------------------------------------- questions

  /** A required radio group, one option per line. */
  const radios = (name, legend, options) => `
    <fieldset class="ck-question">
      <legend>${legend}</legend>
      ${options
        .map(([value, label]) => `<label><input type="radio" name="${name}" value="${value}"
          required> ${label}</label>`)
        .join("")}
    </fieldset>`;

  /**
   * Asked at the end, once the eye tracking is over, so nothing here moves the participant
   * between calibration and test.
   *
   * Eyewear is the covariate: glasses put reflections and a second lens edge over the eyes, and
   * whether that costs accuracy has not been measured for saccade.js. It is asked here rather
   * than taken from Prolific's profile because what matters is this session, not the person's
   * habit. Demographics are not asked at all: Prolific supplies them, keyed on `PROLIFIC_PID`.
   */
  function endQuestions() {
    const q = config.questions;
    const parts = [];
    if (q.eyewear) {
      parts.push(radios("eyewear", "During this study, were you wearing glasses or contact lenses?", [
        ["glasses", "Glasses"],
        ["contacts", "Contact lenses"],
        ["neither", "Neither"],
        ["prefer_not", "Prefer not to say"],
      ]));
    }
    if (q.self_report) {
      // The recording shows *that* a session went badly; these are often the only thing that
      // says why. Technical and attentional causes are asked apart because they point to
      // different fixes: a frozen camera is the setup, someone walking in is the session.
      parts.push(`
        <fieldset class="ck-question">
          <legend>Did you encounter any technical problems? (optional)</legend>
          <textarea name="technical_problems" rows="3" maxlength="1000"
            aria-label="Technical problems"></textarea>
        </fieldset>
        <fieldset class="ck-question">
          <legend>Did anything interrupt your ability to keep your eyes on the targets?
            (optional)</legend>
          <textarea name="interruptions" rows="3" maxlength="1000"
            aria-label="Interruptions"></textarea>
        </fieldset>`);
    }
    if (!parts.length) return null;
    return {
      type: jsPsychSurveyHtmlForm,
      html: `<div class="ck-card ck-form"><h1>A few last questions</h1>${parts.join("")}</div>`,
      button_label: "Continue",
      data: { questionnaire: "end" },
      on_finish: (data) => {
        const r = data.response || {};
        jsPsych.data.addProperties({
          eyewear: r.eyewear ?? null,
          technical_problems: q.self_report ? (r.technical_problems || "").trim() || null : null,
          interruptions: q.self_report ? (r.interruptions || "").trim() || null : null,
        });
      },
    };
  }

  // ---------------------------------------------------------------------------- timeline

  const timeline = [welcome];
  // Screening first: nothing here needs the camera or the 20 MB model, so a participant who
  // cannot do the task finds out before either is downloaded.
  if (config.screening.browser_check) timeline.push(browserCheck);
  if (config.screening.keyboard_gate) timeline.push(keyboardGate);
  timeline.push(webglCheck);
  if (config.screening.camera_question) timeline.push(cameraQuestion);
  timeline.push(cameraPermission);
  if (config.fullscreen) timeline.push(enterFullscreen);
  timeline.push(preview);
  if (config.performance_check !== false) timeline.push(performance_check);
  timeline.push(timeSync, warmUp, howToPlay);
  // A practice block of zero targets is not a shorter practice block, it is a jsPsych node with
  // no timeline variables. Diagnostic builds set `n_practice: 0` to mean "skip it".
  if (config.design.n_practice > 0) timeline.push(practice);

  for (let block = 0; block < config.design.n_blocks; block++) {
    timeline.push({
      ...keyboardScreen(
        `Block ${block + 1} of ${totalBlocks()}`,
        `<p>This block has ${blockStations[block].length} targets. As before, look at the center
         of each one, and press <kbd>space</kbd> if it turns orange.</p>`,
      ),
      // Between blocks the eye was on this text, not on the last target, so the next station
      // has no saccade origin worth recording.
      on_finish: () => jsPsychCookoffCapture.forgetPreviousTarget(),
    });
    timeline.push(staticBlock(block));
    if (block < config.design.n_blocks - 1) {
      // The break is already a pause, so it is the cheapest place to let the worker catch up.
      // Skipped instantly when the queue is empty, which is every WebGPU machine.
      timeline.push(drainScreen(config.encoder.drain_ms_between_blocks, "One moment"));
      timeline.push(blockReport(block));
    }
  }

  if (hasPursuit) timeline.push(movingTargetIntro, ...movingTargetBlock());

  if (config.fullscreen) timeline.push(exitFullscreen);
  const questions = endQuestions();
  if (questions) timeline.push(questions);

  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<div class="ck-card"><h1>Saving</h1>
      <p class="ck-status" id="ck-save-status">Saving your data. Please don't close this
      tab…</p></div>`,
    choices: "NO_KEYS",
    on_load: async () => {
      Cookoff.recorder?.stop();

      // Everything the worker still owes, before anything is serialised. What it cannot finish
      // inside the deadline goes into the data as raw crops instead — twenty times the bytes,
      // the same information, and nothing lost to a slow machine.
      let embedSummary = null;
      if (Cookoff.embedder) {
        const status = document.getElementById("ck-save-status");
        const left = await Cookoff.embedder.drain(config.encoder.drain_ms_at_end, (n, total) => {
          if (status && n > 0) status.textContent = `Finishing up… (${n} of ${total} left)`;
        });
        embedSummary = Cookoff.embedder.finalize(jsPsych, config.privacy.save_eye_crops);
        embedSummary.timedOut = left.timedOut;
        Cookoff.embedder.dispose();
      }
      Cookoff.tracker.dispose();

      jsPsych.data.addProperties({
        frames_embedded: embedSummary ? embedSummary.embedded : null,
        frames_left_as_crops: embedSummary ? embedSummary.asCrops : null,
        frames_dropped_unencoded: embedSummary ? embedSummary.cropsDropped : null,
        eye_crops_saved: config.privacy.save_eye_crops,
        embed_queue_timed_out: embedSummary ? embedSummary.timedOut : null,
        recorder_frames: Cookoff.recorder?.stats.frames ?? null,
        recorder_face_missing: Cookoff.recorder?.stats.faceMissing ?? null,
        recorder_camera_dropped: Cookoff.recorder?.stats.dropped ?? null,
        recorder_errors: Cookoff.recorder?.stats.errors ?? null,
        interaction_events: Cookoff.interactions.length,
      });

      const status = document.getElementById("ck-save-status");
      status.textContent = "Uploading your data. Please don't close this tab…";
      const result = await CookoffSave.saveSession(jsPsych, config.save, {
        onRetry: (attempt) => {
          status.textContent = `The upload didn't go through. Trying again (attempt ${attempt + 1} of ${config.save.attempts})…`;
        },
      });
      const { json, ...summary } = result;

      if (!result.error) {
        status.textContent = "Saved.";
        setTimeout(() => jsPsych.finishTrial(summary), 600);
        return;
      }

      // The upload failed for good. The data is still in this tab, so hand it to the participant
      // rather than lose it: they can send the file through Prolific's messages. The session
      // still counts as complete -- the failure is ours, not theirs.
      status.innerHTML = `
        <span class="ck-ember">We couldn't upload your data.</span>
        <p>This is a problem on our end, not yours, and you will still be paid. Please save your
        data file with the button below, then send it to the researcher in a Prolific
        message.</p>
        <p><small>(${result.error})</small></p>
        <p><button class="ck-btn" id="ck-save-download">Save data file</button></p>
        <p><button class="ck-btn" id="ck-save-continue" disabled>Continue</button></p>`;
      const cont = document.getElementById("ck-save-continue");
      document.getElementById("ck-save-download").addEventListener("click", () => {
        CookoffSave.download(json, result.filename);
        cont.disabled = false;
      });
      cont.addEventListener("click", () => jsPsych.finishTrial({ ...summary, fallback_download: true }));
    },
  });

  /**
   * The end. With a completion code, the button takes the participant to Prolific to be
   * credited; without one (piloting), there is nowhere to send them.
   */
  const completion = config.prolific.completion_code;
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="ck-card">
        <h1>Finished</h1>
        <p>That's everything. Thank you for taking part!</p>
        <p>${completion ? "Click below to return to Prolific and complete the study."
                        : "You can close this tab."}</p>
      </div>`,
    choices: [completion ? "Return to Prolific" : "Close"],
    button_html: (choice) => `<button class="ck-btn">${choice}</button>`,
    on_finish: () => {
      if (completion) window.location.assign(prolificUrl(completion));
    },
  });

  jsPsych.run(timeline);
})(window);
