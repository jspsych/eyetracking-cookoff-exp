/**
 * One camera frame in, two trackers' inputs out.
 *
 * saccade.js and WebGazer both start from a MediaPipe face mesh and then cut the image
 * differently: saccade.js takes one 144x36 strip across both eyes, CLAHE-equalized, and pushes
 * it through a 20 MB ONNX encoder; WebGazer takes a 10x6 patch per eye and uses the pixels
 * themselves. Running two trackers means two face meshes, two canvas readbacks, and two
 * prediction stacks on a 30 fps budget — which is what makes a head-to-head comparison
 * unaffordable in the browser.
 *
 * So this recorder runs the expensive half once. Per frame: one `drawImage`, one
 * `getImageData`, one MediaPipe detection. From that one detection it cuts both trackers'
 * inputs, and it stops there. No ridge fit, no smoothing, no Kalman filter, no gaze estimate —
 * those are cheap, deterministic, and can be replayed offline from what is recorded here, as
 * many times and as many ways as an analysis needs.
 *
 * The ONNX encoder is the one piece that cannot simply be dropped, because its output *is*
 * saccade.js's feature vector and the alternative is 5 KB of raw crop per frame instead of 256
 * bytes of embedding. Where it runs depends on the machine:
 *
 *   - With WebGPU it costs about 20 ms and runs here, pipelined the way saccade.js's own
 *     tracker runs it: the CPU work for frame N+1 overlaps the encode of frame N, and exactly
 *     one inference is ever in flight, which is what the shared input tensor requires.
 *   - Without WebGPU it costs one to several hundred milliseconds on single-threaded wasm,
 *     which is longer than a camera frame. Running it here would not just be slow, it would
 *     cost frames. So it moves to a worker and this loop stops waiting for it entirely; see
 *     `deferred-embedder.js` for the queue, the per-station budget and the write-back.
 */
(function (global) {
  "use strict";

  const EYE_W = 144;
  const EYE_H = 36;
  const EMB_STRIDE = 128; // the shipped model's width; the recorder reads the real one at init

  /**
   * Landmarks kept per frame, as a compact stand-in for head pose: nose tip, forehead, chin,
   * both cheek edges, and the four eye corners. Head position and distance are the obvious
   * confound in a webcam comparison, and 27 numbers a frame is cheap enough to never regret.
   */
  const HEAD_LANDMARKS = [1, 10, 152, 234, 454, 33, 133, 362, 263];

  /**
   * The head pose per frame: MediaPipe's facial transformation matrix, a 4x4 rigid transform
   * from its canonical face model to this face, stored column-major (translation in elements
   * 12-14, in MediaPipe's model units, about cm). `Saccade.headPoseFromMatrix` turns it into
   * yaw, pitch and roll. A frame with no matrix is stored as NaN.
   */
  const POSE_LEN = 16;
  const NO_POSE = new Float32Array(POSE_LEN).fill(NaN);

  /** Eye-aspect-ratio landmarks, [outer, upper1, upper2, inner, lower2, lower1] per eye. */
  const EAR_RIGHT = [33, 160, 158, 133, 153, 144];
  const EAR_LEFT = [362, 385, 387, 263, 373, 380];

  /** rVFC can stop firing for a video the page has hidden; tick anyway after this long. */
  const WATCHDOG_MS = 300;

  function dist(a, b, w, h) {
    const dx = (a.x - b.x) * w;
    const dy = (a.y - b.y) * h;
    return Math.hypot(dx, dy);
  }

  function eyeAspectRatio(lm, idx, w, h) {
    const wide = dist(lm[idx[0]], lm[idx[3]], w, h);
    if (wide <= 0) return 0;
    return (dist(lm[idx[1]], lm[idx[5]], w, h) + dist(lm[idx[2]], lm[idx[4]], w, h)) / (2 * wide);
  }

  /**
   * @param {object} opts
   * @param {SaccadeTracker} opts.tracker - owns the camera; its <video> is the frame source
   * @param {OrtEmbeddingModel} opts.model - the *same* model instance the tracker holds, so the
   *   20 MB encoder is loaded once. Safe only because the tracker's own frame loop is stopped
   *   whenever this one runs: they share one input tensor.
   * @param {object} [opts.assets] - saccade.js asset URLs, for the second MediaPipe instance
   */
  function DualRecorder(opts) {
    this.tracker = opts.tracker;
    this.model = opts.model;
    this.assets = opts.assets || {};

    this.landmarker = null;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.patcher = new global.CookoffWebGazer.EyePatcher();

    this.running = false;
    this.busy = false;
    this.handle = null;
    this.watchdog = null;
    this.generation = 0;
    this.inflight = null;

    this.buffer = null; // non-null while a capture window is open
    this.lastPresented = null;
    this.fps = 0;
    this.lastTick = 0;
    this.embeddingDim = EMB_STRIDE;

    this.stats = { frames: 0, faceMissing: 0, dropped: 0, errors: 0 };
    this.lastFrame = null;
    this.onFrame = opts.onFrame || null;

    /*
     * When set, the encoder is not called from the frame loop at all: crops go to this
     * embedder's worker and the embeddings are written into the data later. This is the path
     * for a machine with no WebGPU, where one encode costs more than a camera frame and
     * running it inline would cost frames rather than merely time. See DeferredEmbedder.
     */
    this.embedder = opts.embedder || null;
  }

  /** Loads this recorder's own MediaPipe face mesh. The camera and the encoder come from the tracker. */
  DualRecorder.prototype.init = async function () {
    if (this.landmarker) return;
    this.landmarker = await global.Saccade.createLandmarker({ assets: this.assets, headPose: true });
    const identity = this.model.identity ? this.model.identity() : null;
    if (identity && identity.dim) this.embeddingDim = identity.dim;
  };

  /**
   * Start the frame loop. The tracker's own loop is stopped first and left stopped: two loops
   * on one camera would double the MediaPipe cost and run two inferences through one tensor.
   */
  DualRecorder.prototype.start = function () {
    if (this.running) return;
    if (!this.landmarker) throw new Error("DualRecorder.init() has not run");
    this.tracker.stop();
    this.running = true;
    this.schedule();
  };

  DualRecorder.prototype.stop = function () {
    this.running = false;
    this.generation++;
    this.clearWatchdog();
    const video = this.tracker.video;
    if (this.handle != null && video.cancelVideoFrameCallback) {
      video.cancelVideoFrameCallback(this.handle);
    }
    this.handle = null;
  };

  /**
   * Open a capture window. Frames land in the buffer until `endCapture()`.
   *
   * `stationId` and `expectedFrames` matter only when the encoder has been deferred to a
   * worker: the first says which station a queued crop belongs to, and the second sets the
   * stride, so the frames that do get encoded are spread across the whole window instead of
   * being its first few.
   */
  DualRecorder.prototype.beginCapture = function (options) {
    const opts = options || {};
    const expected = opts.expectedFrames || Math.max(1, Math.round((this.fps || 30) * 1));
    this.buffer = {
      t0: performance.now(),
      t: [],
      embedding: [],
      weight: [],
      wgLeft: [],
      wgRight: [],
      wgBox: [],
      head: [],
      pose: [],
      ear: [],
      clockSource: null,
      nFaceMissing: 0,
      nDropped: 0,
      nSkipped: 0,
      stationId: opts.stationId != null ? opts.stationId : null,
      stride: this.embedder ? this.embedder.strideFor(expected) : 1,
      queued: [],
    };
    return this.buffer;
  };

  DualRecorder.prototype.endCapture = function () {
    const buffer = this.buffer;
    this.buffer = null;
    return buffer;
  };

  /**
   * Retire the encode that is still in flight, so the last frame of a capture window is not
   * the one frame the pipelining throws away. Callers await this before `endCapture()`.
   */
  DualRecorder.prototype.flush = async function () {
    const pending = this.inflight;
    this.inflight = null;
    if (pending) await this.retire(pending);
  };

  DualRecorder.prototype.schedule = function () {
    if (!this.running) return;
    const generation = ++this.generation;
    const video = this.tracker.video;
    this.clearWatchdog();
    if (video.requestVideoFrameCallback) {
      this.handle = video.requestVideoFrameCallback((now, meta) => {
        if (generation !== this.generation) return;
        this.clearWatchdog();
        void this.tick(now, meta);
      });
      this.armWatchdog(generation);
    } else {
      this.handle = requestAnimationFrame((now) => {
        if (generation !== this.generation) return;
        void this.tick(now, null);
      });
    }
  };

  DualRecorder.prototype.armWatchdog = function (generation) {
    this.clearWatchdog();
    this.watchdog = setTimeout(() => {
      this.watchdog = null;
      if (!this.running || generation !== this.generation) return;
      const video = this.tracker.video;
      if (video.readyState < 2 || video.paused) {
        this.armWatchdog(generation);
        return;
      }
      this.generation++;
      if (this.handle != null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(this.handle);
      }
      this.handle = null;
      void this.tick(performance.now(), null);
    }, WATCHDOG_MS);
  };

  DualRecorder.prototype.clearWatchdog = function () {
    if (this.watchdog != null) {
      clearTimeout(this.watchdog);
      this.watchdog = null;
    }
  };

  DualRecorder.prototype.tick = async function (now, meta) {
    if (!this.running) return;
    if (this.busy) {
      this.schedule();
      return;
    }
    this.busy = true;
    try {
      await this.step(this.frameTime(now, meta));
    } catch (err) {
      this.stats.errors++;
      if (this.stats.errors === 1) console.error("cook-off recorder frame failed:", err);
    } finally {
      this.busy = false;
      this.schedule();
    }
  };

  /**
   * Prefer `captureTime` (when the camera exposed the frame) over `receiveTime` (when the
   * renderer got it) over the callback clock (which says nothing about the camera at all).
   * Which one was available is recorded per capture window, because it decides what the
   * time-sync offset can be subtracted from.
   */
  DualRecorder.prototype.frameTime = function (now, meta) {
    const captureTime = meta && meta.captureTime;
    const receiveTime = meta && meta.receiveTime;
    const presented = meta && meta.presentedFrames != null ? meta.presentedFrames : null;

    let capture = now;
    let source = "callback";
    if (typeof captureTime === "number" && Number.isFinite(captureTime)) {
      capture = captureTime;
      source = "captureTime";
    } else if (typeof receiveTime === "number" && Number.isFinite(receiveTime)) {
      capture = receiveTime;
      source = "receiveTime";
    }

    let dropped = 0;
    if (presented != null) {
      dropped = this.lastPresented == null ? 0 : Math.max(0, presented - this.lastPresented - 1);
      this.lastPresented = presented;
    }
    return { capture, source, callback: now, presented, dropped };
  };

  DualRecorder.prototype.step = async function (time) {
    const t0 = performance.now();
    if (this.lastTick > 0) {
      const dt = t0 - this.lastTick;
      if (dt > 0) this.fps = this.fps === 0 ? 1000 / dt : this.fps * 0.9 + (1000 / dt) * 0.1;
    }
    this.lastTick = t0;

    const stage = this.capture(time);

    // Deferred: the frame loop never touches the encoder. The row is recorded now, with the
    // crop handed to the worker's queue, and the embedding is written into the data later.
    if (this.embedder) {
      this.record(stage, null, null);
      return;
    }

    // Retire the previous frame's encode before starting this one: one inference in flight.
    const pending = this.inflight;
    this.inflight = null;
    if (pending) await this.retire(pending);
    if (!this.running) return;

    if (stage.crop) {
      const started = performance.now();
      this.inflight = {
        stage,
        started,
        promise: this.model.embed(stage.crop),
      };
    } else {
      this.record(stage, null, null);
    }
  };

  /** Everything that touches the image: one draw, one readback, one detection, two crops. */
  DualRecorder.prototype.capture = function (time) {
    const video = this.tracker.video;
    const stage = { time, faceFound: false, crop: null, wg: null, head: null, pose: null, ear: null };
    const W = video.videoWidth;
    const H = video.videoHeight;
    if (!W || !H) return stage;

    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }
    // Never mirrored: both models were trained on unmirrored frames, and a CSS-mirrored
    // preview does not touch these pixels.
    this.ctx.drawImage(video, 0, 0, W, H);
    const frame = this.ctx.getImageData(0, 0, W, H).data;

    // detectFace() is saccade.js core 0.3.0 and later; an older build has only detect(), and
    // records no pose rather than failing.
    const face = this.landmarker.detectFace
      ? this.landmarker.detectFace(video, performance.now())
      : { landmarks: this.landmarker.detect(video, performance.now()), pose: null };
    const landmarks = face && face.landmarks;
    if (!landmarks) return stage;
    stage.faceFound = true;
    stage.pose = face.pose || NO_POSE;

    const crop = global.Saccade.extractEyeCrop(frame, W, H, landmarks);
    if (crop) stage.crop = crop.data;

    stage.wg = this.patcher.patches(frame, W, H, landmarks);

    const head = new Float32Array(HEAD_LANDMARKS.length * 3);
    for (let i = 0; i < HEAD_LANDMARKS.length; i++) {
      const p = landmarks[HEAD_LANDMARKS[i]];
      head[i * 3] = p.x;
      head[i * 3 + 1] = p.y;
      head[i * 3 + 2] = p.z;
    }
    stage.head = head;
    stage.ear = new Float32Array([
      eyeAspectRatio(landmarks, EAR_LEFT, W, H),
      eyeAspectRatio(landmarks, EAR_RIGHT, W, H),
    ]);
    return stage;
  };

  DualRecorder.prototype.retire = async function (pending) {
    let embedding = null;
    let weight = null;
    try {
      const result = await pending.promise;
      embedding = result.embedding;
      weight = result.weight;
    } catch (err) {
      this.stats.errors++;
      if (this.stats.errors === 1) console.error("cook-off recorder encode failed:", err);
    }
    this.record(pending.stage, embedding, weight);
  };

  /**
   * A frame is kept only when both trackers got what they need from it. A row where one of
   * them is missing would have to be dropped by every analysis anyway, and keeping the arrays
   * rectangular is what lets them be packed as flat binary blobs.
   */
  DualRecorder.prototype.record = function (stage, embedding, weight) {
    this.stats.frames++;
    this.stats.dropped += stage.time.dropped;
    if (!stage.faceFound) this.stats.faceMissing++;

    // A frame counts when both trackers got what they need from it. Inline, that means the
    // embedding came back; deferred, it means there is a crop to encode later. Either way the
    // row carries WebGazer's features, which are cheap enough to never skip.
    const usable = this.embedder ? !!(stage.crop && stage.wg) : !!(embedding && stage.wg);
    this.lastFrame = {
      time: stage.time,
      faceFound: stage.faceFound,
      usable,
      fps: this.fps,
      weight,
      ear: stage.ear,
    };
    if (this.onFrame) this.onFrame(this.lastFrame);

    const buffer = this.buffer;
    if (!buffer) return;

    buffer.nDropped += stage.time.dropped;
    if (!stage.faceFound) buffer.nFaceMissing++;
    if (!usable) {
      if (stage.faceFound) buffer.nSkipped++;
      return;
    }
    if (buffer.clockSource === null) buffer.clockSource = stage.time.source;

    const frameIndex = buffer.t.length;
    buffer.t.push(stage.time.capture);

    if (this.embedder) {
      // Every `stride`-th frame goes to the worker. The rest keep their WebGazer features and
      // their timestamp and simply have no embedding — `embedded_frames` records which is
      // which, so an analysis can line the two trackers up on the frames both have.
      if (frameIndex % buffer.stride === 0) {
        buffer.queued.push(frameIndex);
        this.embedder.submit(buffer.stationId, frameIndex, stage.crop);
      }
    } else {
      buffer.embedding.push(embedding);
      buffer.weight.push([weight == null ? NaN : weight]);
    }
    buffer.wgLeft.push(stage.wg.left);
    buffer.wgRight.push(stage.wg.right);
    buffer.wgBox.push([
      stage.wg.boxes.left.x,
      stage.wg.boxes.left.y,
      stage.wg.boxes.left.width,
      stage.wg.boxes.left.height,
      stage.wg.boxes.right.x,
      stage.wg.boxes.right.y,
      stage.wg.boxes.right.width,
      stage.wg.boxes.right.height,
    ]);
    buffer.head.push(stage.head);
    buffer.pose.push(stage.pose);
    buffer.ear.push(stage.ear);
  };

  /** Pack a finished capture window into the fields written to the trial's data row. */
  /**
   * Pack a finished capture window into the fields written to the trial's data row.
   *
   * Deferred, the embedding fields are left null and `embedded_frames` lists the frames whose
   * crops went to the worker. The embedder fills both in later, by `station_id`, once the
   * worker has caught up — see `DeferredEmbedder.finalize`.
   */
  DualRecorder.prototype.pack = function (buffer, trialStart) {
    const E = global.CookoffEncoding;
    const deferred = !!this.embedder;
    return {
      n_frames: buffer.t.length,
      frame_t: buffer.t.map((t) => Math.round((t - trialStart) * 10) / 10),
      clock_source: buffer.clockSource,
      n_face_missing: buffer.nFaceMissing,
      n_camera_dropped: buffer.nDropped,
      n_skipped: buffer.nSkipped,
      // null means "every frame", which is what the inline path produces.
      embedded_frames: deferred ? buffer.queued.slice() : null,
      saccade_embedding: deferred ? null : E.encodeF16Rows(buffer.embedding, this.embeddingDim),
      saccade_weight: deferred ? null : E.encodeF16Rows(buffer.weight, 1),
      crop_frames: null,
      saccade_crop: null,
      webgazer_gray_left: E.encodeU8Rows(buffer.wgLeft, global.CookoffWebGazer.PATCH_LEN),
      webgazer_gray_right: E.encodeU8Rows(buffer.wgRight, global.CookoffWebGazer.PATCH_LEN),
      webgazer_eye_box: E.encodeI16Rows(buffer.wgBox, 8),
      head_landmarks: E.encodeF16Rows(buffer.head, HEAD_LANDMARKS.length * 3),
      head_pose: E.encodeF16Rows(buffer.pose, POSE_LEN),
      eye_aspect_ratio: E.encodeF16Rows(buffer.ear, 2),
    };
  };

  DualRecorder.prototype.describe = function () {
    const identity = this.model.identity ? this.model.identity() : null;
    return {
      embedding_dim: this.embeddingDim,
      embedding_model: global.Saccade.formatModelIdentity(identity),
      execution_provider: this.embedder ? "wasm-worker" : this.model.provider || null,
      landmarker_delegate: this.landmarker ? this.landmarker.delegate : null,
      video_width: this.tracker.video.videoWidth,
      video_height: this.tracker.video.videoHeight,
      head_landmark_indices: HEAD_LANDMARKS,
      head_pose_recorded: !!(this.landmarker && this.landmarker.headPose),
      eye_crop: [EYE_W, EYE_H],
    };
  };

  global.DualRecorder = DualRecorder;
  global.DualRecorder.HEAD_LANDMARKS = HEAD_LANDMARKS;
})(window);
