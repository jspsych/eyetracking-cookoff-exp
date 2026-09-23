/**
 * Backgrounded eye embedding: a queue on this side, a worker on the other.
 *
 * Why this exists, in numbers. The encoder costs about 20 ms on WebGPU and runs inside the
 * frame loop. Without WebGPU it runs on single-threaded wasm at one to several hundred
 * milliseconds — longer than the 33 ms between camera frames, so it cannot run inline at all
 * without costing frames. Moving it to a worker fixes the stall but does not, by itself, fix
 * the arithmetic: at 200 ms a station's thirty frames need six seconds of encoding and a
 * station only lasts about two.
 *
 * So backgrounding comes with a budget. The warm-up measures what an encode actually costs on
 * this machine, and the recorder queues only as many frames per station as the worker can
 * plausibly finish — evenly spaced across the capture window, so what is kept is a thinner
 * sample of the same window rather than its first few frames.
 *
 * Three things keep that honest:
 *
 *   - WebGazer's features are cut for *every* frame regardless, because they cost almost
 *     nothing. The budget thins one tracker's stream and not the other's, so a head-to-head
 *     comparison has to be run on the frames both have. `embedded_frames` on each station says
 *     which those are; `analysis/replay.py` intersects on it.
 *   - Nothing is silently dropped. A frame that was queued but never encoded — because the
 *     machine was slower than the budget assumed — has its raw crop written to the data
 *     instead, and the offline replay encodes it there. Twenty times the bytes, identical
 *     information.
 *   - The budget is recorded, so a session collected at eight frames a station is
 *     distinguishable from one collected at thirty.
 */
(function (global) {
  "use strict";

  const CROP_LEN = 144 * 36;

  /**
   * @param {object} opts
   * @param {string} opts.workerUrl
   * @param {string} opts.modelUrl
   * @param {string} opts.ortUrl - the ESM entry point for onnxruntime-web
   * @param {string} opts.wasmDir - directory holding onnxruntime-web's .wasm
   */
  function DeferredEmbedder(opts) {
    this.opts = opts;
    this.worker = null;
    this.ready = null;
    this.identity = null;

    this.queue = [];
    this.inflight = null;
    this.nextId = 1;
    this.pendingById = new Map();

    /** stationId -> { indices: number[], crops: Map, results: Map } */
    this.stations = new Map();

    this.encodeMs = null;
    this.encoded = 0;
    this.failed = 0;
    this.budget = Infinity;
    this.onProgress = null;
  }

  /** Boot the worker and measure what one encode costs here. Resolves with the measurement. */
  DeferredEmbedder.prototype.init = function () {
    if (this.ready) return this.ready;
    this.ready = new Promise((resolve, reject) => {
      const worker = new Worker(this.opts.workerUrl, { type: "module" });
      this.worker = worker;

      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "ready") {
          this.identity = {
            dim: message.dim,
            emitsWeight: message.emitsWeight,
            sha256: message.sha256,
          };
          resolve(this.identity);
        } else if (message.type === "result") {
          this.settle(message);
        } else if (message.type === "error") {
          if (message.id != null) this.settle(message);
          else reject(new Error(message.message));
        }
      };
      worker.onerror = (event) => reject(new Error(event.message || "the embed worker failed"));

      worker.postMessage({
        type: "init",
        ortUrl: this.opts.ortUrl,
        wasmDir: this.opts.wasmDir,
        modelUrl: this.opts.modelUrl,
      });
    });
    return this.ready;
  };

  /**
   * Time a few real encodes, so the budget is built from this machine rather than from an
   * assumption about it. The worker has already run one inference during `init`, so wasm
   * compilation is not in these numbers.
   */
  DeferredEmbedder.prototype.measure = async function (samples) {
    const n = samples || 3;
    const crop = new Uint8Array(CROP_LEN);
    for (let i = 0; i < CROP_LEN; i++) crop[i] = (i * 37) % 256;

    const times = [];
    for (let i = 0; i < n; i++) {
      const started = performance.now();
      await this.encode(crop.slice());
      times.push(performance.now() - started);
    }
    times.sort((a, b) => a - b);
    this.encodeMs = times[Math.floor(times.length / 2)]; // median: one stray scheduling hiccup
    return this.encodeMs;
  };

  /** One encode, awaited. Used only by `measure`; the recorder never awaits the worker. */
  DeferredEmbedder.prototype.encode = function (crop) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pendingById.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embed", id, crop: crop.buffer }, [crop.buffer]);
    });
  };

  /**
   * How many frames per station the worker can be expected to finish.
   *
   * `perStationMs` is the whole station cycle, not just the gap between capture windows: the
   * worker is on another thread, so it keeps going while the participant is looking at the
   * target. `share` leaves headroom for a machine whose cores are already busy.
   */
  DeferredEmbedder.prototype.setBudget = function (perStationMs, share, min, max) {
    if (!this.encodeMs) return (this.budget = max);
    const affordable = Math.floor((perStationMs * share) / this.encodeMs);
    this.budget = Math.max(min, Math.min(max, affordable));
    return this.budget;
  };

  /** Keep every `stride`-th frame, so the kept frames span the whole capture window. */
  DeferredEmbedder.prototype.strideFor = function (expectedFrames) {
    if (!Number.isFinite(this.budget) || this.budget >= expectedFrames) return 1;
    return Math.max(1, Math.ceil(expectedFrames / this.budget));
  };

  /** Queue one frame's crop. The crop is copied: the recorder reuses its buffer. */
  DeferredEmbedder.prototype.submit = function (stationId, frameIndex, crop) {
    let station = this.stations.get(stationId);
    if (!station) {
      station = { indices: [], crops: new Map(), results: new Map() };
      this.stations.set(stationId, station);
    }
    const copy = new Uint8Array(crop); // the recorder's crop buffer is reused next frame
    station.indices.push(frameIndex);
    station.crops.set(frameIndex, copy);
    this.queue.push({ stationId, frameIndex });
    this.pump();
  };

  /** Keep exactly one encode in flight; the worker has one session and one input tensor. */
  DeferredEmbedder.prototype.pump = function () {
    if (this.inflight || this.queue.length === 0 || !this.worker) return;
    const item = this.queue.shift();
    const station = this.stations.get(item.stationId);
    const crop = station && station.crops.get(item.frameIndex);
    if (!crop) return this.pump();

    const id = this.nextId++;
    this.inflight = { id, item };
    this.pendingById.set(id, null); // marker: this one is driven by the queue, not awaited
    // A copy is transferred, so the original stays available as the fallback if the worker
    // never gets to it or fails.
    const transfer = crop.slice();
    this.worker.postMessage({ type: "embed", id, crop: transfer.buffer }, [transfer.buffer]);
  };

  DeferredEmbedder.prototype.settle = function (message) {
    const awaited = this.pendingById.get(message.id);
    this.pendingById.delete(message.id);

    if (awaited) {
      // From `encode()`: someone is waiting on this specific one.
      if (message.type === "error") awaited.reject(new Error(message.message));
      else awaited.resolve({ embedding: new Float32Array(message.embedding), weight: message.weight });
      return;
    }

    const inflight = this.inflight;
    this.inflight = null;
    if (inflight && inflight.id === message.id) {
      const station = this.stations.get(inflight.item.stationId);
      if (message.type === "error") {
        this.failed++;
        if (this.failed === 1) console.error("cook-off embed worker:", message.message);
      } else if (station) {
        station.results.set(inflight.item.frameIndex, {
          embedding: new Float32Array(message.embedding),
          weight: message.weight,
        });
        // The crop was only kept as the fallback for a frame that never got encoded.
        station.crops.delete(inflight.item.frameIndex);
        this.encoded++;
        if (typeof message.ms === "number") {
          this.encodeMs = this.encodeMs == null ? message.ms : this.encodeMs * 0.8 + message.ms * 0.2;
        }
      }
    }
    if (this.onProgress) this.onProgress(this.pending(), this.encoded);
    this.pump();
  };

  DeferredEmbedder.prototype.pending = function () {
    return this.queue.length + (this.inflight ? 1 : 0);
  };

  /**
   * Wait for the queue to empty, or for `deadlineMs` to pass.
   *
   * The deadline is what stops a slow machine from holding a participant on a progress bar
   * indefinitely. Whatever is still queued when it expires is written out as crops instead, so
   * the wait is bounded without costing any data.
   */
  DeferredEmbedder.prototype.drain = function (deadlineMs, onProgress) {
    const started = performance.now();
    const total = this.pending();
    return new Promise((resolve) => {
      const tick = () => {
        const left = this.pending();
        if (onProgress) onProgress(left, total);
        if (left === 0) return resolve({ left: 0, timedOut: false });
        if (performance.now() - started > deadlineMs) return resolve({ left, timedOut: true });
        setTimeout(tick, 100);
      };
      this.pump();
      tick();
    });
  };

  /**
   * Write what the worker produced into the station's row in the jsPsych dataset.
   *
   * The row is found by `station_id` and mutated in place — `jsPsych.data.get().values()`
   * hands back the stored trial objects themselves, so a station that finished minutes ago can
   * still be completed. Frames the worker never reached keep their crops, which go into the
   * row instead and are encoded by the offline replay.
   */
  DeferredEmbedder.prototype.finalize = function (jsPsych, saveCrops) {
    const E = global.CookoffEncoding;
    const keepCrops = saveCrops !== false;
    const summary = { stations: 0, embedded: 0, asCrops: 0, cropsDropped: 0 };

    for (const [stationId, station] of this.stations) {
      const rows = jsPsych.data.get().filter({ station_id: stationId }).values();
      if (rows.length === 0) continue;
      const row = rows[0];

      const embeddedFrames = [];
      const embeddings = [];
      const weights = [];
      const cropFrames = [];
      const crops = [];

      for (const frameIndex of station.indices) {
        const result = station.results.get(frameIndex);
        if (result) {
          embeddedFrames.push(frameIndex);
          embeddings.push(result.embedding);
          weights.push([result.weight == null ? NaN : result.weight]);
        } else if (keepCrops) {
          const crop = station.crops.get(frameIndex);
          if (crop) {
            cropFrames.push(frameIndex);
            crops.push(crop);
          }
        } else {
          // A 144x36 strip across both eyes is a recognizable picture of someone's eyes, and
          // this dataset is meant to be publishable. The frame is dropped rather than shipped.
          summary.cropsDropped++;
        }
      }

      const dim = (this.identity && this.identity.dim) || 128;
      row.embedded_frames = embeddedFrames;
      row.saccade_embedding = E.encodeF16Rows(embeddings, dim);
      row.saccade_weight = E.encodeF16Rows(weights, 1);
      row.crop_frames = cropFrames.length ? cropFrames : null;
      row.saccade_crop = crops.length ? E.encodeU8Rows(crops, CROP_LEN) : null;

      summary.stations++;
      summary.embedded += embeddedFrames.length;
      summary.asCrops += cropFrames.length;
    }

    // The crops were the fallback and are now in the data; do not hold megabytes of them.
    this.stations.clear();
    return summary;
  };

  DeferredEmbedder.prototype.dispose = function () {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.stations.clear();
    this.queue = [];
  };

  global.DeferredEmbedder = DeferredEmbedder;
})(window);
