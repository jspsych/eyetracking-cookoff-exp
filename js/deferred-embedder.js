/**
 * Backgrounded eye embedding: a queue on this side, a pool of workers on the other.
 *
 * Why this exists, in numbers. The encoder costs about 20 ms on WebGPU and runs inside the
 * frame loop. Without WebGPU it runs on single-threaded wasm at one to several hundred
 * milliseconds — longer than the 33 ms between camera frames, so it cannot run inline at all
 * without costing frames. In a worker it no longer stalls the recording, and a pool of them
 * encodes several frames at once.
 *
 * Every frame is queued, the arrival saccade in the settle included (from version 1.3.1; before
 * that the worker got a per-station budget of fixation frames). A fast machine keeps up, and
 * the station gaps, the breaks and the "finishing up" screen let it catch up. A slow one falls
 * behind, and what it never reaches is written into the data as raw eye crops for the offline
 * replay to encode.
 *
 * Three things keep that honest:
 *
 *   - WebGazer's features are cut for *every* frame regardless. `embedded_frames` on each
 *     station says which frames also have an embedding; `crop_frames` which have a crop.
 *   - Nothing is silently dropped, up to a cap. The upload has to stay under DataPipe's 32 MB
 *     limit, so past `maxCrops` the leftover crops are thinned evenly across the session, and
 *     the frames dropped are counted.
 *   - The encode time is measured and recorded, so a session that ran on one slow worker is
 *     distinguishable from one that ran on four fast ones.
 */
(function (global) {
  "use strict";

  const CROP_LEN = 144 * 36;

  /**
   * A pool of workers, each single-threaded with its own copy of the model, each encoding a
   * different frame. Frames are independent, so throughput scales with the pool where splitting
   * one frame's encode across threads did not: the model is too small for that (on an M3, four
   * threads encoded slower than two). It also needs no cross-origin isolation, so it works on
   * any host. The cost is memory: every worker holds the model and ONNX Runtime.
   *
   * @param {object} opts
   * @param {string} opts.workerUrl
   * @param {string} opts.modelUrl
   * @param {string} opts.ortUrl - the ESM entry point for onnxruntime-web
   * @param {string} opts.wasmDir - directory holding onnxruntime-web's .wasm
   * @param {number} [opts.workers=1] - how many workers to run
   * @param {number} [opts.numThreads=1] - threads per worker (only on an isolated page)
   */
  function DeferredEmbedder(opts) {
    this.opts = opts;
    this.slots = []; // [{ worker, inflight }]
    this.worker = null; // the first worker, for anything that needs just one
    this.ready = null;
    this.identity = null;

    this.queue = [];
    this.nextId = 1;
    this.pendingById = new Map(); // id -> { resolve, reject } (awaited) or { slot, item } (queued)

    /** stationId -> { indices: number[], crops: Map, results: Map } */
    this.stations = new Map();

    this.encodeMs = null; // effective ms per frame across the whole pool
    this.latencyMs = null; // ms for one encode on one worker
    this.encoded = 0;
    this.failed = 0;
    this.onProgress = null;
  }

  /**
   * Start the pool. Resolves once at least one worker is ready; a worker that fails to start
   * (out of memory, say) is dropped rather than failing the whole encoder.
   */
  DeferredEmbedder.prototype.init = function () {
    if (this.ready) return this.ready;
    const count = Math.max(1, this.opts.workers || 1);
    const boots = [];
    for (let i = 0; i < count; i++) boots.push(this.startWorker());
    this.ready = Promise.allSettled(boots).then((outcomes) => {
      this.slots = outcomes.filter((o) => o.status === "fulfilled").map((o) => o.value);
      const failures = outcomes.filter((o) => o.status === "rejected");
      if (this.slots.length === 0) throw failures[0].reason;
      if (failures.length) console.warn(`cook-off: ${failures.length} embed worker(s) did not start`);
      this.worker = this.slots[0].worker;
      return this.identity;
    });
    return this.ready;
  };

  DeferredEmbedder.prototype.startWorker = function () {
    return new Promise((resolve, reject) => {
      const worker = new Worker(this.opts.workerUrl, { type: "module" });
      const slot = { worker, inflight: null };
      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "ready") {
          this.threads = { isolated: !!message.crossOriginIsolated, count: message.numThreads || 1 };
          this.identity = this.identity || {
            dim: message.dim,
            emitsWeight: message.emitsWeight,
            sha256: message.sha256,
          };
          resolve(slot);
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
        numThreads: this.opts.numThreads || 1,
      });
    });
  };

  /**
   * Time a few real encodes, for the record: nothing is decided by them, but they say how fast
   * this machine's pool was. The worker has already run one inference during `init`, so wasm
   * compilation is not in these numbers.
   */
  DeferredEmbedder.prototype.measure = async function (samples) {
    const n = samples || 3;
    const crop = new Uint8Array(CROP_LEN);
    for (let i = 0; i < CROP_LEN; i++) crop[i] = (i * 37) % 256;
    const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

    // Throughput: rounds of one encode on every worker at once. With one worker this is simply
    // the latency of one encode.
    const rounds = [];
    for (let i = 0; i < n; i++) {
      const started = performance.now();
      await Promise.all(this.slots.map((slot) => this.encodeOn(slot, crop.slice())));
      rounds.push(performance.now() - started);
    }
    const round = median(rounds); // median: one stray scheduling hiccup
    this.latencyMs = round;
    this.encodeMs = round / this.slots.length;
    return this.encodeMs;
  };

  /** One encode on the first worker, awaited. The recorder never awaits the pool. */
  DeferredEmbedder.prototype.encode = function (crop) {
    return this.encodeOn(this.slots[0], crop);
  };

  DeferredEmbedder.prototype.encodeOn = function (slot, crop) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pendingById.set(id, { resolve, reject });
      slot.worker.postMessage({ type: "embed", id, crop: crop.buffer }, [crop.buffer]);
    });
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

  /**
   * Give every idle worker the next queued frame. One encode per worker at a time: each has
   * one session and one input tensor.
   */
  DeferredEmbedder.prototype.pump = function () {
    for (const slot of this.slots) {
      while (!slot.inflight && this.queue.length) {
        const item = this.queue.shift();
        const station = this.stations.get(item.stationId);
        const crop = station && station.crops.get(item.frameIndex);
        if (!crop) continue;
        const id = this.nextId++;
        slot.inflight = { id, item };
        this.pendingById.set(id, { slot, item }); // driven by the queue, not awaited
        // A copy is transferred, so the original stays available as the fallback if the worker
        // never gets to it or fails.
        const transfer = crop.slice();
        slot.worker.postMessage({ type: "embed", id, crop: transfer.buffer }, [transfer.buffer]);
      }
    }
  };

  DeferredEmbedder.prototype.settle = function (message) {
    const pending = this.pendingById.get(message.id);
    this.pendingById.delete(message.id);
    if (!pending) return;

    if (pending.resolve) {
      // From `encode()`: someone is waiting on this specific one.
      if (message.type === "error") pending.reject(new Error(message.message));
      else pending.resolve({ embedding: new Float32Array(message.embedding), weight: message.weight });
      return;
    }

    const inflight = pending.slot.inflight;
    pending.slot.inflight = null;
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
          this.latencyMs = this.latencyMs == null ? message.ms : this.latencyMs * 0.8 + message.ms * 0.2;
          this.encodeMs = this.latencyMs / this.slots.length;
        }
      }
    }
    if (this.onProgress) this.onProgress(this.pending(), this.encoded);
    this.pump();
  };

  DeferredEmbedder.prototype.pending = function () {
    return this.queue.length + this.slots.filter((slot) => slot.inflight).length;
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
   *
   * `maxCrops` caps how many crops go into the data, so a very slow machine's session still
   * fits in one upload. Past it, the crops kept are spread evenly over every frame the worker
   * never reached, so each part of the session keeps some; the rest are counted as dropped.
   */
  DeferredEmbedder.prototype.finalize = function (jsPsych, saveCrops, maxCrops) {
    const E = global.CookoffEncoding;
    const keepCrops = saveCrops !== false;
    const summary = { stations: 0, embedded: 0, asCrops: 0, cropsDropped: 0 };

    let leftover = 0;
    for (const station of this.stations.values()) {
      for (const frameIndex of station.indices) {
        if (!station.results.has(frameIndex) && station.crops.has(frameIndex)) leftover++;
      }
    }
    const cap = Number.isFinite(maxCrops) ? Math.max(0, maxCrops) : Infinity;
    let seen = 0;
    // True for exactly `cap` of the `leftover` crops, evenly spaced in queue order.
    const keepNext = () => {
      const k = seen++;
      if (leftover <= cap) return true;
      return Math.floor(((k + 1) * cap) / leftover) > Math.floor((k * cap) / leftover);
    };

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
          if (!crop) continue;
          if (keepNext()) {
            cropFrames.push(frameIndex);
            crops.push(crop);
          } else {
            summary.cropsDropped++;
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
    for (const slot of this.slots) slot.worker.terminate();
    this.slots = [];
    this.worker = null;
    this.stations.clear();
    this.queue = [];
  };

  global.DeferredEmbedder = DeferredEmbedder;
})(window);
