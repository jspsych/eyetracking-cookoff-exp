/**
 * The eye encoder, on its own thread.
 *
 * On a machine with WebGPU the encoder costs about 20 ms and runs inside the frame loop. On a
 * machine without it, ONNX Runtime falls back to single-threaded wasm and the same call costs
 * one to several hundred milliseconds — more than a camera frame, sometimes many times more.
 * Run that on the main thread and it does not merely slow the task down: it stalls the frame
 * loop, so MediaPipe misses frames, the capture windows come back half empty, and the
 * participant's data is worse than if the encoder had never run.
 *
 * So on those machines the encoder moves here. The main thread keeps doing the part that must
 * happen live — camera readback, face mesh, both feature cuts — and posts crops over. This
 * thread answers whenever it can, and nothing waits for it.
 *
 * Protocol:
 *   -> {type: "init", ortUrl, wasmDir, modelUrl}    <- {type: "ready", ...} | {type: "error"}
 *   -> {type: "embed", id, crop: ArrayBuffer}       <- {type: "result", id, embedding, weight}
 *
 * Crop buffers are transferred, not copied, and the embedding comes back the same way.
 */

const EYE_W = 144;
const EYE_H = 36;
const CROP_LEN = EYE_W * EYE_H;

/** Output names that mean "per-frame quality weight", matching @saccadejs/core. */
const WEIGHT_NAMES = new Set(["cal_weight", "weight", "quality"]);

let ort = null;
let session = null;
let inputName = "eye_image";
let outputName = "embedding";
let weightName = null;
let inputTensor = null;
let inputData = null;

async function sha256Hex(bytes) {
  if (!self.crypto || !self.crypto.subtle) return null;
  try {
    const digest = await self.crypto.subtle.digest("SHA-256", new Uint8Array(bytes));
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return null;
  }
}

async function init(message) {
  ort = await import(message.ortUrl);
  ort.env.wasm.wasmPaths = message.wasmDir;
  // Multi-threaded wasm needs SharedArrayBuffer, which a worker only has when the page that
  // started it is cross-origin isolated (COOP/COEP headers). The page decides how many threads
  // to ask for; without isolation it asks for one, and ORT is held to one regardless.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, message.numThreads || 1) : 1;

  const response = await fetch(message.modelUrl);
  if (!response.ok) throw new Error(`failed to fetch the model: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());

  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  if (!session.inputNames.includes(inputName)) inputName = session.inputNames[0];
  if (!session.outputNames.includes(outputName)) outputName = session.outputNames[0];
  weightName = session.outputNames.find((n) => n !== outputName && WEIGHT_NAMES.has(n)) ?? null;

  inputData = new Float32Array(CROP_LEN);
  inputTensor = new ort.Tensor("float32", inputData, [1, EYE_H, EYE_W, 1]);

  // One real inference before anything is timed: the first run pays for wasm compilation and
  // would otherwise be mistaken for the steady-state cost of an encode.
  const warm = await run(new Uint8Array(CROP_LEN));

  return {
    type: "ready",
    dim: warm.embedding.length,
    emitsWeight: weightName != null,
    sha256: await sha256Hex(bytes),
    bytes: bytes.length,
    // What the worker actually got, which is what the timing below was measured with.
    crossOriginIsolated: !!self.crossOriginIsolated,
    numThreads: ort.env.wasm.numThreads,
  };
}

async function run(crop) {
  if (crop.length !== CROP_LEN) throw new Error(`crop length ${crop.length} != ${CROP_LEN}`);
  // Same preprocessing as @saccadejs/core's OrtEmbeddingModel: the raw 0-255 crop, widened to
  // float32. No scaling, no mean subtraction — that all lives inside the graph.
  for (let i = 0; i < CROP_LEN; i++) inputData[i] = crop[i];

  const out = await session.run({ [inputName]: inputTensor });
  const embedding = Float32Array.from(out[outputName].data);

  let weight = null;
  if (weightName) {
    const w = out[weightName].data[0];
    if (typeof w === "number" && Number.isFinite(w) && w >= 0 && w <= 1) weight = w;
  }
  return { embedding, weight };
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      self.postMessage(await init(message));
    } else if (message.type === "embed") {
      if (!session) throw new Error("the worker has not been initialized");
      const started = performance.now();
      const { embedding, weight } = await run(new Uint8Array(message.crop));
      self.postMessage(
        {
          type: "result",
          id: message.id,
          embedding: embedding.buffer,
          weight,
          ms: performance.now() - started,
        },
        [embedding.buffer],
      );
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id: message && message.id,
      message: String((error && error.message) || error),
    });
  }
};
