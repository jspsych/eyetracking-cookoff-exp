/**
 * What this machine can do, recorded so a session can be read without guessing from the user
 * agent afterwards.
 *
 * Two facts that are easy to conflate and worth keeping apart:
 *
 *   - **WebGPU exists here.** A browser-level capability: `navigator.gpu`, an adapter, and
 *     whether that adapter is a real GPU or the software fallback.
 *   - **ONNX Runtime actually used it.** A separate question. ORT tries WebGPU with graph
 *     capture, then WebGPU without, then wasm, and takes the first that survives creating a
 *     session and running one inference. A machine can have WebGPU and still end up on wasm.
 *
 * Recording only the second leaves you unable to tell a machine with no GPU from one whose GPU
 * ORT could not use — which are different problems with the same symptom.
 */
(function (global) {
  "use strict";

  /**
   * Probe WebGPU without assuming any particular Chrome version.
   *
   * `GPUAdapter.info` is the current spelling; `requestAdapterInfo()` was the previous one and
   * is still what older Chrome offers. Both can return empty strings — the fields are
   * fingerprinting surface and the browser is entitled to withhold them — so everything here
   * degrades to null rather than throwing.
   */
  async function probeWebGPU() {
    const result = {
      webgpu_available: false,
      webgpu_adapter: false,
      webgpu_fallback_adapter: null,
      webgpu_vendor: null,
      webgpu_architecture: null,
      webgpu_description: null,
      webgpu_error: null,
    };

    if (!global.navigator || !global.navigator.gpu) return result;
    result.webgpu_available = true;

    try {
      const adapter = await global.navigator.gpu.requestAdapter();
      if (!adapter) {
        // navigator.gpu exists but no adapter: the usual cause is a blocklisted driver.
        result.webgpu_error = "no adapter";
        return result;
      }
      result.webgpu_adapter = true;

      let info = adapter.info || null;
      if (!info && typeof adapter.requestAdapterInfo === "function") {
        info = await adapter.requestAdapterInfo();
      }
      if (info) {
        result.webgpu_vendor = info.vendor || null;
        result.webgpu_architecture = info.architecture || null;
        result.webgpu_description = info.description || null;
      }
      // A software adapter is WebGPU by the letter and not by the spirit: it reports as
      // available and runs at wasm-like speeds, which would otherwise look like a puzzling
      // outlier in the frame-rate data.
      const fallback =
        typeof adapter.isFallbackAdapter === "boolean"
          ? adapter.isFallbackAdapter
          : info && typeof info.isFallbackAdapter === "boolean"
            ? info.isFallbackAdapter
            : null;
      result.webgpu_fallback_adapter = fallback;
    } catch (error) {
      result.webgpu_error = String((error && error.message) || error);
    }
    return result;
  }

  /**
   * How long saccade.js's per-frame ridge prediction takes.
   *
   * Recorded because it is the step everyone assumes is free, and an assumption load-bearing
   * enough to appear in a methods section should be a number instead. It is a dot product of
   * the embedding with a `2 x d` kernel — 256 multiply-adds at the shipped width, plus one
   * small object allocated for the result — and it comes in around a microsecond. Against tens
   * of milliseconds for the encoder, that is about four decimal places below the thing that
   * decides the frame rate.
   *
   * Returns microseconds per prediction.
   */
  function measureRidgePredict(dim) {
    const d = dim || 128;
    const embedding = new Float32Array(d);
    const kernel = new Float32Array(d * 2);
    for (let i = 0; i < d; i++) {
      embedding[i] = Math.sin(i) * 0.5;
      kernel[i * 2] = Math.cos(i) * 0.01;
      kernel[i * 2 + 1] = Math.sin(i * 2) * 0.01;
    }

    const predict = global.Saccade.predict;
    let sink = 0;
    for (let i = 0; i < 2000; i++) sink += predict(embedding, kernel, 0.5).x; // warm up the JIT

    const N = 20000;
    const started = performance.now();
    for (let i = 0; i < N; i++) sink += predict(embedding, kernel, 0.5).x;
    const perCall = ((performance.now() - started) / N) * 1000;

    // Keep `sink` observable so the loop cannot be optimized away entirely.
    if (!Number.isFinite(sink)) throw new Error("ridge benchmark produced a non-finite result");
    return perCall;
  }

  /** Coarse machine facts worth having as covariates. Both are allowed to be absent. */
  function hardware() {
    return {
      hardware_concurrency: global.navigator.hardwareConcurrency || null,
      device_memory_gb: global.navigator.deviceMemory || null,
    };
  }

  /**
   * Whether the page can get a WebGL context at all, and whose renderer it is.
   *
   * MediaPipe's face tracker needs WebGL even on its CPU path: it uploads each video frame
   * through a WebGL texture, and without a context every detection throws. A browser with no
   * WebGL (GPU blocklisted, hardware acceleration off, or `--disable-gpu`) therefore can never
   * find a face, however good the camera. Recent Chrome no longer falls back to software WebGL
   * on its own, so this is a real configuration, not a curiosity.
   */
  function probeWebGL() {
    for (const kind of ["webgl2", "webgl"]) {
      try {
        const gl = document.createElement("canvas").getContext(kind);
        if (!gl) continue;
        const info = gl.getExtension("WEBGL_debug_renderer_info");
        const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : null;
        gl.getExtension("WEBGL_lose_context")?.loseContext();
        return {
          webgl_version: kind,
          webgl_renderer: renderer,
          webgl_software: /swiftshader|llvmpipe|software/i.test(renderer || ""),
        };
      } catch {
        // try the next kind
      }
    }
    return { webgl_version: null, webgl_renderer: null, webgl_software: null };
  }

  global.CookoffCapability = { probeWebGPU, probeWebGL, measureRidgePredict, hardware };
})(window);
