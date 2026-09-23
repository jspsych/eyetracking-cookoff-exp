/**
 * WebGazer's eye features, extracted from saccade.js's MediaPipe landmarks.
 *
 * This is a port of webgazer 3.5.3 — `facemesh.mjs: getEyePatches`, `util.mjs: resizeEye,
 * grayscale, equalizeHistogram, getEyeFeats` — arranged so it can run off a face mesh that has
 * already been computed for something else. WebGazer's eye boxes are the bounding box of two
 * arcs of MediaPipe FaceMesh landmarks, and saccade.js runs a MediaPipe face mesh with the same
 * 468-landmark topology, so one detection per frame feeds both trackers instead of two.
 *
 * What is recorded is the 10x6 grayscale patch per eye, *before* histogram equalization.
 * WebGazer equalizes each patch over its own 60 pixels with nothing else as input, so the
 * equalized feature vector is a pure function of these 60 bytes: storing them is lossless, and
 * a third of the size of storing the equalized floats. `getEyeFeats` below is the same function
 * the offline replay applies, and is here so the two halves can be checked against each other.
 *
 * One deliberate approximation. WebGazer runs the legacy `face_mesh` solution through
 * @tensorflow-models/face-landmarks-detection; saccade.js runs the newer `face_landmarker.task`
 * through @mediapipe/tasks-vision. Same landmark indices, same topology, slightly different
 * weights — so an eye box here can differ from the one real WebGazer would have drawn, usually
 * by a pixel. The box of every frame is recorded so the size of that difference can be measured
 * against a real WebGazer session rather than assumed.
 */
(function (global) {
  "use strict";

  /** webgazer/src/facemesh.mjs. "left" and "right" are from the subject's perspective. */
  const EYE_INDICES = {
    leftEyeUpper0: [466, 388, 387, 386, 385, 384, 398],
    leftEyeLower0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
    rightEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
    rightEyeLower0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
  };

  const RESIZE_W = 10;
  const RESIZE_H = 6;
  const PATCH_LEN = RESIZE_W * RESIZE_H;

  /**
   * The eye boxes WebGazer would cut, in pixels of the camera frame.
   *
   * WebGazer works in canvas pixels and MediaPipe Tasks reports normalized coordinates, so the
   * landmarks are scaled here; `Math.round` then lands on the same integers WebGazer's
   * `Math.round` does. The box is the top arc's minimum against the bottom arc's maximum —
   * not the extent of all the points — which is WebGazer's rule, quirks included.
   */
  function eyeBoxes(landmarks, width, height) {
    const box = (upper, lower) => {
      let minX = Infinity;
      let minY = Infinity;
      for (const i of upper) {
        const p = landmarks[i];
        if (p.x * width < minX) minX = p.x * width;
        if (p.y * height < minY) minY = p.y * height;
      }
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const i of lower) {
        const p = landmarks[i];
        if (p.x * width > maxX) maxX = p.x * width;
        if (p.y * height > maxY) maxY = p.y * height;
      }
      const x = Math.round(minX);
      const y = Math.round(minY);
      return { x, y, width: Math.round(maxX) - x, height: Math.round(maxY) - y };
    };

    const left = box(EYE_INDICES.leftEyeUpper0, EYE_INDICES.leftEyeLower0);
    const right = box(EYE_INDICES.rightEyeUpper0, EYE_INDICES.rightEyeLower0);
    if (left.width <= 0 || left.height <= 0 || right.width <= 0 || right.height <= 0) return null;
    return { left, right };
  }

  /**
   * Cuts the 10x6 grayscale patches.
   *
   * The resize goes through a canvas on purpose. WebGazer downsamples with `drawImage`, so the
   * filtering is the browser's, and reimplementing it in JavaScript would produce a different
   * feature vector than WebGazer computes on the same frame.
   *
   * Two details here look like things worth optimizing and are not. Both were caught by
   * `test/port-check.html`, which runs this code and webgazer 3.5.3's own code over the same
   * pixels and demands identical bytes.
   *
   *   1. The canvases are allocated per call, as WebGazer allocates them, instead of being
   *      reused across frames. Chrome rasterizes a reused canvas down a different path, and for
   *      an eye box that runs off the edge of the frame — carrying transparent pixels — the
   *      resized patch came out up to 178 gray levels different. Four small canvases a frame is
   *      what WebGazer pays anyway.
   *   2. The context options are WebGazer's. `willReadFrequently` on the *destination* canvas
   *      changes the downsampling filter and shifts the feature vector by up to 23 gray levels
   *      per pixel. WebGazer passes it on the source canvas and not on the destination.
   *
   * The one thing reused is the RGBA scratch buffer the eye box is cut into, which is a plain
   * typed array and cannot affect a rasterizer.
   */
  function EyePatcher() {
    this.scratch = null;
  }

  /**
   * @param {Uint8ClampedArray} frame - RGBA of the whole camera frame
   * @param {{x,y,width,height}} box
   * @returns {Uint8Array} 60 grayscale bytes, row-major
   */
  EyePatcher.prototype.patch = function (frame, frameW, frameH, box) {
    const n = box.width * box.height * 4;
    if (!this.scratch || this.scratch.length < n) this.scratch = new Uint8ClampedArray(n);
    const rgba = this.scratch.length === n ? this.scratch : this.scratch.subarray(0, n);
    // Out-of-frame pixels stay transparent black, which is what getImageData hands WebGazer
    // when its box runs off the edge of the canvas.
    rgba.fill(0, 0, n);
    for (let row = 0; row < box.height; row++) {
      const sy = box.y + row;
      if (sy < 0 || sy >= frameH) continue;
      let d = row * box.width * 4;
      for (let col = 0; col < box.width; col++, d += 4) {
        const sx = box.x + col;
        if (sx < 0 || sx >= frameW) continue;
        const s = (sy * frameW + sx) * 4;
        rgba[d] = frame[s];
        rgba[d + 1] = frame[s + 1];
        rgba[d + 2] = frame[s + 2];
        rgba[d + 3] = frame[s + 3];
      }
    }

    const src = document.createElement("canvas");
    src.width = box.width;
    src.height = box.height;
    src.getContext("2d", { willReadFrequently: true })
      .putImageData(new ImageData(rgba, box.width, box.height), 0, 0);

    const dst = document.createElement("canvas");
    dst.width = RESIZE_W;
    dst.height = RESIZE_H;
    const dstCtx = dst.getContext("2d");
    dstCtx.drawImage(src, 0, 0, box.width, box.height, 0, 0, RESIZE_W, RESIZE_H);

    const resized = dstCtx.getImageData(0, 0, RESIZE_W, RESIZE_H).data;
    // webgazer/src/util.mjs: grayscale() writes into a Uint8ClampedArray, so the rounding is
    // the clamped array's round-half-to-even and not a truncation. Same container, same bytes.
    const gray = new Uint8ClampedArray(PATCH_LEN);
    for (let i = 0, p = 0; i < PATCH_LEN; i++, p += 4) {
      gray[i] = resized[p] * 0.299 + resized[p + 1] * 0.587 + resized[p + 2] * 0.114;
    }
    return gray;
  };

  /** Both patches for one frame, or null if either eye box is unusable. */
  EyePatcher.prototype.patches = function (frame, frameW, frameH, landmarks) {
    const boxes = eyeBoxes(landmarks, frameW, frameH);
    if (!boxes) return null;
    return {
      boxes,
      left: this.patch(frame, frameW, frameH, boxes.left),
      right: this.patch(frame, frameW, frameH, boxes.right),
    };
  };

  /**
   * webgazer/src/util.mjs: equalizeHistogram(src, step = 5).
   *
   * Note the sampling: the histogram is built from every 5th pixel, so 12 of the 60 decide the
   * mapping, and the result is a float, not a byte. Both are load-bearing for reproducing
   * WebGazer's features exactly.
   */
  function equalizeHistogram(src, step) {
    step = step || 5;
    const hist = new Array(256).fill(0);
    for (let i = 0; i < src.length; i += step) hist[src[i]]++;

    const norm = (255 * step) / src.length;
    let prev = 0;
    for (let i = 0; i < 256; i++) {
      prev = hist[i] += prev;
      hist[i] = prev * norm;
    }

    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = hist[src[i]];
    return dst;
  }

  /**
   * webgazer/src/util.mjs: getEyeFeats(), for `trackEye: 'both'` — left eye then right eye,
   * 120 numbers. This is the vector WebGazer's ridge regression is fitted on and predicts from.
   */
  function getEyeFeats(leftGray, rightGray) {
    const l = equalizeHistogram(leftGray, 5);
    const r = equalizeHistogram(rightGray, 5);
    const out = new Float32Array(l.length + r.length);
    out.set(l, 0);
    out.set(r, l.length);
    return out;
  }

  global.CookoffWebGazer = {
    EYE_INDICES,
    RESIZE_W,
    RESIZE_H,
    PATCH_LEN,
    EyePatcher,
    eyeBoxes,
    equalizeHistogram,
    getEyeFeats,
  };
})(window);
