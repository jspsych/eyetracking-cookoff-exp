/**
 * Packing helpers for the per-frame feature streams.
 *
 * A point's capture window holds ~25 camera frames, and every frame carries a 128-d embedding
 * and a pair of eye patches. Written as JSON numbers that is about 8 bytes per value; as
 * base64 over a typed array it is 1.3. The whole session then fits in about a megabyte, which
 * is the difference between a dataset you can post to a server and one you cannot.
 *
 * Two widths are used, and the choice is not arbitrary:
 *   - uint8 for WebGazer's eye patches, which *are* 8-bit grayscale. Lossless.
 *   - float16 for the embedding, the head landmarks and the model weight, where 11 bits of
 *     mantissa is far below the frame-to-frame noise of a webcam.
 */
(function (global) {
  "use strict";

  /** IEEE 754 binary32 -> binary16, round-to-nearest-even, with overflow to Infinity. */
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);

  function toHalf(value) {
    f32[0] = value;
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;

    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // Inf / NaN
    exp = exp - 127 + 15;
    if (exp >= 0x1f) return sign | 0x7c00; // overflow -> Inf
    if (exp <= 0) {
      // Subnormal, or an underflow to zero.
      if (exp < -10) return sign;
      mant |= 0x800000;
      const shift = 14 - exp;
      const half = (mant >>> shift) + ((mant >>> (shift - 1)) & 1);
      return sign | half;
    }
    // Round the 23-bit mantissa to 10 bits, half to even; a carry rolls into the exponent,
    // which is exactly what adding into the packed word does.
    const rounded = (exp << 10) + (mant >>> 13) + ((mant >>> 12) & 1);
    return sign | rounded;
  }

  function fromHalf(h) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >>> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) return sign * mant * Math.pow(2, -24);
    if (exp === 0x1f) return mant ? NaN : sign * Infinity;
    return sign * (mant + 1024) * Math.pow(2, exp - 25);
  }

  /** base64 of a byte view, chunked so a long array never blows the argument limit. */
  function bytesToBase64(bytes) {
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const CHUNK = 0x8000;
    let binary = "";
    for (let i = 0; i < view.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, view.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  }

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }

  /**
   * Concatenate `rows` (each an array-like of `stride` numbers) into one little-endian
   * float16 blob, base64 encoded. Returns null for an empty set so the data says "nothing
   * here" rather than "an empty string here".
   */
  function encodeF16Rows(rows, stride) {
    if (!rows.length) return null;
    const out = new Uint16Array(rows.length * stride);
    for (let r = 0, o = 0; r < rows.length; r++) {
      const row = rows[r];
      for (let i = 0; i < stride; i++) out[o++] = toHalf(row[i]);
    }
    return bytesToBase64(littleEndian(out));
  }

  function encodeU8Rows(rows, stride) {
    if (!rows.length) return null;
    const out = new Uint8Array(rows.length * stride);
    for (let r = 0; r < rows.length; r++) out.set(rows[r].subarray(0, stride), r * stride);
    return bytesToBase64(out);
  }

  function encodeI16Rows(rows, stride) {
    if (!rows.length) return null;
    const out = new Int16Array(rows.length * stride);
    for (let r = 0, o = 0; r < rows.length; r++) {
      const row = rows[r];
      for (let i = 0; i < stride; i++) out[o++] = clampI16(row[i]);
    }
    return bytesToBase64(littleEndian(out));
  }

  function clampI16(v) {
    const r = Math.round(v);
    return r < -32768 ? -32768 : r > 32767 ? 32767 : r;
  }

  /**
   * Byte order is part of the file format, not of the machine that wrote it. Every consumer
   * can then read the blob as little-endian without asking where it came from.
   */
  const IS_LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  function littleEndian(typed) {
    const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    if (IS_LE) return bytes;
    const swapped = new Uint8Array(bytes.length);
    const width = typed.BYTES_PER_ELEMENT;
    for (let i = 0; i < bytes.length; i += width) {
      for (let b = 0; b < width; b++) swapped[i + b] = bytes[i + width - 1 - b];
    }
    return swapped;
  }

  /** Decoders, for the self-test and for anyone reading the data back in the browser. */
  function decodeF16(b64, stride) {
    const bytes = base64ToBytes(b64);
    const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const rows = [];
    for (let i = 0; i < halves.length; i += stride) {
      const row = new Float32Array(stride);
      for (let j = 0; j < stride; j++) row[j] = fromHalf(halves[i + j]);
      rows.push(row);
    }
    return rows;
  }

  function decodeU8(b64, stride) {
    const bytes = base64ToBytes(b64);
    const rows = [];
    for (let i = 0; i < bytes.length; i += stride) rows.push(bytes.subarray(i, i + stride));
    return rows;
  }

  global.CookoffEncoding = {
    toHalf,
    fromHalf,
    bytesToBase64,
    base64ToBytes,
    encodeF16Rows,
    encodeU8Rows,
    encodeI16Rows,
    decodeF16,
    decodeU8,
  };
})(window);
