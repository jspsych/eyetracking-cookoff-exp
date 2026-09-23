/**
 * Getting the data off the participant's machine.
 *
 * A session is roughly a megabyte of base64, which is small for a server and large for a
 * survey tool's text field. Both paths below send the whole jsPsych dataset as one JSON
 * document; nothing is trimmed on the way out, because the point of the session is the feature
 * streams.
 */
(function (global) {
  "use strict";

  function timestamp() {
    return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  }

  function download(json, filename) {
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoked on a later turn of the event loop: Safari has not finished with the URL when
    // click() returns.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /**
   * One POST to DataPipe, gzipped.
   *
   * DataPipe refuses any request over 32 MB before its own code runs, so a failure there looks
   * like a network error, not a helpful 4xx. Gzip keeps a normal session far below that; the
   * base64 feature streams compress less than plain JSON, but the JSON around them compresses
   * well.
   *
   * 201 is stored and 202 is accepted and queued. DataPipe says to treat both as success and
   * not resubmit. A 400 is DataPipe refusing the data (collection off, session limit reached),
   * which resending will not change, so it is marked final.
   */
  async function toDataPipe(json, experimentId, filename) {
    const body = JSON.stringify({ experimentID: experimentId, filename, data: json });
    const headers = { "Content-Type": "application/json", Accept: "*/*" };
    let payload = body;
    if (typeof CompressionStream === "function") {
      const stream = new Blob([body]).stream().pipeThrough(new CompressionStream("gzip"));
      payload = await new Response(stream).blob();
      headers["Content-Encoding"] = "gzip";
    }
    const response = await fetch("https://pipe.jspsych.org/api/data/", {
      method: "POST",
      headers,
      body: payload,
    });
    if (response.status === 201 || response.status === 202) return response.status;
    // DataPipe says why in the body's `error` field (collection off, session limit, duplicate
    // filename, failed validation). Without it a 400 is undiagnosable after the fact.
    let reason = "";
    try {
      const text = await response.text();
      try {
        const parsed = JSON.parse(text);
        reason = [parsed.error, parsed.message].filter(Boolean).join(" — ") || text;
      } catch {
        reason = text;
      }
    } catch {
      // no body to read
    }
    reason = String(reason || "").trim().slice(0, 300);
    const error = new Error(`DataPipe returned ${response.status}${reason ? `: ${reason}` : ""}`);
    error.final = response.status >= 400 && response.status < 500;
    throw error;
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /** Retry what might succeed on another try; stop at once on what will not. */
  async function withRetries(attempts, fn, onRetry) {
    let lastError;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastError = err;
        if (err && err.final) break;
        if (attempt < attempts) {
          onRetry?.(attempt);
          await sleep(1000 * 2 ** (attempt - 1)); // 1, 2, 4 s
        }
      }
    }
    throw lastError;
  }

  /** The session's filename: `<prefix>_<participant>_<timestamp>.json`. */
  function filenameFor(jsPsych, config, prefix) {
    const id = jsPsych.data.get().select("participant_id").values[0] || "anon";
    return `${prefix || config.filename_prefix || "cookoff"}_${id}_${timestamp()}.json`;
  }

  /**
   * @param onRetry called with the attempt number before each retry, to tell the participant
   * @returns {Promise<{mode: string, filename: string, bytes: number, attempts: number,
   *   status: number|null, error: string|null, json: string}>}
   *   Never rejects: a failed upload is reported to the participant and recorded in the data,
   *   not thrown into the middle of a debriefing screen. `json` is returned so the caller can
   *   offer it as a download when the upload fails.
   */
  async function saveSession(jsPsych, config, { prefix, onRetry } = {}) {
    /*
     * The trials and the interaction log, together.
     *
     * `jsPsych.data.get().json()` is the trials only. jsPsych separately records every focus,
     * blur and fullscreen change with a timestamp and the trial it happened in, and that log is
     * thrown away unless it is asked for — so it is wrapped in alongside the trials rather than
     * lost. The analysis loaders already accept either shape: a bare array of trials, or an
     * object with a `trials` key.
     */
    const payload = {
      format: "cookoff/1",
      trials: JSON.parse(jsPsych.data.get().json()),
      interactions: JSON.parse(jsPsych.data.getInteractionData().json()),
    };
    const json = JSON.stringify(payload);
    const filename = filenameFor(jsPsych, config, prefix);
    const result = { mode: config.mode, filename, bytes: json.length, attempts: 0, status: null,
                     error: null, json };

    try {
      if (config.mode === "datapipe") {
        if (!config.datapipe_experiment_id) throw new Error("no DataPipe experiment ID is set");
        result.status = await withRetries(
          config.attempts || 1,
          () => {
            result.attempts++;
            return toDataPipe(json, config.datapipe_experiment_id, filename);
          },
          onRetry,
        );
      } else {
        download(json, filename);
      }
    } catch (err) {
      result.error = err instanceof Error ? err.message : String(err);
      console.error("cook-off: saving the session failed:", err);
    }
    return result;
  }

  global.CookoffSave = { saveSession, download, timestamp };
})(window);
