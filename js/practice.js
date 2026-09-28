/**
 * Practice, repeated until every color change in a round is caught.
 *
 * Two sessions in the first 1.3.x batches never pressed a key in the whole study, practice
 * included, and were found only afterwards, by the attention rule. So a practice round with a
 * missed color change is repeated, after a reminder of what to look for, and after `maxRounds`
 * rounds without a clean one the participant is screened out rather than looped forever.
 *
 * Every round's rows stay in the data with `practice: true` and `practice_round` (1, 2, ...);
 * none are analyzed. Kept out of experiment.js so test/practice-check.html can drive the same
 * code with a stand-in for the capture trial.
 */
(function (global) {
  "use strict";

  /**
   * @param {object} o
   * @param {object} o.jsPsych
   * @param {object} o.trial - the station trial; its rows must carry `probe` and `probe_outcome`
   * @param {object[]} o.stations - timeline variables for one round (the same every round)
   * @param {number} [o.maxRounds] - rounds before giving up; 0 or missing means no limit
   * @param {(probes: number, missed: number) => string} o.reminder - the "practice again" screen
   * @param {(rounds: number) => void} o.onFail - called when the last round still had a miss
   * @returns {object[]} two timeline nodes: the practice loop, then the gate that decides
   */
  function build(o) {
    const jsPsych = o.jsPsych;
    const maxRounds = o.maxRounds || Infinity;
    let round = 0;

    const probed = () =>
      jsPsych.data.get().filter({ practice: true, practice_round: round, probe: true }).values();
    const missed = () => probed().filter((r) => r.probe_outcome !== "hit").length;
    const clean = () => round > 0 && missed() === 0;

    const once = {
      timeline: [
        {
          ...o.trial,
          on_finish: (data) => {
            data.practice_round = round;
            if (o.trial.on_finish) o.trial.on_finish(data);
          },
        },
      ],
      timeline_variables: o.stations,
      on_timeline_start: () => {
        round++;
      },
      data: { practice: true },
    };

    const again = {
      timeline: [
        {
          type: global.jsPsychHtmlButtonResponse,
          // Built when shown, so the counts are the round that just finished.
          stimulus: () => o.reminder(probed().length, missed()),
          choices: ["Practice again"],
          button_html: (choice) => `<button class="ck-btn">${choice}</button>`,
          data: { screening: "practice_again" },
        },
      ],
      conditional_function: () => !clean() && round < maxRounds,
    };

    const loop = {
      timeline: [once, again],
      loop_function: () => !clean() && round < maxRounds,
    };

    const gate = {
      type: global.jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 0,
      data: { screening: "practice_gate" },
      on_finish: (data) => {
        data.practice_rounds = round;
        data.practice_passed = clean();
        jsPsych.data.addProperties({ practice_rounds: round });
        if (!clean()) o.onFail(round);
      },
    };

    return [loop, gate];
  }

  global.CookoffPractice = { build };
})(window);
