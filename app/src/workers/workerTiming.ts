/**
 * Import FIRST in any worker that trains TensorFlow.js models (before TF.js is evaluated).
 *
 * TF.js yields between training batches with `requestAnimationFrame` when it exists — and it does
 * exist in dedicated workers. Animation frames only fire while the page is painting, so in a
 * background tab or a hidden/minimised window LSTM training would stall indefinitely. Hiding rAF
 * makes TF.js fall back to `setImmediate`, shimmed here as a 0 ms timeout: a macrotask that always
 * runs and also lets queued messages (e.g. cancel requests) through between batches.
 */
const scope = self as unknown as {
  requestAnimationFrame?: unknown;
  setImmediate?: (fn: () => void) => void;
};
scope.requestAnimationFrame = undefined;
scope.setImmediate = (fn: () => void) => {
  setTimeout(fn, 0);
};

export {};
