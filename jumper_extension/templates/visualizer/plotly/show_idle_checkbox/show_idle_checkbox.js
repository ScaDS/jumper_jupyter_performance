/**
 * show_idle_checkbox.js
 * Wires the show-idle checkbox to a callback fired on every toggle.
 *
 * @param {string}   cid       - container ID prefix
 * @param {Function} onToggle  - called (no arguments) whenever the checkbox changes
 */
function initShowIdle(cid, onToggle) {
  var cb = document.getElementById(cid + '-show-idle');
  if (cb) cb.addEventListener('change', onToggle);
}
