/**
 * add_panel_button.js
 * Wires the "+ Add Panel" button and exposes a helper to disable it.
 */

/**
 * Attaches a click listener to the Add Panel button.
 *
 * @param {string}   cid     - container ID prefix
 * @param {Function} onClick - called (no arguments) on each click
 */
function initAddPanelButton(cid, onClick) {
  var btn = document.getElementById(cid + '-add-btn');
  if (btn) btn.addEventListener('click', onClick);
}

/**
 * Disables the Add Panel button (called once the panel limit is reached).
 *
 * @param {string} cid - container ID prefix
 */
function disableAddPanelButton(cid) {
  var btn = document.getElementById(cid + '-add-btn');
  if (btn) btn.disabled = true;
}
