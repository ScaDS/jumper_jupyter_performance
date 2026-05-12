/**
 * show_bali_checkbox.js
 * Wires the show-BALI-segments checkbox to a callback fired on every toggle.
 * If no BALI segments are present (BALI.segments empty), the checkbox is
 * disabled and visually muted.
 *
 * @param {string}   cid       - container ID prefix
 * @param {Function} onToggle  - called (no arguments) whenever the checkbox changes
 */
function initShowBali(cid, onToggle) {
  var cb = document.getElementById(cid + '-show-bali');
  if (!cb) return;
  var hasSegments = (
    typeof BALI !== 'undefined'
    && BALI && Array.isArray(BALI.segments)
    && BALI.segments.length > 0
  );
  if (!hasSegments) {
    cb.disabled = true;
    cb.parentElement && (cb.parentElement.style.opacity = '0.5');
    cb.parentElement && cb.parentElement.setAttribute(
      'title', 'No BALI segments available'
    );
    return;
  }
  cb.addEventListener('change', onToggle);
}

/** Returns true when the show-BALI checkbox is checked. */
function isShowBali(cid) {
  var cb = document.getElementById(cid + '-show-bali');
  return !!(cb && cb.checked);
}
