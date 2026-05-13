/**
 * bali_scales.js
 * Shows/hides and labels the BALI tokens-per-second and energy-efficiency
 * gradient bars at the top of the visualizer.  The bars themselves are
 * styled via CSS gradients to match the matplotlib colormaps.
 */

function _fmtScale(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  var abs = Math.abs(v);
  if (abs === 0)            return '0';
  if (abs >= 100)           return v.toFixed(0);
  if (abs >= 10)            return v.toFixed(1);
  if (abs >= 1)             return v.toFixed(2);
  return v.toPrecision(2);
}

/** Updates the displayed min/max labels and visibility of the BALI scales. */
function updateBaliScales(cid) {
  var wrap = document.getElementById(cid + '-bali-scales');
  if (!wrap) return;

  var hasSegments = (
    typeof BALI !== 'undefined'
    && BALI && Array.isArray(BALI.segments)
    && BALI.segments.length > 0
  );
  var enabled = (
    typeof isShowBali === 'function'
    && isShowBali(cid)
    && hasSegments
  );

  wrap.style.display = enabled ? 'flex' : 'none';
  if (!enabled) return;

  var tpsMin = document.getElementById(cid + '-bali-tps-min');
  var tpsMax = document.getElementById(cid + '-bali-tps-max');
  var tpjMin = document.getElementById(cid + '-bali-tpj-min');
  var tpjMax = document.getElementById(cid + '-bali-tpj-max');
  if (tpsMin) tpsMin.textContent = _fmtScale(BALI.vmin);
  if (tpsMax) tpsMax.textContent = _fmtScale(BALI.vmax);
  if (tpjMin) tpjMin.textContent = _fmtScale(BALI.vmin_e);
  if (tpjMax) tpjMax.textContent = _fmtScale(BALI.vmax_e);
}
