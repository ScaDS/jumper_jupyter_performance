/**
 * panel.js
 * Panel creation, event wiring, boundary helpers, and Plotly rendering.
 */

/* ── Panel DOM factory ───────────────────────────────────────────────────── */

/**
 * Builds and returns a panel <div> with metric and level dropdowns.
 * The element is NOT yet attached to the document; call appendChild first.
 *
 * @param {string}   pid    - unique panel ID (e.g. "jump-vis-abc123-panel-0")
 * @param {string}   metric - initially selected metric value
 * @param {string}   level  - initially selected level value
 * @param {Array}    opts   - [[label, value], …] metric options
 * @param {string[]} levs   - available level names
 * @returns {HTMLElement}
 */
function createPanelElement(pid, metric, level, opts, levs) {
  var metricOpts = opts.map(function (o) {
    return '<option value="' + o[1] + '"' + (o[1] === metric ? ' selected' : '') + '>'
           + o[0] + '</option>';
  }).join('');

  var levelOpts = levs.map(function (l) {
    return '<option value="' + l + '"' + (l === level ? ' selected' : '') + '>'
           + l + '</option>';
  }).join('');

  var div       = document.createElement('div');
  div.className    = 'jump-vis-panel';
  div.dataset.pid  = pid;
  div.innerHTML    =
    '<div class="jump-vis-ctrl-row">'
    + '<label>Metric: <select class="jump-vis-metric-sel" id="' + pid + '-metric">'
    + metricOpts + '</select></label>'
    + '<label>Level: <select class="jump-vis-level-sel" id="' + pid + '-level">'
    + levelOpts + '</select></label>'
    + '</div>'
    + '<div class="jump-vis-plot-area" id="' + pid + '-plot"></div>';

  return div;
}

/**
 * Attaches change listeners to a panel's metric and level dropdowns.
 * onUpdate(pid, metric, level) is called on every dropdown change.
 * Does NOT trigger an initial render — the caller is responsible for that.
 *
 * @param {string}   pid      - panel ID
 * @param {Function} onUpdate - (pid, metric, level) → void
 */
function attachPanelEvents(pid, onUpdate) {
  var mSel = document.getElementById(pid + '-metric');
  var lSel = document.getElementById(pid + '-level');
  if (!mSel || !lSel) return;
  mSel.addEventListener('change', function () { onUpdate(pid, mSel.value, lSel.value); });
  lSel.addEventListener('change', function () { onUpdate(pid, mSel.value, lSel.value); });
}

/* ── Boundary helpers ────────────────────────────────────────────────────── */

/**
 * Builds Plotly shapes (rectangles) and annotations (cell labels) for all
 * boundaries whose cell_index falls within [cellRange[0], cellRange[1]].
 *
 * @param {Object[]} bndData   - [{cell_index, x0, x1, color}, …]
 * @param {number[]} cellRange - [lo, hi]
 * @param {number[]} ylim      - [ymin, ymax] of the target figure axis
 * @returns {{ shapes: Object[], annotations: Object[] }}
 */
function buildBoundaryUpdates(bndData, cellRange, ylim) {
  var shapes = [], annots = [];
  var lo = cellRange[0], hi = cellRange[1];
  var ymin = ylim[0], ymax = ylim[1];
  var height = (ymax - ymin) || 1.0;

  for (var i = 0; i < bndData.length; i++) {
    var b = bndData[i];
    if (b.cell_index < lo || b.cell_index > hi) continue;

    shapes.push({
      type: 'rect',
      x0: b.x0, x1: b.x1, y0: ymin, y1: ymax,
      xref: 'x', yref: 'y',
      fillcolor: b.color, opacity: 0.4,
      line: { color: 'black', dash: 'dash', width: 1 },
      layer: 'below'
    });
    annots.push({
      x: (b.x0 + b.x1) / 2,
      y: ymax - height * 0.1,
      xref: 'x', yref: 'y',
      text: '#' + b.cell_index,
      showarrow: false,
      font: { size: 10 },
      bgcolor: 'rgba(255,255,255,0.8)'
    });
  }
  return { shapes: shapes, annotations: annots };
}

/**
 * Computes a comfortable x-axis range [xMin-pad, xMax+pad] that frames all
 * boundaries within the selected cell range.  Returns null when none match.
 *
 * @param {Object[]} bndData   - [{cell_index, x0, x1}, …]
 * @param {number[]} cellRange - [lo, hi]
 * @returns {number[]|null}
 */
function xRangeForCells(bndData, cellRange) {
  var lo = cellRange[0], hi = cellRange[1];
  var xMin = Infinity, xMax = -Infinity;

  for (var i = 0; i < bndData.length; i++) {
    var b = bndData[i];
    if (b.cell_index < lo || b.cell_index > hi) continue;
    if (b.x0 < xMin) xMin = b.x0;
    if (b.x1 > xMax) xMax = b.x1;
  }
  if (!isFinite(xMin)) return null;
  var pad = (xMax - xMin) * 0.05 || 0.5;
  return [xMin - pad, xMax + pad];
}

/* ── Plotly rendering ────────────────────────────────────────────────────── */

/**
 * Renders a Plotly figure into a panel's plot-area div.
 * Polls for Plotly.js availability (CDN may still be loading) with a 10 s
 * timeout before showing an error message.
 *
 * @param {HTMLElement}  plotDiv - the .jump-vis-plot-area element
 * @param {Object[]|null} traces - Plotly data array; null shows a no-data message
 * @param {Object}        layout - Plotly layout (already augmented with shapes etc.)
 */
function renderPlotInPanel(plotDiv, traces, layout) {
  if (!plotDiv) return;

  if (!traces) {
    plotDiv.innerHTML =
      '<p class="jump-vis-no-data">No data for selected metric and level.</p>';
    return;
  }

  function doPlot() {
    Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: true });
  }

  if (typeof window.Plotly !== 'undefined') {
    doPlot();
  } else {
    var tries = 0;
    var timer = setInterval(function () {
      tries++;
      if (typeof window.Plotly !== 'undefined') {
        clearInterval(timer);
        doPlot();
      } else if (tries > 100) {   /* 10 s */
        clearInterval(timer);
        plotDiv.innerHTML =
          '<p class="jump-vis-no-data">Plotly.js could not be loaded. '
          + 'Check your network or try re-running the cell.</p>';
      }
    }, 100);
  }
}
