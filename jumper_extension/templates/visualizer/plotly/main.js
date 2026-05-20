/**
 * main.js  –  Orchestration IIFE for the interactive Plotly visualizer.
 *
 * Depends on data variables embedded by Python directly before this script:
 *   CID, FIGS, YLIMS, BND_F, BND_T, OPTS, LEVS, MAX, MIN_CELL, MAX_CELL,
 *   INIT_RNG, NODES
 *
 * FIGS shape: { node: { metric: { level: { "true"|"false": figDict } } } }
 * The "" key holds the aggregate view.  NODES lists the real node hostnames.
 * Single-node sessions have NODES = ["local"]; the node-selector dropdown is
 * rendered but hidden by the HTML hidden attribute.
 *
 * Depends on component functions (loaded in order before this file):
 *   show_idle_checkbox : initShowIdle
 *   cell_range_slider  : getCellRange, initCellRangeSlider
 *   node_selector      : initNodeSelector
 *   add_panel_button   : initAddPanelButton, disableAddPanelButton
 *   panel              : createPanelElement, attachPanelEvents,
 *                        buildBoundaryUpdates, xRangeForCells, renderPlotInPanel
 */
(function () {

  /* ── state ────────────────────────────────────────────────────────────── */
  var panelCount    = 0;
  var usedMetrics   = [];
  /* pid → { metricSel: <select>, levelSel: <select> } */
  var panelRegistry = {};
  /* '' = aggregated view; hostname = per-node view */
  var currentNode   = '';

  /* ── helpers ──────────────────────────────────────────────────────────── */

  /** Returns the next unused metric, cycling back to the first when exhausted. */
  function nextMetric() {
    for (var i = 0; i < OPTS.length; i++) {
      if (usedMetrics.indexOf(OPTS[i][1]) < 0) {
        usedMetrics.push(OPTS[i][1]);
        return OPTS[i][1];
      }
    }
    return OPTS.length ? OPTS[0][1] : null;
  }

  /** Returns 'true' or 'false' based on the show-idle checkbox state. */
  function showIdleKey() {
    var cb = document.getElementById(CID + '-show-idle');
    return (cb && cb.checked) ? 'true' : 'false';
  }

  /* ── single-panel render ──────────────────────────────────────────────── */

  /**
   * Assembles layout (boundaries, x-axis range) and calls renderPlotInPanel
   * for the given panel and selected metric/level.
   */
  function renderPlot(pid, metric, level) {
    var plotDiv = document.getElementById(pid + '-plot');
    var key     = showIdleKey();
    var figData = (((FIGS[currentNode] || {})[metric] || {})[level] || {})[key];

    if (!figData) {
      renderPlotInPanel(plotDiv, null, {});
      return;
    }

    var ylim   = ((((YLIMS[currentNode] || {})[metric] || {})[level]) || {})[key] || [0, 1];
    var rng    = getCellRange(CID, MIN_CELL, MAX_CELL);
    var bndArr = (key === 'true') ? BND_T : BND_F;
    var bnd    = buildBoundaryUpdates(bndArr, rng, ylim);
    var xRng   = xRangeForCells(bndArr, rng);

    /* Clone layout to avoid mutating the shared stored object */
    var layout = JSON.parse(JSON.stringify(figData.layout || {}));
    layout.shapes      = bnd.shapes;
    layout.annotations = bnd.annotations;
    layout.autosize    = true;
    if (xRng) { layout.xaxis = layout.xaxis || {}; layout.xaxis.range = xRng; }

    renderPlotInPanel(plotDiv, figData.data || [], layout);
  }

  /** Re-renders every registered panel. */
  function refreshAll() {
    Object.keys(panelRegistry).forEach(function (pid) {
      var p = panelRegistry[pid];
      renderPlot(pid, p.metricSel.value, p.levelSel.value);
    });
  }

  /* ── panel-row management ─────────────────────────────────────────────── */

  /**
   * Creates a two-panel row, appends it to the panels container, wires events,
   * and triggers the initial render for each new panel.
   */
  function addPanelRow() {
    if (panelCount >= MAX) return;

    var wrap   = document.getElementById(CID + '-panels');
    var row    = document.createElement('div');
    row.className = 'jump-vis-panel-row';

    var pids = [];
    for (var i = 0; i < 2 && panelCount < MAX; i++) {
      var pid    = CID + '-panel-' + panelCount;
      var metric = nextMetric();
      var defLev = (LEVS.indexOf('process') >= 0) ? 'process' : (LEVS[0] || 'process');
      row.appendChild(createPanelElement(pid, metric, defLev, OPTS, LEVS));
      pids.push(pid);
      panelCount++;
    }

    if (pids.length > 0) {
      wrap.appendChild(row);
      pids.forEach(function (pid) {
        attachPanelEvents(pid, renderPlot);
        panelRegistry[pid] = {
          metricSel: document.getElementById(pid + '-metric'),
          levelSel:  document.getElementById(pid + '-level')
        };
        renderPlot(pid, panelRegistry[pid].metricSel.value,
                        panelRegistry[pid].levelSel.value);
      });
    }

    if (panelCount >= MAX) {
      disableAddPanelButton(CID);
      var notice       = document.createElement('p');
      notice.className = 'jump-vis-max-notice';
      notice.textContent = 'All panels have been added.';
      wrap.appendChild(notice);
    }
  }

  /* ── bootstrap ────────────────────────────────────────────────────────── */

  function init() {
    initCellRangeSlider(CID, MIN_CELL, MAX_CELL, INIT_RNG, refreshAll);
    initShowIdle(CID, refreshAll);
    initNodeSelector(CID, function (node) {
      currentNode = node;
      refreshAll();
    });
    initAddPanelButton(CID, addPanelRow);
    addPanelRow();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
