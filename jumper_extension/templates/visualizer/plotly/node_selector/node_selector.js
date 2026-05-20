/**
 * node_selector.js
 * Wires the node selector dropdown to a callback fired on every change.
 *
 * @param {string}   cid          - container ID prefix
 * @param {Function} onNodeChange - called with the selected node key ('' = aggregated)
 */
function initNodeSelector(cid, onNodeChange) {
  var sel = document.getElementById(cid + '-node');
  if (sel) sel.addEventListener('change', function () { onNodeChange(sel.value); });
}
