"use strict";

// The page holds no state beyond the last snapshot it was given. Update
// fetches a fresh one and everything is drawn again: a run is written by
// several jobs at once, so anything remembered here would be a guess about
// what the others have done since.

let snapshot = null;
let colourByShard = false;
let knownRuns = [];

const $ = (id) => document.getElementById(id);

function text(value) {
  return value === null || value === undefined ? "" : String(value);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) {
      node.setAttribute(key, String(value));
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

function table(headers, rows, options = {}) {
  if (!rows.length) return el("p", { class: "empty" }, options.empty || "nothing yet");
  const head = el("tr", {}, headers.map((h) =>
    el("th", { class: h.num ? "num" : "" }, h.label ?? h)));
  const body = rows.map((row) => {
    const cells = row.cells.map((cell, index) =>
      el("td", { class: headers[index] && headers[index].num ? "num" : (cell.class || "") },
        cell.node || text(cell.value !== undefined ? cell.value : cell)));
    return el("tr", { class: row.class || "" }, cells);
  });
  return el("table", {}, [el("thead", {}, head), el("tbody", {}, body)]);
}

function bytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value, unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(size < 10 && unit ? 1 : 0)} ${units[unit]}`;
}

function seconds(value) {
  if (!value) return "-";
  const total = Math.round(value);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${total % 60}s`;
  return `${total}s`;
}

// -- loading ------------------------------------------------------------

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function currentRun() {
  return $("run-select").value;
}

async function loadRuns() {
  const { runs } = await getJSON("/api/runs");
  knownRuns = runs;
  const select = $("run-select");
  const chosen = select.value;
  const option = (run) =>
    el("option", { value: run.name }, `${run.name}  (${run.records} records)`);
  select.replaceChildren(...runs.map(option));
  if (chosen && runs.some((run) => run.name === chosen)) select.value = chosen;
  $("compare-runs").replaceChildren(...runs.map(option));
}

async function update() {
  const button = $("update");
  button.disabled = true;
  button.textContent = "reading...";
  try {
    await loadRuns();
    const run = currentRun();
    if (joinState.rows && joinState.info && joinState.info.mine !== run) {
      // A join is a statement about one run against another. Changing which
      // run is being watched makes the old merge a statement about nothing.
      joinState.rows = null;
      joinState.info = null;
    }
    snapshot = await getJSON(`/api/snapshot?run=${encodeURIComponent(run)}`);
    render();
  } catch (failure) {
    $("counters").replaceChildren(el("span", { class: "error" }, failure.message));
  } finally {
    button.disabled = false;
    button.textContent = "Update";
  }
}

// -- rendering ----------------------------------------------------------

function render() {
  renderHeader();
  renderInputs();
  renderShards();
  renderResults();
}

function renderHeader() {
  const { run, totals, records } = snapshot;
  $("run-path").textContent = run.path;
  $("read-at").textContent = `read ${snapshot.read_at.slice(11, 19)}`;

  const done = totals.states.ok || 0;
  const failed = totals.passes - done - (totals.states.running || 0) -
    (totals.states.pending || 0);
  $("counters").replaceChildren(
    counter(`${done}/${totals.passes}`, "passes done"),
    counter(`${totals.records}/${totals.records_expected}`, "records"),
    counter(String(records.empty_context), "empty context"),
    counter(String(failed > 0 ? failed : 0), "not ok"),
    counter(seconds(records.llm_latency_s), "llm time"),
    counter(bytes(run.size_bytes), "on disk"),
  );
}

function counter(value, label) {
  return el("div", { class: "counter" }, [el("b", {}, value), el("span", {}, label)]);
}

function renderInputs() {
  const { matrix, usecases, files } = snapshot.inputs;
  $("matrix").replaceChildren(el("div", { class: "matrix" }, [
    table(
      [{ label: "preset" }, { label: "family" }, ...matrix.sources.map((s) => ({ label: s }))],
      matrix.rows.map((row) => ({
        cells: [
          { node: el("code", {}, row.ablation) },
          row.family,
          ...matrix.sources.map((source) => ({
            value: row.sources[source] ? "on" : "off",
            class: row.sources[source] ? "on" : "off",
          })),
        ],
      })),
    ),
  ]));

  const rows = Object.entries(usecases).map(([id, manifest]) => ({
    cells: [
      { node: el("code", {}, id) },
      manifest.payload_type || "",
      { value: (manifest.reference_facts || []).length },
      { value: (manifest.benchmark || {}).replay_mode || "" },
      {
        node: el("button", {
          class: "cell", style: "border-left-width:1px",
          onclick: () => showUsecase(id, manifest),
        }, "facts and environment"),
      },
    ],
  }));
  $("usecases").replaceChildren(table(
    [{ label: "usecase" }, { label: "payload" }, { label: "facts", num: true },
     { label: "replay" }, { label: "" }],
    rows,
  ));

  $("input-files").replaceChildren(table(
    [{ label: "what" }, { label: "path" }],
    [
      { cells: ["strategies snapshot", { node: el("code", {}, files.strategies) }] },
      { cells: ["composed config", { node: el("code", {}, files.composed_config) }] },
    ],
  ));
}

function showUsecase(id, manifest) {
  const facts = (manifest.reference_facts || []).map((fact) => ({
    cells: [
      { node: el("code", {}, fact.id) },
      fact.source,
      { value: fact.weight },
      fact.fact,
    ],
  }));
  const environment = Object.entries(manifest.environment || {}).map(
    ([key, value]) => ({ cells: [{ node: el("code", {}, key) }, value] }));
  dialog(id, el("div", {}, [
    el("h2", {}, "Reference facts"),
    table([{ label: "id" }, { label: "source" }, { label: "weight", num: true },
           { label: "fact" }], facts),
    el("h2", {}, "Environment"),
    table([{ label: "variable" }, { label: "value" }], environment,
          { empty: "none pinned" }),
  ]));
}

function renderShards() {
  const { grid, shards, shard_count } = snapshot;
  const ablations = [...new Set(grid.map((entry) => entry.ablation))];
  const columns = Math.max(1, ablations.length);

  $("legend").replaceChildren(...(colourByShard
    ? Array.from({ length: shard_count }, (_, index) =>
        legendItem(`var(--shard-${index % 6})`, `shard ${index}`))
    : [["--ok", "ok"], ["--running", "running"], ["--pending", "pending"],
       ["--warn", "incomplete"], ["--bad", "failed or lost"]]
        .map(([token, label]) => legendItem(`var(${token})`, label))));

  const container = el("div", {
    class: "grid",
    style: `grid-template-columns: repeat(${columns}, minmax(150px, 1fr))`,
  });
  for (const entry of grid) {
    const share = entry.records_expected
      ? Math.min(100, (entry.records / entry.records_expected) * 100) : 0;
    container.append(el("button", {
      class: `cell ${colourByShard ? `shard-${entry.shard % 6}` : `state-${entry.state}`}`,
      onclick: () => showPass(entry),
      title: `${entry.usecase} | ${entry.ablation} | r${entry.repetition}`,
    }, [
      el("span", { class: "who" }, entry.ablation),
      el("span", { class: "what" },
        `${entry.state} · ${entry.records}/${entry.records_expected} · shard ${entry.shard}`),
      el("div", { class: "meter" }, el("i", { style: `width:${share}%` })),
    ]));
  }
  $("grid").replaceChildren(container);

  $("shards").replaceChildren(table(
    [{ label: "shard" }, { label: "job" }, { label: "queue" }, { label: "node" },
     { label: "elapsed" }, { label: "passes", num: true },
     { label: "records", num: true }, { label: "now" }],
    shards.map((row) => ({
      cells: [
        row.label || String(row.shard),
        row.job_id || "-",
        row.slurm_state || "-",
        row.node || "-",
        row.elapsed || "-",
        { value: `${row.passes_done}/${row.passes}` },
        { value: `${row.records}/${row.records_expected}` },
        row.current || "-",
      ],
    })),
  ));
}

function legendItem(colour, label) {
  return el("span", {}, [el("b", { style: `background:${colour}` }), label]);
}

async function showPass(entry) {
  const body = el("div", {}, [
    table([{ label: "field" }, { label: "value" }], [
      { cells: ["usecase", entry.usecase] },
      { cells: ["preset", entry.ablation] },
      { cells: ["repetition", String(entry.repetition)] },
      { cells: ["shard", String(entry.shard)] },
      { cells: ["state", entry.state] },
      { cells: ["status in index", entry.status || "not written yet"] },
      { cells: ["records", `${entry.records} of ${entry.records_expected}`] },
      { cells: ["generations seen", entry.generations_seen.join(", ") || "-"] },
      { cells: ["empty context", String(entry.empty_context)] },
      { cells: ["degraded replays", String(entry.degraded)] },
      { cells: ["error", entry.error || "-"] },
    ]),
  ]);

  const mine = snapshot.records.rows.filter((row) =>
    row.usecase === entry.usecase && row.ablation === entry.ablation &&
    row.repetition === entry.repetition);
  body.append(el("h2", {}, "Records"), table(
    [{ label: "generation", num: true }, { label: "phase" }, { label: "mode" },
     { label: "suggestions", num: true }, { label: "latency" }, { label: "tokens", num: true },
     { label: "" }],
    mine.map((row) => ({
      class: row.empty_context ? "gap" : "",
      cells: [
        { value: row.generation }, row.phase,
        row.degraded ? `${row.actual_replay_mode} (fell back)` : row.actual_replay_mode,
        { value: row.suggestions }, seconds(row.llm_latency_s),
        { value: row.total_tokens ?? "-" },
        { node: el("button", { class: "cell", style: "border-left-width:1px",
            onclick: () => showRecord(row.record_id) }, "open") },
      ],
    })),
  ));
  dialog(`${entry.usecase} | ${entry.ablation} | r${entry.repetition}`, body);
}

async function showRecord(recordId) {
  const run = $("run-select").value;
  dialog(recordId, el("p", { class: "empty" }, "reading..."));
  try {
    const record = await getJSON(
      `/api/record?run=${encodeURIComponent(run)}&id=${encodeURIComponent(recordId)}`);
    dialog(recordId, el("pre", {}, JSON.stringify(record, null, 2)));
  } catch (failure) {
    dialog(recordId, el("p", { class: "error" }, failure.message));
  }
}

function renderResults() {
  const { run, results, records } = snapshot;
  $("where").replaceChildren(table(
    [{ label: "what" }, { label: "where or how many" }],
    [
      { cells: ["run directory", { node: el("code", {}, run.path) }] },
      { cells: ["records", String(records.total)] },
      { cells: ["on disk", bytes(run.size_bytes)] },
      ...Object.entries(results.files).map(([name, present]) => ({
        cells: [name, present ? "written" : "not yet"],
      })),
    ],
  ));

  renderSummary();

  $("judge").replaceChildren(table(
    [{ label: "what" }, { label: "count", num: true }],
    [
      { cells: ["packets exported", { value: results.judge_packets }] },
      { cells: ["verdicts written", { value: results.judge_verdicts }] },
      { cells: ["gaps", { value: results.judge_gaps.length }] },
    ],
  ));
}

async function compare() {
  const chosen = [...$("compare-runs").selectedOptions].map((one) => one.value);
  if (chosen.length < 2) {
    $("comparison").replaceChildren(
      el("p", { class: "empty" }, "pick two or more runs"));
    return;
  }
  try {
    const merged = await getJSON(
      `/api/aggregate?runs=${encodeURIComponent(chosen.join(","))}`);
    const { comparable, differs } = merged.compatibility;
    const banner = comparable
      ? el("p", { class: "note" }, "These runs asked the same question.")
      : el("p", { class: "error" },
          `Not comparable: they differ in ${differs.join(", ")}. ` +
          "Shown side by side, not pooled.");
    const headers = [{ label: "usecase" }, { label: "preset" },
      { label: "metric" }, { label: "value" },
      ...merged.runs.map((run) => ({ label: run, num: true }))];
    const rows = merged.rows.map((row) => ({
      cells: [
        row.usecase, row.ablation, row.metric, row.reported_value,
        ...merged.runs.map((run) => ({
          value: row[run] ? fixed(row[run].estimate) : "-",
        })),
      ],
    }));
    $("comparison").replaceChildren(banner, table(headers, rows));
  } catch (failure) {
    $("comparison").replaceChildren(el("p", { class: "error" }, failure.message));
  }
}

// -- filters ------------------------------------------------------------

// What the viewer has narrowed the metric table to. An empty set means "no
// opinion", not "nothing": a facet whose values all got filtered out of a
// particular run would otherwise blank the table rather than ignore itself.
const filters = {
  show: "results",
  usecase: new Set(),
  ablation: new Set(),
  evaluation_method: new Set(),
  metric: new Set(),
};

const FACETS = [
  { key: "usecase", label: "usecase" },
  { key: "ablation", label: "preset" },
  { key: "evaluation_method", label: "method" },
  { key: "metric", label: "metric" },
];

// The three states of the one control the whole table hangs on. "results" is
// the default because two thirds of the rows are denominators and counters:
// useful next to their rate, noise when every one of them has its own line.
const SHOW_MODES = [
  { id: "results", label: "results", hint: "only the metrics' own answers" },
  { id: "inputs", label: "+inputs", hint: "and the quantities they are computed from" },
  { id: "all", label: "all", hint: "and the rows nothing measured" },
];

function restoreFilters() {
  // A per-viewer convenience, so a chosen view survives a reload. It lives
  // in the browser, never in the run directory, and a browser that refuses
  // to store it simply starts from the default.
  try {
    const saved = JSON.parse(localStorage.getItem("monitor.filters") || "{}");
    if (SHOW_MODES.some((mode) => mode.id === saved.show)) filters.show = saved.show;
    for (const facet of FACETS) {
      if (Array.isArray(saved[facet.key])) filters[facet.key] = new Set(saved[facet.key]);
    }
  } catch (failure) {
    // Private windows, cleared site data, storage disabled: not an error.
  }
}

function saveFilters() {
  try {
    localStorage.setItem("monitor.filters", JSON.stringify({
      show: filters.show,
      ...Object.fromEntries(FACETS.map((f) => [f.key, [...filters[f.key]]])),
    }));
  } catch (failure) {
    // See restoreFilters.
  }
}

function hasValue(row) {
  return !Number.isNaN(parseFloat(row.estimate));
}

function applyFilters(rows) {
  let kept = (rows || []).filter((row) =>
    FACETS.every((facet) =>
      !filters[facet.key].size || filters[facet.key].has(row[facet.key])));

  if (filters.show === "results") {
    // Rows written before value_kind existed carry none, and a row with no
    // kind is a result: the alternative hides data because a run is old.
    kept = kept.filter((row) => (row.value_kind || "result") === "result");
  }
  if (filters.show !== "all") {
    // A reported value that no preset measured is a gap, not a comparison.
    // Dropping the whole key rather than the empty cells keeps the surviving
    // row readable: a metric is only absent when it is absent everywhere.
    const measured = new Set();
    for (const row of kept) {
      if (hasValue(row)) measured.add(`${row.usecase}\u0000${row.metric}.${row.reported_value}`);
    }
    kept = kept.filter((row) =>
      measured.has(`${row.usecase}\u0000${row.metric}.${row.reported_value}`));
  }
  return kept;
}

function picker(facet, rows) {
  const values = [...new Set(rows.map((row) => row[facet.key]).filter(Boolean))].sort();
  const chosen = filters[facet.key];
  const count = chosen.size ? ` (${chosen.size})` : "";
  return el("details", { class: "picker" }, [
    el("summary", {}, `${facet.label}${count}`),
    el("div", { class: "options" }, values.map((value) =>
      el("label", {}, [
        el("input", {
          type: "checkbox",
          checked: chosen.has(value) ? "checked" : null,
          onchange: (event) => {
            if (event.target.checked) chosen.add(value);
            else chosen.delete(value);
            saveFilters();
            renderSummary({ keepOpen: facet.key });
          },
        }),
        value,
      ]))),
  ]);
}

function filterBar(rows) {
  const modes = el("div", { class: "modes" }, SHOW_MODES.map((mode) =>
    el("label", { title: mode.hint }, [
      el("input", {
        type: "radio",
        name: "show",
        checked: filters.show === mode.id ? "checked" : null,
        onchange: () => {
          filters.show = mode.id;
          saveFilters();
          renderSummary();
        },
      }),
      mode.label,
    ])));

  const reset = el("button", {
    type: "button",
    class: "link",
    onclick: () => {
      filters.show = "results";
      for (const facet of FACETS) filters[facet.key].clear();
      saveFilters();
      renderSummary();
    },
  }, "reset");

  return el("div", { class: "filters" }, [
    el("span", { class: "filters-label" }, "show:"),
    modes,
    ...FACETS.map((facet) => picker(facet, rows)),
    reset,
  ]);
}

// -- joining another run ------------------------------------------------

// Taking a value from one run or the other is not pooling: nothing is
// averaged and every row says where it came from, which is what makes it
// allowed between runs that asked different questions.
const JOIN_LABELS = {
  mine_wins: "this run wins",
  theirs_win: "the other run wins",
  shared: "only where both measured",
  missing_here: "only what this run is missing",
};

const joinState = { theirs: "", variant: "mine_wins", rows: null, info: null };

function joinBar() {
  const others = knownRuns
    .map((one) => one.name)
    .filter((name) => name !== currentRun());

  const pick = el("select", {
    onchange: (event) => { joinState.theirs = event.target.value; },
  }, [
    el("option", { value: "" }, "join another run..."),
    ...others.map((name) => el("option", {
      value: name,
      selected: name === joinState.theirs ? "selected" : null,
    }, name)),
  ]);

  const variant = el("select", {
    onchange: (event) => {
      joinState.variant = event.target.value;
      if (joinState.rows) runJoin();
    },
  }, Object.entries(JOIN_LABELS).map(([id, label]) =>
    el("option", {
      value: id,
      selected: id === joinState.variant ? "selected" : null,
    }, label)));

  const children = [
    el("span", { class: "filters-label" }, "join:"),
    pick,
    variant,
    el("button", { type: "button", onclick: runJoin }, "join"),
  ];
  if (joinState.rows) {
    children.push(el("button", {
      type: "button", class: "link", onclick: () => {
        joinState.rows = null;
        joinState.info = null;
        renderSummary();
      },
    }, "clear"));
  }
  return el("div", { class: "filters" }, children);
}

function joinBanner() {
  const info = joinState.info;
  if (!info) return null;
  const { comparable, differs } = info.compatibility;
  const counts = info.counts;
  const lines = [
    `${counts.total} rows: ${counts.from_mine} from ${info.mine}, ` +
    `${counts.from_theirs} from ${info.theirs}` +
    (counts.empty ? `, ${counts.empty} measured by neither` : "") +
    `. ${counts.in_both} measured by both.`,
  ];
  if (!comparable) {
    // Allowed, and worth saying out loud: the runs asked different
    // questions, so a row's meaning depends on which run it came from.
    lines.push(
      `These runs differ in ${differs.join(", ")}. Nothing is averaged - ` +
      "each row is one run's own number - but read the source column.");
  }
  return el("p", { class: comparable ? "note" : "error" }, lines.join(" "));
}

async function runJoin() {
  if (!joinState.theirs) {
    joinState.rows = null;
    joinState.info = null;
    renderSummary();
    return;
  }
  try {
    const result = await getJSON("/api/join?" + new URLSearchParams({
      mine: currentRun(),
      theirs: joinState.theirs,
      variant: joinState.variant,
    }));
    joinState.info = result;
    joinState.rows = result.rows.map((row) => ({
      ...row,
      from_other: Boolean(row.source) && row.source !== result.mine,
    }));
  } catch (failure) {
    joinState.rows = null;
    joinState.info = null;
    $("summary").replaceChildren(
      el("p", { class: "error" }, failure.message));
    return;
  }
  renderSummary();
}

function renderSummary(options = {}) {
  const own = (snapshot && snapshot.results && snapshot.results.summary) || [];
  const all = joinState.rows || own;
  const shown = applyFilters(all);
  const bar = filterBar(all);
  const counted = el("p", { class: "note counted" },
    `${shown.length} of ${all.length} rows`);
  $("summary").replaceChildren(
    joinBar(), joinBanner(), bar, counted, ...comparisonTables(shown));
  if (options.keepOpen) {
    // Changing one checkbox re-renders the bar, which would otherwise shut
    // the menu after every single click.
    const index = FACETS.findIndex((facet) => facet.key === options.keepOpen);
    const menus = bar.querySelectorAll("details.picker");
    if (menus[index]) menus[index].open = true;
  }
}

// A preset reads as a column, a metric as a row. The experiment asks one
// question - what changes when a source is removed - and that question is
// answered by reading across, which a flat list of (preset, metric, value)
// rows makes a person do with their finger.
const CATEGORY_TITLES = {
  analysis: "Analysis metrics",
  suggestions: "Suggestions metrics",
};

function findsSomething(row) {
  const low = parseFloat(row.paired_delta_ci_low);
  const high = parseFloat(row.paired_delta_ci_high);
  return !Number.isNaN(low) && !Number.isNaN(high) &&
    ((low > 0 && high > 0) || (low < 0 && high < 0));
}

function comparisonTables(rows) {
  if (!rows || !rows.length) {
    return [el("p", { class: "empty" }, "no summary.csv yet - run cli.report")];
  }

  const baseline = rows.find((row) => row.ablation === "base")
    ? "base" : rows[0].ablation;
  const presets = [...new Set(rows.map((row) => row.ablation))]
    .sort((a, b) => (a === baseline ? -1 : b === baseline ? 1 : a < b ? -1 : 1));
  const usecases = [...new Set(rows.map((row) => row.usecase))].sort();

  const out = [];
  for (const usecase of usecases) {
    if (usecases.length > 1) out.push(el("h3", {}, usecase));
    for (const category of ["analysis", "suggestions"]) {
      const mine = rows.filter((row) =>
        row.usecase === usecase && row.category === category);
      if (!mine.length) continue;

      // One row per reported value, in the order the metrics registered
      // them, so a metric's values stay together.
      const keys = [];
      const cells = new Map();
      const named = new Map();
      for (const row of mine) {
        const key = `${row.metric}.${row.reported_value}`;
        if (!cells.has(key)) { keys.push(key); cells.set(key, {}); }
        cells.get(key)[row.ablation] = row;
        if (row.metric_name) named.set(key, row.metric_name);
      }

      out.push(el("h3", {}, CATEGORY_TITLES[category] || category));
      out.push(table(
        [{ label: "key" }, { label: "metric" },
          ...presets.map((p) => ({
            label: p === baseline ? `${p} (baseline)` : p, num: true }))],
        keys.map((key) => ({
          cells: [
            { node: el("code", {}, key) },
            // The spreadsheet's own name, and only for a result: a
            // denominator is not a row of that table, and neither is a
            // metric the spreadsheet has no row for. Both read as a dash
            // rather than as a repeat of the key.
            { value: named.get(key) || "-",
              class: named.get(key) ? "" : "empty" },
            ...presets.map((preset) => {
              const row = cells.get(key)[preset];
              if (!row) return { value: "-" };
              const shown = fixed(row.estimate);
              if (row.from_other) {
                // Borrowed from the joined run. Marked on the cell rather
                // than stated once above it: a merged table whose numbers
                // cannot be traced back is worse than two tables.
                return {
                  node: el("span", {
                    class: "borrowed",
                    title: `from ${row.source}`,
                  }, shown),
                };
              }
              if (preset === baseline || !findsSomething(row)) {
                return { value: shown };
              }
              // Marked where the interval of the difference from the
              // baseline stays clear of zero - the point estimate alone
              // says nothing about whether the presets differ.
              return {
                node: el("b", {
                  class: "finding-cell",
                  title: `vs ${baseline}: ${fixed(row.paired_delta)} ` +
                    `(${interval(row.paired_delta_ci_low,
                                 row.paired_delta_ci_high)}, ` +
                    `n=${row.paired_n})`,
                }, shown),
              };
            }),
          ],
        })),
      ));
    }
  }
  return out;
}

function fixed(value) {
  const number = parseFloat(value);
  return Number.isNaN(number) ? "-" : number.toFixed(3);
}

function interval(low, high) {
  const a = parseFloat(low), b = parseFloat(high);
  return Number.isNaN(a) || Number.isNaN(b) ? "-" : `${a.toFixed(3)} .. ${b.toFixed(3)}`;
}

// -- chrome -------------------------------------------------------------

function dialog(title, body) {
  $("dialog-title").textContent = title;
  $("dialog-body").replaceChildren(body);
  $("overlay").hidden = false;
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((one) => one.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((one) => one.classList.remove("is-active"));
    tab.classList.add("is-active");
    $(`panel-${tab.dataset.tab}`).classList.add("is-active");
  });
});

$("compare").addEventListener("click", compare);
$("update").addEventListener("click", update);
$("run-select").addEventListener("change", update);
$("colour-by-shard").addEventListener("change", (event) => {
  colourByShard = event.target.checked;
  if (snapshot) renderShards();
});
$("dialog-close").addEventListener("click", () => { $("overlay").hidden = true; });
$("overlay").addEventListener("click", (event) => {
  if (event.target === $("overlay")) $("overlay").hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("overlay").hidden = true;
  if (event.key === "r" && !event.metaKey && !event.ctrlKey) update();
});

restoreFilters();
update();
