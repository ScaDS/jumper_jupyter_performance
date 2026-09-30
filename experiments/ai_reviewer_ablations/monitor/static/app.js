"use strict";

// The page holds no state beyond the last snapshot it was given. Update
// fetches a fresh one and everything is drawn again: a run is written by
// several jobs at once, so anything remembered here would be a guess about
// what the others have done since.

let snapshot = null;
let colourByShard = false;

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

async function loadRuns() {
  const { runs } = await getJSON("/api/runs");
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
    const run = $("run-select").value;
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

  const summary = results.summary.map((row) => {
    const low = parseFloat(row.paired_delta_ci_low);
    const high = parseFloat(row.paired_delta_ci_high);
    const clear = !Number.isNaN(low) && !Number.isNaN(high) &&
      ((low > 0 && high > 0) || (low < 0 && high < 0));
    return {
      class: clear ? "finding" : "",
      cells: [
        row.ablation, row.metric, row.reported_value,
        { value: fixed(row.estimate) },
        { value: interval(row.ci_low, row.ci_high) },
        { value: fixed(row.paired_delta) },
        { value: interval(row.paired_delta_ci_low, row.paired_delta_ci_high) },
        { value: row.paired_n || "-" },
        { value: row.gaps || "0" },
      ],
    };
  });
  $("summary").replaceChildren(table(
    [{ label: "preset" }, { label: "metric" }, { label: "value" },
     { label: "estimate", num: true }, { label: "interval", num: true },
     { label: "vs base", num: true }, { label: "interval of difference", num: true },
     { label: "pairs", num: true }, { label: "gaps", num: true }],
    summary,
    { empty: "no summary.csv yet - run cli.report" },
  ));

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

update();
