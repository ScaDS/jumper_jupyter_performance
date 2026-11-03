"use strict";

// What a metric's number is measured against, and how it is computed.
//
// Loaded before MathJax and app.js: it carries the MathJax configuration,
// the reference group of every metric the table colours, and the formula
// popup that opens over a metric's name. Paper metrics are written in the
// notation of docs/paper_metrics_mapping.md ("What it is"); metrics only we
// report use the notation of its "Ours" column.

window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]] },
  // No shared glyph cache: a popup is rebuilt on every hover, and a cache
  // living in a removed node leaves the next formula without its glyphs.
  svg: { fontCache: "none", mtextInheritFont: true },
  startup: { typeset: false },
};

const metricHelp = (() => {
  // -- reference groups ---------------------------------------------------

  // A ratio over the original cell and one over a person's solution are not
  // the same kind of claim: 500x over the slow input can still be 0.7x of
  // what a human wrote. The table puts them under separate bands.
  const REFERENCE_GROUPS = [
    {
      id: "original",
      title: "vs original code",
      metrics: [
        "mean_median_speedup",
        "mean_median_correct_speedup",
        "best_speedup",
        "correctness_conditioned_speedup",
        "memory_reduction",
        "fast_p",
        "compile_speedup",
      ],
    },
    {
      id: "human",
      title: "vs human solutions",
      metrics: [
        "reference_speedup",
        "reference_memory_reduction",
        "runtime_percentile",
        "memory_percentile",
      ],
    },
    { id: "none", title: "no reference program", metrics: [] },
  ];

  function referenceOf(key) {
    const metric = key.split(".")[0];
    const group = REFERENCE_GROUPS.find((one) => one.metrics.includes(metric));
    return group ? group.id : "none";
  }

  // -- notation -----------------------------------------------------------

  const SPEEDUP = String.raw`\frac{T(\text{slow input})}{T(\text{generated}_i)}`;
  const MEMORY_REDUCTION = String.raw`\frac{M(\text{slow input})}{M(\text{generated}_i)}`;

  const SYMBOLS = {
    runtime: [String.raw`T(\cdot)`,
      "runtime of a program on the private tests; ours: as cli.verify times it, one fresh process each"],
    memory: [String.raw`M(\cdot)`,
      "memory of a program; ours: peak RSS of one fresh process"],
    slowInput: [String.raw`\text{slow input}`,
      "the slow input program; ours: the original cell (KernelBench: the eager Model)"],
    generated: [String.raw`\text{generated}_i`,
      "the i-th generated program; ours: the i-th suggestion"],
    speedup: [String.raw`s_i`, "Speedup of one suggestion, T(slow input) / T(generated)"],
    measured: [String.raw`N`, "suggestions that ran and were timed (measured_suggestions)"],
    correct: [String.raw`C`, "the suggestions verified as correct"],
    correctCount: [String.raw`|C|`, "their number (correct_suggestions)"],
    generation: [String.raw`g`, "one generation: one reviewer response with its list of suggestions"],
    generations: [String.raw`G`,
      "generations with at least one measured suggestion (generations_with_a_measurement)"],
    geometric: [String.raw`\exp(\mathrm{mean}(\log\,\cdot))`,
      "Geometric mean speedup: the mean of the logarithms, mapped back by exp"],
    indicator: [String.raw`\mathbb{1}[\cdot]`, "1 when the condition holds, 0 otherwise"],
    humanFast: [String.raw`\text{ECCO fast human}`,
      "the fast human solution ECCO pairs with the slow input"],
    suggestion: [String.raw`\text{suggestion}_i`, "the i-th suggestion"],
  };

  // -- aggregates ---------------------------------------------------------

  const AGGREGATES = {
    geometric: (set, count, term) =>
      String.raw`\exp\!\left(\frac{1}{${count}}\sum_{i \in ${set}} \log ${term}\right)`,
    mean: (set, count, term) =>
      String.raw`\frac{1}{${count}}\sum_{i \in ${set}} ${term}`,
    median: (set, count, term) =>
      String.raw`\operatorname*{median}_{i \in ${set}}\; ${term}`,
  };

  function speedupOver(set, count, aggregate, name) {
    return String.raw`\mathrm{${name}} = ${AGGREGATES[aggregate](set, count, SPEEDUP)}`;
  }

  // -- formulas -----------------------------------------------------------

  const HELP = {};

  for (const [aggregate, value] of [
    ["geometric", "geometric_mean_speedup"],
    ["mean", "mean_speedup"],
    ["median", "median_speedup"],
  ]) {
    const name = value.replaceAll("_", "\\_");
    HELP[`mean_median_speedup.${value}`] = {
      source: "ECCO: Speedup · KernelBench: Speedup vs PyTorch Eager" +
        (aggregate === "geometric" ? ", Geometric mean speedup" : ""),
      tex: speedupOver("S", "|S|", aggregate, name),
      legend: [
        SYMBOLS.runtime, SYMBOLS.slowInput, SYMBOLS.generated,
        [String.raw`S`, "every suggestion that ran, pooled over the generations"],
        [String.raw`|S|`, "their number (measured_suggestions)"],
        ...(aggregate === "geometric" ? [SYMBOLS.geometric] : []),
      ],
      read: "> 1: faster than the original cell.",
    };
    HELP[`mean_median_correct_speedup.${value}`] = {
      source: "ECCO: Speedup · KernelBench: Speedup vs PyTorch Eager" +
        (aggregate === "geometric" ? ", Geometric mean speedup" : ""),
      tex: speedupOver("C", "|C|", aggregate, name),
      legend: [
        SYMBOLS.runtime, SYMBOLS.slowInput, SYMBOLS.generated,
        SYMBOLS.correct, SYMBOLS.correctCount,
        ...(aggregate === "geometric" ? [SYMBOLS.geometric] : []),
      ],
      read: "> 1: the correct suggestions are faster than the original cell.",
    };
  }

  for (const [aggregate, value] of [
    ["geometric", "geometric_mean_speedup"],
    ["median", "median_speedup"],
  ]) {
    HELP[`correctness_conditioned_speedup.${value}`] = {
      source: "KernelBench: Geometric mean speedup over correct kernels",
      tex: speedupOver("C", "|C|", aggregate, value.replaceAll("_", "\\_")),
      legend: [
        SYMBOLS.runtime, SYMBOLS.slowInput, SYMBOLS.generated,
        SYMBOLS.correct, SYMBOLS.correctCount,
        ...(aggregate === "geometric" ? [SYMBOLS.geometric] : []),
      ],
      read: "Read it beside coverage: a high speedup over few correct suggestions says little.",
    };
  }
  HELP["correctness_conditioned_speedup.coverage"] = {
    source: "Ours",
    tex: String.raw`\mathrm{coverage} = \frac{|C|}{N}`,
    legend: [
      SYMBOLS.correct,
      [String.raw`N`, "every suggestion of every generation (suggestions)"],
    ],
    read: "The share of suggestions the conditioned speedup is computed over.",
  };

  for (const [outer, value, set] of [
    ["mean", "mean_best_speedup", "measured"],
    ["median", "median_best_speedup", "measured"],
    ["mean", "mean_best_correct_speedup", "correct"],
    ["median", "median_best_correct_speedup", "correct"],
  ]) {
    const within = set === "correct" ? String.raw`i \in g \cap C` : String.raw`i \in g`;
    const over = outer === "mean"
      ? String.raw`\frac{1}{G}\sum_{g=1}^{G}`
      : String.raw`\operatorname*{median}_{g=1..G}\;`;
    HELP[`best_speedup.${value}`] = {
      source: "ECCO: Speedup, of the best suggestion in a response",
      tex: String.raw`\mathrm{${value.replaceAll("_", "\\_")}} = ${over} \max_{${within}} ${SPEEDUP}`,
      legend: [
        SYMBOLS.runtime, SYMBOLS.slowInput, SYMBOLS.generated, SYMBOLS.generation,
        [String.raw`i \in g`, "the suggestions of generation g that ran"],
        ...(set === "correct" ? [SYMBOLS.correct] : []),
        SYMBOLS.generations,
      ],
      read: "The speedup of the option a user would pick.",
    };
  }

  HELP["memory_reduction.geometric_mean_memory_reduction"] = {
    source: "ECCO: Memory Reduction",
    tex: String.raw`\mathrm{geometric\_mean\_memory\_reduction} =
      ${AGGREGATES.geometric("C", "|C|", MEMORY_REDUCTION)}`,
    legend: [SYMBOLS.memory, SYMBOLS.slowInput, SYMBOLS.generated,
      SYMBOLS.correct, SYMBOLS.correctCount],
    read: "> 1: the suggestion needs less memory than the original cell.",
  };

  HELP["reference_speedup.geometric_mean_reference_speedup"] = {
    source: "Ours; neither paper reports it",
    tex: String.raw`\mathrm{geometric\_mean\_reference\_speedup} =
      ${AGGREGATES.geometric("C", "|C|",
        String.raw`\frac{T(\text{ECCO fast human})}{T(\text{suggestion}_i)}`)}`,
    legend: [SYMBOLS.runtime, SYMBOLS.humanFast, SYMBOLS.suggestion,
      SYMBOLS.correct, SYMBOLS.correctCount],
    read: "> 1: faster than the human solution; < 1: slower.",
  };

  HELP["reference_memory_reduction.geometric_mean_reference_memory_reduction"] = {
    source: "Ours; neither paper reports it",
    tex: String.raw`\mathrm{geometric\_mean\_reference\_memory\_reduction} =
      ${AGGREGATES.geometric("C", "|C|",
        String.raw`\frac{M(\text{ECCO fast human})}{M(\text{suggestion}_i)}`)}`,
    legend: [SYMBOLS.memory, SYMBOLS.humanFast, SYMBOLS.suggestion,
      SYMBOLS.correct, SYMBOLS.correctCount],
    read: "> 1: needs less memory than the human solution.",
  };

  HELP["runtime_percentile.mean_runtime_percentile"] = {
    source: "ECCO: Runtime %",
    tex: String.raw`\mathrm{mean\_runtime\_percentile} = \frac{1}{|C|}\sum_{i \in C} 100 \cdot
      \frac{\bigl|\{\, s \in S : T(s) > T(\text{generated}_i) \,\}\bigr|}{|S|}`,
    legend: [
      [String.raw`S`, "the CodeNet submissions of the problem; ours: re-run on our inputs. " +
        "Only submissions that ran correctly; a timeout counts as T = ∞"],
      [String.raw`s`, "one submission"],
      SYMBOLS.runtime, SYMBOLS.generated,
      [String.raw`T(s) > T(\text{generated}_i)`, "the submission is slower than the suggestion"],
      SYMBOLS.correct,
      [String.raw`100\,\cdot`, "the share as a percentage"],
    ],
    read: "The percentage of people whose solution is slower. Higher is better.",
  };

  HELP["memory_percentile.mean_memory_percentile"] = {
    source: "ECCO: Memory %",
    tex: String.raw`\mathrm{mean\_memory\_percentile} = \frac{1}{|C|}\sum_{i \in C} 100 \cdot
      \frac{\bigl|\{\, s \in S : M(s) > M(\text{generated}_i) \,\}\bigr|}{|S|}`,
    legend: [
      [String.raw`S`, "the CodeNet submissions of the problem, re-run on our inputs. " +
        "Only submissions that ran correctly; a timeout reported no peak and is left out"],
      [String.raw`s`, "one submission"],
      SYMBOLS.memory, SYMBOLS.generated,
      [String.raw`M(s) > M(\text{generated}_i)`, "the submission needs more memory than the suggestion"],
      SYMBOLS.correct,
      [String.raw`100\,\cdot`, "the share as a percentage"],
    ],
    read: "The percentage of people whose solution needs more memory. Higher is better.",
  };

  HELP["pass_at_k.pass_at_1"] = {
    source: "ECCO: pass@1",
    tex: String.raw`\mathrm{pass@1} = P(\text{first sample passes all tests})
      = \frac{1}{G}\sum_{g=1}^{G} \mathbb{1}\bigl[\text{first sample}_g \text{ passes all tests}\bigr]`,
    legend: [
      [String.raw`P(\cdot)`, "a probability, estimated as a share of generations"],
      [String.raw`\text{first sample}_g`, "the first sample of generation g; ours: the first-ranked suggestion"],
      [String.raw`\text{passes all tests}`, "ours: verified, the output fingerprint matches"],
      SYMBOLS.indicator,
      [String.raw`G`, "generations with at least one verdict (generations)"],
    ],
    read: "The share of responses whose first option is correct.",
  };

  HELP["correctness_rates.verified_rate"] = {
    source: "KernelBench: Correctness (fast_0); ours is defined differently",
    tex: String.raw`\mathrm{verified\_rate} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\Bigl[\,\forall\, x \in X:\;
      \mathrm{isclose}\bigl(x_{\text{ref}},\, x_{\text{gen},i},\; \mathrm{rel\_tol}=10^{-6}\bigr)\Bigr]`,
    paperTex: String.raw`\mathrm{fast}_0 = P\bigl(\mathrm{allclose}(out_{\text{ref}}, out_{\text{gen}},\, atol=rtol=10^{-2})\bigr),
      \qquad |a-b| \le atol + rtol\cdot|b|`,
    legend: [
      [String.raw`X`, "statistics of the output: shape, mean, std, min, max"],
      [String.raw`x_{\text{ref}}`, "statistic x of the original cell's output (stat_ref)"],
      [String.raw`x_{\text{gen},i}`, "the same statistic of the i-th suggestion's output (stat_gen)"],
      [String.raw`\mathrm{isclose}`, "math.isclose with relative tolerance rel_tol"],
      [String.raw`N`, "every suggestion; compared on one input"],
      SYMBOLS.indicator,
      [String.raw`out_{\text{ref}},\, out_{\text{gen}}`, "paper: outputs of the reference and the generated program"],
      [String.raw`atol,\ rtol`, "paper: absolute and relative tolerance, both 1e-2"],
      [String.raw`a,\ b`, "paper: elements of the two outputs, compared one by one"],
      [String.raw`P(\cdot)`, "paper: over 5 random inputs"],
    ],
    note: "The paper's element-wise allclose is computed by allclose_correctness.allclose_rate.",
    read: "The share of suggestions whose output matched the original.",
  };

  HELP["allclose_correctness.allclose_rate"] = {
    source: "KernelBench: Correctness (fast_0), as in the paper",
    tex: String.raw`\mathrm{allclose\_rate} = P\bigl(\mathrm{allclose}(out_{\text{ref}}, out_{\text{gen}},\, atol, rtol)\bigr),
      \qquad |a-b| \le atol + rtol\cdot|b|`,
    legend: [
      [String.raw`P(\cdot)`, "the share of suggestions; a failed or timed-out suggestion counts as incorrect"],
      [String.raw`out_{\text{ref}}`, "output of the reference; ours: the original cell's Model"],
      [String.raw`out_{\text{gen}}`, "output of the generated program; ours: the suggestion's Model"],
      [String.raw`atol,\ rtol`, "absolute and relative tolerance; paper: both 1e-2"],
      [String.raw`a,\ b`, "elements of out_gen and out_ref, compared one by one"],
      [String.raw`\mathrm{allclose}`, "holds on every one of the random inputs; paper: 5"],
    ],
    read: "The share of suggestions whose every output element is within tolerance.",
  };

  for (const [value, threshold] of [
    ["fast_0", "0"], ["fast_1", "1"], ["fast_1_5", "1.5"], ["fast_2", "2"],
  ]) {
    HELP[`fast_p.${value}`] = {
      source: "KernelBench: fast_p",
      tex: String.raw`\mathrm{fast}_{${threshold}} = P\bigl(\text{correct} \wedge \text{speedup} > ${threshold}\bigr)
        = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\Bigl[\text{correct}_i \wedge
        \frac{T(\text{eager Model})}{T(\text{generated}_i)} > ${threshold}\Bigr]`,
      legend: [
        [String.raw`\text{correct}`, "allclose-correct, as in KernelBench Correctness"],
        [String.raw`\text{speedup}`, "T(eager Model) / T(generated)"],
        [String.raw`\text{eager Model}`, "the PyTorch Model in default eager mode; ours: the original cell"],
        SYMBOLS.runtime, SYMBOLS.generated,
        [String.raw`N`, "every suggestion, failed ones included (suggestions)"],
        SYMBOLS.indicator,
      ],
      read: value === "fast_0"
        ? "The correctness rate."
        : `The share of suggestions that are correct and more than ${threshold}x faster than eager.`,
    };
  }

  HELP["compile_speedup.geometric_mean_compile_speedup"] = {
    source: "KernelBench: Speedup vs torch.compile",
    tex: String.raw`\mathrm{geometric\_mean\_compile\_speedup} =
      ${AGGREGATES.geometric("C", "|C|",
        String.raw`\frac{T(\text{torch.compile Model})}{T(\text{generated}_i)}`)}`,
    legend: [
      SYMBOLS.runtime,
      [String.raw`\text{torch.compile Model}`, "the original Model compiled with torch.compile"],
      SYMBOLS.generated, SYMBOLS.correct, SYMBOLS.correctCount, SYMBOLS.geometric,
    ],
    read: "> 1: the rewrite beats the optimisation PyTorch offers in one line.",
  };

  HELP["compile_speedup.geometric_mean_eager_speedup"] = {
    source: "KernelBench: Speedup vs PyTorch Eager, Geometric mean speedup",
    tex: String.raw`\mathrm{geometric\_mean\_eager\_speedup} =
      ${AGGREGATES.geometric("C", "|C|",
        String.raw`\frac{T(\text{eager Model})}{T(\text{generated}_i)}`)}`,
    legend: [
      SYMBOLS.runtime,
      [String.raw`\text{eager Model}`, "the PyTorch Model in default eager mode; ours: the original cell"],
      SYMBOLS.generated, SYMBOLS.correct, SYMBOLS.correctCount, SYMBOLS.geometric,
    ],
    read: "> 1: faster than eager, timed per call.",
  };

  // -- popup --------------------------------------------------------------

  const popup = document.createElement("div");
  popup.className = "metric-popup";
  popup.id = "metric-popup";
  popup.setAttribute("role", "tooltip");
  popup.hidden = true;

  let anchor = null;
  let pinned = false;
  let hideTimer = null;
  const typeset = new Map();

  function node(tag, className, children = []) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    for (const child of [].concat(children)) {
      if (child === null || child === undefined) continue;
      element.append(child.nodeType ? child : document.createTextNode(child));
    }
    return element;
  }

  function groupOf(key) {
    return REFERENCE_GROUPS.find((one) => one.id === referenceOf(key));
  }

  function content(key, name) {
    if (typeset.has(key)) return typeset.get(key).cloneNode(true);
    const help = HELP[key];
    const formula = (tex) => node("div", "formula", `\\[${tex}\\]`);
    return node("div", "", [
      node("header", "", [
        node("span", "reference-chip", groupOf(key).title),
        node("div", "popup-name", name || key),
        node("div", "popup-key", key),
        node("div", "popup-source", help.source),
      ]),
      help.paperTex ? node("div", "popup-label", "ours") : null,
      formula(help.tex),
      help.paperTex ? node("div", "popup-label", "paper (What it is)") : null,
      help.paperTex ? formula(help.paperTex) : null,
      node("div", "popup-label", "notation"),
      node("dl", "", help.legend.flatMap(([symbol, meaning]) => [
        node("dt", "", `\\(${symbol}\\)`), node("dd", "", meaning),
      ])),
      node("p", "popup-read", help.read),
      help.note ? node("p", "popup-note", help.note) : null,
    ]);
  }

  function place(target) {
    const box = target.getBoundingClientRect();
    const margin = 12;
    const width = popup.offsetWidth;
    const height = popup.offsetHeight;
    const fitsRight = box.right + 10 + width <= innerWidth - margin;
    let left = fitsRight ? box.right + 10 : box.left;
    left = Math.max(margin, Math.min(left, innerWidth - width - margin));
    let top = fitsRight ? box.top - 8 : box.bottom + 6;
    if (top + height > innerHeight - margin) top = Math.max(margin, box.top - height - 6);
    // Whole pixels: a fractional offset blurs the text and the SVG glyphs.
    popup.style.left = `${Math.round(left)}px`;
    popup.style.top = `${Math.round(Math.max(margin, top))}px`;
  }

  function show(target) {
    clearTimeout(hideTimer);
    const key = target.dataset.metric;
    if (!HELP[key]) return;
    if (anchor === target && !popup.hidden) return;
    anchor = target;
    if (!popup.isConnected) document.body.append(popup);
    popup.className = `metric-popup reference-${referenceOf(key)}`;
    const body = content(key, target.textContent);
    popup.replaceChildren(body);
    popup.scrollTop = 0;
    popup.hidden = false;
    place(target);
    if (!typeset.has(key) && window.MathJax && MathJax.typesetPromise) {
      MathJax.startup.promise
        .then(() => MathJax.typesetPromise([body]))
        .then(() => {
          typeset.set(key, body.cloneNode(true));
          if (anchor === target) place(target);
        })
        .catch(() => {});
    }
  }

  function hide(now = false) {
    clearTimeout(hideTimer);
    const close = () => { popup.hidden = true; anchor = null; pinned = false; };
    if (now) close();
    // A short grace, so the pointer can travel from the name into the popup
    // to scroll it.
    else hideTimer = setTimeout(() => { if (!pinned) close(); }, 220);
  }

  document.addEventListener("mouseover", (event) => {
    if (document.querySelector(".metric-table.resizing")) return;
    const target = event.target.closest(".metric-trigger");
    if (target) { pinned = false; show(target); return; }
    if (event.target.closest("#metric-popup")) clearTimeout(hideTimer);
  });
  document.addEventListener("mouseout", (event) => {
    const from = event.target.closest(".metric-trigger, #metric-popup");
    const to = event.relatedTarget && event.relatedTarget.closest
      ? event.relatedTarget.closest(".metric-trigger, #metric-popup") : null;
    if (from && !to && !pinned) hide();
  });
  document.addEventListener("focusin", (event) => {
    const target = event.target.closest(".metric-trigger");
    if (target) show(target);
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest(".metric-trigger");
    if (target) {
      // A touch screen has no hover: a tap pins the popup open.
      pinned = !(pinned && anchor === target);
      if (pinned) show(target); else hide(true);
      return;
    }
    if (!event.target.closest("#metric-popup")) hide(true);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hide(true);
  });
  window.addEventListener("scroll", () => {
    if (anchor && !popup.hidden) place(anchor);
  }, true);
  window.addEventListener("resize", () => {
    if (anchor && !popup.hidden) place(anchor);
  });

  return {
    groups: REFERENCE_GROUPS,
    referenceOf,
    has: (key) => Boolean(HELP[key]),
    hide,
  };
})();
