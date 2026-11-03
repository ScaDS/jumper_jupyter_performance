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

  // -- metrics only we report ---------------------------------------------

  // A run-scope metric is computed once per reviewer invocation and the
  // table shows the mean over them; a cell-scope one is computed once over
  // all generations of the cell.
  const PER_INVOCATION = [String.raw`R`,
    "the judged reviewer invocations of the cell; the value is computed for each and averaged"];
  const SUGGESTIONS = [String.raw`N`, "every suggestion of every generation in the cell"];
  const GENERATIONS = [String.raw`G`, "the generations of the cell"];
  const ATTEMPTS = [String.raw`a_i`, "code versions suggestion i needed; 1 means no repair"];
  const RAN = [String.raw`\text{ran}_i`, "suggestion i finished without an error or a timeout"];

  function perInvocation(name, term) {
    return String.raw`\mathrm{${name.replaceAll("_", "\\_")}} = \frac{1}{R}\sum_{r=1}^{R} ${term}`;
  }

  function share(name, numerator, denominator) {
    return String.raw`\mathrm{${name.replaceAll("_", "\\_")}} = \frac{${numerator}}{${denominator}}`;
  }

  // Precision and relevance: shares of the claims the judge counted.
  const CLAIMS = [String.raw`\text{claims}_r`, "the claims the judge counted in analysis r"];
  for (const [key, counted, meaning, read] of [
    ["precision.supported_claim_precision", "supported",
      "claims supported by the sources in the packet", "Higher is better."],
    ["precision.contradiction_rate", "contradicted",
      "claims the sources contradict", "Lower is better."],
    ["precision.hallucination_rate", "hallucinated",
      "claims with nothing in the sources behind them", "Lower is better."],
    ["relevance_conciseness.relevant_claim_fraction", "relevant",
      "claims about the optimisation decision", "Higher is better."],
  ]) {
    HELP[key] = {
      source: "Ours, judged",
      tex: perInvocation(key.split(".")[1],
        String.raw`\frac{\text{${counted}}_r}{\text{claims}_r}`),
      legend: [
        PER_INVOCATION, CLAIMS,
        [String.raw`\text{${counted}}_r`, meaning],
      ],
      note: "A claim that is true of the world but not supported by the sources counts as unsupported.",
      read,
    };
  }

  const BOTTLENECK_PARTS = {
    resource_score: ["\\text{resource}_r", "the resource the analysis named"],
    code_region_score: ["\\text{region}_r", "the place in the code it named"],
    causal_explanation_score: ["\\text{cause}_r", "why it called that the bottleneck"],
  };
  for (const [value, [symbol, meaning]] of Object.entries(BOTTLENECK_PARTS)) {
    HELP[`bottleneck_identification.${value}`] = {
      source: "Ours, judged",
      tex: perInvocation(value, symbol),
      legend: [
        PER_INVOCATION,
        [symbol, `the judge's score for ${meaning}: 0 wrong, 1 partly right, 2 right`],
      ],
      read: "From 0 to 2. Higher is better.",
    };
  }
  const BOTTLENECK_LEGEND = Object.values(BOTTLENECK_PARTS).map(([symbol, meaning]) =>
    [symbol, `score for ${meaning}: 0, 1 or 2`]);
  HELP["bottleneck_identification.total_score"] = {
    source: "Ours, judged",
    tex: perInvocation("total_score",
      String.raw`\bigl(\text{resource}_r + \text{region}_r + \text{cause}_r\bigr)`),
    legend: [PER_INVOCATION, ...BOTTLENECK_LEGEND],
    read: "From 0 to 6. Higher is better.",
  };
  HELP["bottleneck_identification.normalised_score"] = {
    source: "Ours, judged",
    tex: perInvocation("normalised_score",
      String.raw`\frac{\text{resource}_r + \text{region}_r + \text{cause}_r}{3 \cdot 2}`),
    legend: [PER_INVOCATION, ...BOTTLENECK_LEGEND],
    read: "From 0 to 1. Higher is better.",
  };

  const COVERAGE_TERM = String.raw`\frac{\sum_{f \in F} w_f\, c_{r,f}}{\sum_{f \in F} w_f}`;
  const COVERAGE_LEGEND = [
    [String.raw`w_f`, "the weight of reference fact f, from the usecase manifest"],
    [String.raw`c_{r,f}`, "1 if analysis r states fact f, 0.5 if partly, 0 if not, as the judge decided"],
  ];
  HELP["conditional_evidence_coverage.weighted_recall"] = {
    source: "Ours, judged",
    tex: perInvocation("weighted_recall", COVERAGE_TERM),
    legend: [
      PER_INVOCATION,
      [String.raw`F`, "the reference facts whose source this preset still had switched on"],
      ...COVERAGE_LEGEND,
    ],
    read: "Did the model use what it was given. Higher is better.",
  };
  HELP["global_evidence_coverage.weighted_recall"] = {
    source: "Ours, judged",
    tex: perInvocation("weighted_recall", COVERAGE_TERM),
    legend: [
      PER_INVOCATION,
      [String.raw`F`, "every reference fact of the usecase, for every preset alike"],
      ...COVERAGE_LEGEND,
    ],
    read: "What was lost by removing a source. Higher is better.",
  };

  const TRANSFER_TOTAL = [String.raw`\text{suggestions}_r`, "the suggestions of invocation r"];
  HELP["analysis_to_suggestion_transfer.bottleneck_aligned_rate"] = {
    source: "Ours, judged",
    tex: perInvocation("bottleneck_aligned_rate",
      String.raw`\frac{\text{aligned}_r}{\text{suggestions}_r}`),
    legend: [PER_INVOCATION, TRANSFER_TOTAL,
      [String.raw`\text{aligned}_r`, "suggestions that address the bottleneck the analysis named"]],
    read: "Higher is better.",
  };
  HELP["analysis_to_suggestion_transfer.fully_constrained_suggestion_rate"] = {
    source: "Ours, judged",
    tex: perInvocation("fully_constrained_suggestion_rate",
      String.raw`\frac{\text{preserving all}_r}{\text{suggestions}_r}`),
    legend: [PER_INVOCATION, TRANSFER_TOTAL,
      [String.raw`\text{preserving all}_r`,
        "suggestions that keep every constraint the analysis stated; breaking one is enough to fail"]],
    read: "Higher is better.",
  };
  HELP["analysis_to_suggestion_transfer.preserved_constraint_rate"] = {
    source: "Ours, judged",
    tex: perInvocation("preserved_constraint_rate",
      String.raw`\frac{\text{preserved}_r}{\text{observations}_r}`),
    legend: [PER_INVOCATION,
      [String.raw`\text{observations}_r`, "pairs of a suggestion and a stated constraint the judge checked"],
      [String.raw`\text{preserved}_r`, "those pairs where the suggestion kept the constraint"]],
    read: "Higher is better.",
  };

  for (const [value, set, meaning, read] of [
    ["differs_rate", String.raw`\text{differs}`,
      "suggestions that ran but whose output differs from the original", "Lower is better."],
    ["unverified_rate", String.raw`\text{unverified}`,
      "suggestions that ran but whose output could not be compared", "Lower is better."],
    ["failed_rate", String.raw`\neg\,\text{ran}`,
      "suggestions that did not finish: an error or a timeout", "Lower is better."],
  ]) {
    HELP[`correctness_rates.${value}`] = {
      source: "Ours",
      tex: share(value, String.raw`\bigl|\{\, i : ${set}_i \,\}\bigr|`, "N"),
      legend: [SUGGESTIONS, [String.raw`${set}_i`, meaning]],
      note: "verified, differs, unverified and failed add up to 1.",
      read,
    };
  }

  HELP["execution_success.pre_repair_success_rate"] = {
    source: "Ours",
    tex: share("pre_repair_success_rate",
      String.raw`\bigl|\{\, i : \text{ran}_i \wedge a_i \le 1 \,\}\bigr|`, "N"),
    legend: [SUGGESTIONS, RAN, ATTEMPTS],
    read: "Ran as the model first wrote it.",
  };
  HELP["execution_success.post_repair_success_rate"] = {
    source: "Ours",
    tex: share("post_repair_success_rate", String.raw`\bigl|\{\, i : \text{ran}_i \,\}\bigr|`, "N"),
    legend: [SUGGESTIONS, RAN],
    read: "Ran at all, after up to three repairs.",
  };
  HELP["execution_success.repair_uplift"] = {
    source: "Ours",
    tex: String.raw`\mathrm{repair\_uplift} = \mathrm{post\_repair\_success\_rate} - \mathrm{pre\_repair\_success\_rate}`,
    legend: [[String.raw`\mathrm{post}, \mathrm{pre}`, "the two success rates beside it"]],
    read: "How much the repair loop carried. A large uplift flatters the model.",
  };

  HELP["pass_at_k.pass_at_k"] = {
    source: "Ours; ECCO reports pass@1 only",
    tex: String.raw`\mathrm{pass@k} = \frac{1}{G}\sum_{g=1}^{G} \mathbb{1}\bigl[\exists\, i \in g : \text{correct}_i\bigr]`,
    legend: [
      [String.raw`G`, "generations with at least one verdict (generations)"],
      [String.raw`i \in g`, "the suggestions of generation g; k is however many it returned"],
      [String.raw`\text{correct}_i`, "suggestion i ran and is verified"],
      SYMBOLS.indicator,
    ],
    read: "The share of responses with at least one correct option.",
  };

  const RANKED = [String.raw`C_g`,
    "the correct suggestions of generation g with a measured speedup"];
  HELP["rank_quality.first_is_fastest_rate"] = {
    source: "Ours",
    tex: String.raw`\mathrm{first\_is\_fastest\_rate} = \frac{1}{|G'|}\sum_{g \in G'}
      \mathbb{1}\Bigl[\text{position}_{\arg\max_{i \in C_g} s_i} = 0\Bigr]`,
    legend: [
      [String.raw`G'`, "generations with at least one correct, timed suggestion"],
      RANKED,
      SYMBOLS.speedup,
      [String.raw`\text{position}_i`, "where the model ranked suggestion i, 0 for its first"],
      SYMBOLS.indicator,
    ],
    read: "Higher: the model's first option is its best one.",
  };
  HELP["rank_quality.mean_rank_correlation"] = {
    source: "Ours",
    tex: String.raw`\mathrm{mean\_rank\_correlation} = \frac{1}{|G''|}\sum_{g \in G''}
      \rho\bigl(\,(\text{position}_i)_{i \in C_g},\; (-s_i)_{i \in C_g}\bigr)`,
    legend: [
      [String.raw`G''`, "generations with at least two correct, timed suggestions"],
      RANKED,
      [String.raw`\text{position}_i`, "where the model ranked suggestion i, 0 for its first"],
      SYMBOLS.speedup,
      [String.raw`\rho`, "Spearman rank correlation"],
    ],
    read: "From -1 to 1. 1: the proposed order is the measured order.",
  };

  HELP["repairs.attempts_per_suggestion"] = {
    source: "Ours",
    tex: String.raw`\mathrm{attempts\_per\_suggestion} = \frac{1}{N}\sum_{i=1}^{N} \max(a_i - 1,\, 0)`,
    legend: [SUGGESTIONS, ATTEMPTS],
    read: "Repairs per suggestion. Lower is better.",
  };
  HELP["repairs.repaired_share"] = {
    source: "Ours",
    tex: share("repaired_share", String.raw`\bigl|\{\, i : a_i > 1 \,\}\bigr|`, "N"),
    legend: [SUGGESTIONS, ATTEMPTS],
    read: "The share that needed the repair loop. Lower is better.",
  };
  HELP["repairs.repair_success_rate"] = {
    source: "Ours",
    tex: share("repair_success_rate",
      String.raw`\bigl|\{\, i : a_i > 1 \wedge \text{ran}_i \,\}\bigr|`,
      String.raw`\bigl|\{\, i : a_i > 1 \,\}\bigr|`),
    legend: [ATTEMPTS, RAN],
    read: "Of the repaired suggestions, the share that ran in the end.",
  };
  HELP["repairs.idea_change_rate"] = {
    source: "Ours",
    tex: share("idea_change_rate",
      String.raw`\bigl|\{\, i \in P : \text{code}^{\text{before}}_i \ne \text{code}^{\text{after}}_i \,\}\bigr|`,
      String.raw`|P|`),
    legend: [
      [String.raw`P`, "repaired suggestions whose code is recorded both before and after the repair"],
      [String.raw`\text{code}^{\text{before}}_i`, "the code as the model wrote it, from the review record"],
      [String.raw`\text{code}^{\text{after}}_i`, "the code after the repair loop, from the benchmark record"],
      [String.raw`\ne`, "differ once formatting is normalised away"],
    ],
    note: "A lower bound on a changed idea: a rename counts as a change too.",
    read: "How often the repair rewrote the suggestion rather than fixed it.",
  };

  const MODE_PAIRS = [String.raw`Q`,
    "the pairs (generation, two replay modes) that scored the same stored suggestions"];
  HELP["replay_mode_agreement.correctness_agreement"] = {
    source: "Ours",
    tex: String.raw`\mathrm{correctness\_agreement} = \frac{1}{|I|}\sum_{(q,\,i) \in I}
      \mathbb{1}\bigl[\text{correctness}^{m}_i = \text{correctness}^{m'}_i\bigr]`,
    legend: [
      MODE_PAIRS,
      [String.raw`I`, "the suggestions i both modes of pair q scored"],
      [String.raw`m,\ m'`, "the two replay modes of pair q"],
      [String.raw`\text{correctness}^{m}_i`, "verified, differs or unverified under mode m"],
      SYMBOLS.indicator,
    ],
    read: "1: the modes agree on every suggestion.",
  };
  HELP["replay_mode_agreement.mean_speedup_dispersion"] = {
    source: "Ours",
    tex: String.raw`\mathrm{mean\_speedup\_dispersion} = \frac{1}{|I|}\sum_{(q,\,i) \in I}
      \frac{\max\bigl(s^{m}_i,\, s^{m'}_i\bigr)}{\min\bigl(s^{m}_i,\, s^{m'}_i\bigr)}`,
    legend: [
      MODE_PAIRS,
      [String.raw`I`, "the suggestions i with a positive speedup under both modes of pair q"],
      [String.raw`s^{m}_i`, "the speedup of suggestion i measured under mode m"],
    ],
    read: "1: both modes measured the same speedup. Higher: they disagree.",
  };
  HELP["replay_mode_agreement.mean_rank_correlation"] = {
    source: "Ours",
    tex: String.raw`\mathrm{mean\_rank\_correlation} = \frac{1}{|Q|}\sum_{q \in Q}
      \rho\bigl(\,(s^{m}_i)_{i},\; (s^{m'}_i)_{i}\bigr)`,
    legend: [
      MODE_PAIRS,
      [String.raw`s^{m}_i`, "the speedup of suggestion i under mode m"],
      [String.raw`\rho`, "Spearman rank correlation over the suggestions both modes scored"],
    ],
    read: "1: both modes order the suggestions alike.",
  };
  HELP["replay_mode_agreement.fallback_rate"] = {
    source: "Ours",
    tex: share("fallback_rate", String.raw`\bigl|\{\, b \in B : \text{degraded}_b \,\}\bigr|`, "|B|"),
    legend: [
      [String.raw`B`, "the benchmark records of the cell"],
      [String.raw`\text{degraded}_b`, "the requested replay mode could not be served and fell back to the full replay"],
    ],
    read: "Lower is better: a fallback is another mode's result under this mode's name.",
  };

  const RETURNED = [String.raw`n_r`, "suggestions review r returned"];
  const REQUESTED = [String.raw`k`, "suggestions requested per review"];
  const REVIEWS = [String.raw`R`, "the reviews of the cell"];
  HELP["requested_vs_returned.mean_returned"] = {
    source: "Ours",
    tex: String.raw`\mathrm{mean\_returned} = \frac{1}{R}\sum_{r=1}^{R} n_r`,
    legend: [REVIEWS, RETURNED],
    read: "Read before the per-suggestion rates: they rest on this many.",
  };
  HELP["requested_vs_returned.mean_shortfall"] = {
    source: "Ours",
    tex: String.raw`\mathrm{mean\_shortfall} = \frac{1}{R}\sum_{r=1}^{R} \max(k - n_r,\, 0)`,
    legend: [REVIEWS, RETURNED, REQUESTED],
    read: "Lower is better.",
  };
  HELP["requested_vs_returned.compliance_rate"] = {
    source: "Ours",
    tex: share("compliance_rate", String.raw`\bigl|\{\, r : n_r \ge k \,\}\bigr|`, "R"),
    legend: [REVIEWS, RETURNED, REQUESTED],
    read: "The share of reviews that returned as many options as asked.",
  };

  HELP["resource_effect_agreement.agreement_rate"] = {
    source: "Ours, judged and measured",
    tex: perInvocation("agreement_rate",
      String.raw`\frac{\bigl|\{\, p \in P_r : d^{\text{measured}}_p = d^{\text{predicted}}_p \,\}\bigr|}{|P_r|}`),
    legend: [
      PER_INVOCATION,
      [String.raw`P_r`, "the judge's predictions for invocation r that could be checked against a measurement"],
      [String.raw`d^{\text{predicted}}_p`, "the direction the suggestion's description implies for a resource: up, down or none"],
      [String.raw`d^{\text{measured}}_p`,
        "the measured direction: the relative change (v - b) / |b| of the resource's mean, " +
        "none within ±10%"],
      [String.raw`b,\ v`, "the resource's mean under the original cell and under the suggestion"],
    ],
    read: "Low: suggestions that worked did not work the way they said.",
  };

  const LABELS = [String.raw`L_r`, "the technique labels the judge gave the suggestions of invocation r, case and spacing ignored"];
  const LABELLED = [String.raw`n_r`, "the suggestions of invocation r"];
  HELP["suggestion_diversity.distinct_technique_rate"] = {
    source: "Ours, judged",
    tex: perInvocation("distinct_technique_rate", String.raw`\frac{|\,\mathrm{set}(L_r)\,|}{n_r}`),
    legend: [PER_INVOCATION, LABELS, LABELLED,
      [String.raw`\mathrm{set}(L_r)`, "the distinct labels"]],
    read: "1: every option is a different idea.",
  };
  HELP["suggestion_diversity.duplicate_rate"] = {
    source: "Ours, judged",
    tex: perInvocation("duplicate_rate", String.raw`\frac{n_r - |\,\mathrm{set}(L_r)\,|}{n_r}`),
    legend: [PER_INVOCATION, LABELS, LABELLED,
      [String.raw`\mathrm{set}(L_r)`, "the distinct labels"]],
    read: "Lower is better: the same idea rephrased.",
  };

  for (const [value, term, meaning, scope] of [
    ["mean_review_latency_s", String.raw`\text{latency}_r`,
      "seconds the model calls of review r took", REVIEWS],
    ["mean_input_tokens", String.raw`\textstyle\sum_{c \in r} \text{input}_c`,
      "input tokens over the model calls c of review r", REVIEWS],
    ["mean_output_tokens", String.raw`\textstyle\sum_{c \in r} \text{output}_c`,
      "output tokens over the model calls c of review r", REVIEWS],
    ["mean_total_tokens", String.raw`\text{tokens}_r`,
      "all tokens of review r", REVIEWS],
    ["mean_benchmark_wall_s", String.raw`\text{wall}_g`,
      "seconds the benchmark of generation g took", GENERATIONS],
    ["mean_repair_calls", String.raw`\text{fix calls}_g`,
      "model calls the repair loop made in generation g", GENERATIONS],
  ]) {
    const over = scope === REVIEWS ? "r=1}^{R" : "g=1}^{G";
    const count = scope === REVIEWS ? "R" : "G";
    HELP[`end_to_end_cost.${value}`] = {
      source: "Ours",
      tex: String.raw`\mathrm{${value.replaceAll("_", "\\_")}} = \frac{1}{${count}}\sum_{${over}} ${term}`,
      legend: [scope, [term, meaning]],
      note: value.includes("tokens") ? "Empty when the model endpoint reports no usage." : null,
      read: "Lower is cheaper. Read next to quality: removing a source removes tokens with it.",
    };
  }

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

  function content(key, name, family) {
    if (typeset.has(key)) return typeset.get(key).cloneNode(true);
    const help = HELP[key];
    const formula = (tex) => node("div", "formula", `\\[${tex}\\]`);
    return node("div", "", [
      node("header", "", [
        node("span", "reference-chip", groupOf(key).title),
        family ? node("div", "popup-source", family) : null,
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
    const body = content(key, target.textContent, target.dataset.family);
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
