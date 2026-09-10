# Rubric: bottleneck identification

Read `output/analysis.md` and score three things separately. They fail
separately, and collapsing them would hide what this experiment is measuring.

Score each **0**, **1** or **2**.

## 1. Resource (`resource_score`)

Which resource does the analysis name as the limit?

- **2** - names the resource that the sources actually show as the limit, and
  names it unambiguously.
- **1** - names a resource that is plausibly involved but not the limit, or
  hedges across several without committing.
- **0** - names the wrong resource, or names none.

## 2. Code region (`code_region_score`)

Which part of the code does it hold responsible?

- **2** - points at the specific construct that dominates the work, precisely
  enough that a reader would know which lines to change.
- **1** - points at the right general area but not the operation, or names
  several candidates without ranking them.
- **0** - points at the wrong code, or at none.

## 3. Causal explanation (`causal_explanation_score`)

Does it explain *why* that code produces that resource behaviour?

- **2** - gives a mechanism, and the mechanism is consistent with the sources.
- **1** - asserts a link without a mechanism, or gives a mechanism that only
  partly follows from the sources.
- **0** - no explanation, or one the sources contradict.

## What counts as evidence

Only `sources/`. Judge against what the model was given, not against what you
know about the library.
{% if reference_facts %}

The usecase's reference facts, for what "the limit" and "the responsible code"
mean here:
{% for fact in reference_facts %}
- `{{ fact.id }}` (from `{{ fact.source }}`, weight {{ fact.weight }}): {{ fact.fact }}
{% endfor %}
{% endif %}
{% if disabled_sources %}

Some sources were withheld from this run: `{{ disabled_sources | join('`, `') }}`.
That is the experiment, not a defect. Score what the analysis did with what it
had; a fact it could not have known is not a point lost here - the coverage
rubric is where that is measured.
{% endif %}

Quote what the analysis actually named in `named_resource` and
`named_code_region` so the scores can be checked against it.
