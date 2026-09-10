# Rubric: evidence coverage

One question: **which of the reference facts below does the analysis in
`output/analysis.md` actually state?**

Nothing else. Do not weight them, do not judge whether the analysis could have
known them, do not score anything. The harness does the arithmetic, twice,
with two different denominators - which is why it must not be done here.

## The facts

{% for fact in reference_facts %}
- `{{ fact.id }}`{% if fact.tolerance %} (tolerance {{ fact.tolerance }}){% endif %}: {{ fact.fact }}
{% endfor %}

## How to decide

- **covered** - the analysis states the fact, in its own words. A number
  counts as stated if it is within the fact's tolerance where one is given.
- **partially covered** - the analysis gestures at it without committing:
  names the phenomenon but not its role, or states it hedged into meaning
  nothing.
- neither - list it in neither array.

Put ids in `covered_fact_ids` and `partially_covered_fact_ids`. An id may not
appear in both.
{% if disabled_sources %}

For information, and not to be acted on: this run had
`{{ disabled_sources | join('`, `') }}` withheld. Do **not** give credit for a
fact the analysis could not have known - "could not have known" is exactly
what the harness computes from the two denominators, and giving credit here
would erase the difference.
{% endif %}
