# Rubric: factuality and groundedness

Count claims in `output/analysis.md`. A **claim** is one assertion about the
code, the measurements or the hardware that could be true or false. Restating
the task is not a claim; "the GPU is idle" is.

For each claim, decide against `sources/` only:

- **supported** - the sources state it, or it follows directly from them.
- **contradicted** - the sources say otherwise.
- **hallucinated** - the sources contain nothing about it. Code that is not in
  the given source, numbers that appear nowhere in the payload, hardware that
  was not described.

A claim can be true in general and still be hallucinated here. The question is
whether the model reasoned from its evidence, not whether it guessed well.
{% if disabled_sources %}

Withheld from this run: `{{ disabled_sources | join('`, `') }}`. A confident
claim about a withheld source is a hallucination, and catching that is a large
part of why this metric exists.
{% endif %}

Fill in `total_claims`, `supported_claims`, `contradictions`,
`hallucinations`. The three categories are exclusive; a claim counted as
contradicted is not also counted as hallucinated. Put two or three of the
clearest cases in `examples`, quoted.
