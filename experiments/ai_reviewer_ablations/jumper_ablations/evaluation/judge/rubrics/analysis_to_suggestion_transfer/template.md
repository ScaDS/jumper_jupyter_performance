# Rubric: analysis to suggestion transfer

The reviewer's two steps are chained: the analysis is handed to the suggestion
step as the diagnosis to act on. This asks whether that handover worked.

Read `output/analysis.md` and `output/suggestions.json`
({{ suggestion_count }} suggestion{{ '' if suggestion_count == 1 else 's' }}).

## Alignment

`bottleneck_aligned` - how many suggestions address the bottleneck the
analysis named. A suggestion that speeds something else up is not aligned,
however good it is.

## Constraints

`constraints_stated` - how many constraints the analysis committed to.
A constraint is something the analysis says any fix must respect: results must
stay identical, no GPU is available, a library cannot be introduced.

`suggestions_preserving_all` - how many suggestions respect **every** one of
those constraints. A suggestion that keeps three constraints and breaks the
fourth is not counted here: it cannot be applied either. This is the number
the report leads with.

`constraint_observations` - the number of (suggestion, constraint) pairs you
checked. With 2 constraints and 3 suggestions that is 6.

`constraints_preserved` - how many of those pairs the suggestion respects.
This is the finer-grained view, and on its own it flatters: if every
suggestion breaks one of two constraints it reads 50% while nothing is
usable. Fill both in.

If the analysis states no constraints, set `constraints_stated`,
`constraint_observations` and `constraints_preserved` to 0, and set
`suggestions_preserving_all` to `suggestions_total` - with nothing to break,
every suggestion trivially preserves the constraints. The harness reads the
zeroes as "not applicable", not as a failure.
