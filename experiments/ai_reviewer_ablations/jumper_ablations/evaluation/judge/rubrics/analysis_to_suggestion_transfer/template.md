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

`constraint_observations` - the number of (suggestion, constraint) pairs you
checked. With 2 constraints and 3 suggestions that is 6.

`constraints_preserved` - how many of those pairs the suggestion respects.

If the analysis states no constraints, set `constraints_stated`,
`constraint_observations` and `constraints_preserved` to 0. The harness reads
that as "not applicable", not as a failure.
