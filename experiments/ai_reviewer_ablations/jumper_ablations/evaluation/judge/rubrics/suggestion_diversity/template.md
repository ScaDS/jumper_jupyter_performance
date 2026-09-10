# Rubric: suggestion diversity

Read `output/suggestions.json`. Give each suggestion **one short label**
naming the optimization technique it applies - "vectorise the loop", "cache
the invariant", "batch the compute calls", "parallelise over workers".

Two suggestions get the **same** label exactly when they are the same idea,
regardless of how differently the code is written. Two suggestions get
different labels when a user choosing between them would be choosing between
genuinely different approaches.

Put the labels in `technique_labels`, in suggestion order, and the number of
suggestions in `suggestions_total`. The harness counts the distinct ones.

This is judged rather than diffed because two rewrites can share almost no
source text and still be one idea, and can differ by a line and be two.
