"""Context-source ablation experiments for the JUmPER AI reviewer.

The experiment answers one question: how much does each source of context the
reviewer feeds its model contribute to the analysis it produces and to the
rewrites it proposes?

Three phases, three commands, deliberately separate. `cli.run` executes the
real ``%perfmonitor_ai_review`` magic in a real kernel and writes down
everything it produced. `cli.evaluate` scores those records. `cli.report`
aggregates the scores. Splitting them is what lets a new metric be added
without paying for the LLM and the benchmark all over again.
"""
