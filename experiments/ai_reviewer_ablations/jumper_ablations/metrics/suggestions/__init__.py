"""Suggestion metrics: what the proposed rewrites turned out to be worth.

Most of these are computed, not judged, because the reviewer already measured
them: it ran every suggestion, repaired what failed and compared results
against the baseline. What is left here is arithmetic over those verdicts, and
the arithmetic is where the honesty lives - `unverified` never counts as
correct, and a speedup is always reported next to the coverage it was computed
over.
"""
