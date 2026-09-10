"""Analysis metrics: what the reviewer concluded before proposing anything.

Every metric here is judged rather than computed. What makes an analysis good
- naming the right resource, grounding a claim in the telemetry, not wandering
off the question - has no closed form, and the harness does not pretend
otherwise. Each module declares its scale and the shape of the verdict; the
judging itself happens in an agent session, over the same sources the reviewer
had. See JUDGE_PROTOCOL.md.
"""
