"""From one number per unit to one row per claim.

A metric produces a value per record or per grid cell. What the experiment
reports is narrower and harder: a point estimate per preset, an interval
around it, and the difference from the full-context baseline - paired by
generation, because generation 3 of two presets was asked for under the same
conditions and comparing them in bulk throws that away.
"""
from jumper_ablations.aggregate.summary import summarise

__all__ = ["summarise"]
