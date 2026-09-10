"""Command line entry points, one per phase of the experiment.

``build_strategies`` renders the ablations into a file the reviewer reads;
``run`` executes the real magic and records what it produced; ``export_judge``
prepares the packets an agent session judges; ``evaluate`` scores the records;
``report`` aggregates the scores into the two metric tables.
"""
