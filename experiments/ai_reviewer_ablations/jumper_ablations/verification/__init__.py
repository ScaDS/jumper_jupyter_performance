"""Re-measuring suggestions the way the source papers measured them.

The reviewer's own benchmark times the whole cell and compares statistical
fingerprints. KernelBench and ECCO measure something narrower - one call of a
model, one run of a program - against references the reviewer never sees: the
model under ``torch.compile``, the human fast solution, every CodeNet
submission. ``cli.verify`` produces those numbers after a run, without asking
the model anything, and the metrics that reproduce the papers read them.
"""
