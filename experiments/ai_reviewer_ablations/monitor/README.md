# Run monitor

A read-only web view of an ablation run. It reads files and asks the queue;
it never writes to a run, and it does not import the experiment package.
`PROTOCOL.md` is the contract it relies on.

## Running it

On the cluster, from the experiment root:

```bash
python -m monitor --port 8765
```

It binds to loopback only. From your own machine:

```bash
ssh -N -L 8765:localhost:8765 <user>@login2.romeo.hpc.tu-dresden.de
```

then open <http://localhost:8765/>. Rendering happens in your browser; only
JSON crosses the network.

`--results <dir>` points it elsewhere; the default follows
`JUMPER_ABLATION_STORAGE`, then the experiment's own `results/`.

## What it shows

| Tab | |
|---|---|
| Inputs | Which context sources each preset had, the usecases with their reference facts, and the files the run used |
| Shards | The pass grid, coloured by state or by the job that owns it, and a row per shard with its queue state |
| Results | Where the output went, a comparison across runs, the metric table, and judge coverage |

Progress comes from the records rather than from the pass index: a pass
writes its verdict only when it ends, which under a full replay is hours
away, so the index cannot say whether anything is happening.

## Why it is a web page and not a desktop one

The cluster's Python has no `_tkinter`, and a desktop toolkit over X11 draws
on the login node and ships pixels over the network. A page renders on your
machine and needs nothing installed there.
