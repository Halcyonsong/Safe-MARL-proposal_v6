# Partial Ant 10M run (stopped, do not resume)

This directory holds the upstream `v6_lag_10m_seed0` output that was stopped at
2026-10-27 ~11:06 CST, shortly after launch, so that the Nav P0/P1 ablations
could run first.

- The run is **incomplete** (about 8 minutes of a planned 10M-step run).
- It must **not** be resumed, evaluated, or reported as an Ant baseline.
- The Ant 10M external baseline is **postponed** by user decision.
- A future restart must use a fresh `EXPERIMENT` name; the runner refuses to
  reuse an existing upstream experiment directory by design.
