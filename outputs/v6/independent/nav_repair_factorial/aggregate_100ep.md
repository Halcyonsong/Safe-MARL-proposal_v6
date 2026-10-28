# Nav P0 + P1 factorial aggregate

100 deterministic episodes per seed, eval scenes 10000..10099; mean ± sample std (n-1) over seeds 0/1/2.

| row | h checkpoint | actor h input | gate | complete | safe_ep | collision_ep | official_safety | success | reach | return |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| H0 | original v6_h.pkl | on | off | yes | 0.4500 ± 0.1931 | 0.5500 ± 0.1931 | 0.8783 ± 0.0659 | 0.4067 ± 0.1316 | 0.4450 ± 0.1332 | 61.7286 ± 10.6393 |
| P0 | original v6_h.pkl | on | on | yes | 0.6433 ± 0.1343 | 0.3567 ± 0.1343 | 0.9325 ± 0.0295 | 0.4263 ± 0.0557 | 0.4525 ± 0.0692 | 70.3253 ± 9.1697 |
| A | repaired | on | on | yes | 0.6933 ± 0.1680 | 0.3067 ± 0.1680 | 0.9417 ± 0.0378 | 0.4100 ± 0.0583 | 0.4296 ± 0.0695 | 66.8545 ± 5.2637 |
| B | repaired | on | off | yes | 0.6333 ± 0.2401 | 0.3667 ± 0.2401 | 0.9267 ± 0.0579 | 0.3229 ± 0.1708 | 0.3517 ± 0.1970 | 55.6748 ± 23.3537 |
| C | repaired | off | on | yes | 0.0200 ± 0.0346 | 0.9800 ± 0.0346 | 0.4987 ± 0.2087 | 0.0367 ± 0.0182 | 0.1008 ± 0.0763 | -6.6314 ± 46.7006 |
| D | repaired | off | off | yes | 0.0333 ± 0.0351 | 0.9667 ± 0.0351 | 0.5479 ± 0.0382 | 0.0367 ± 0.0038 | 0.0708 ± 0.0105 | 7.2091 ± 38.5127 |

Row definitions:

- **H0**: original `v6_h.pkl`, actor h input on, gate off (existing formal h-only reference).
- **P0**: original `v6_h.pkl`, actor h input on, broad gate (high `0.13`).
- **A**: repaired h, actor h input on, gate high `0.26727` (existing `nav_stageb_repair_16k`).
- **B**: repaired h, actor h input on, gate off.
- **C**: repaired h frozen for the gate, actor h/grad channels zeroed, gate on.
- **D**: repaired h frozen for the gate, actor h/grad channels zeroed, gate off.

Caveats:

- **H0 vs B is the clean h-retrain effect with the gate off** (both use policy h input and no training-time gate): `+18.3 pp` safe but `-8.4 pp` success and `-6.1` return, so repaired h alone is a safety/task trade.
- **C/D collapse**: removing policy h input gives safe `0.020/0.033` and success `0.037` in both cells across all six seeds; D is consistent with the matched no-h fixed-penalty baseline.  The gate cannot compensate (C - D safe `-1.3 pp`, success `0.0 pp`).
- P0 differs from A in h checkpoint, gate threshold, and gate scope; it completes the original-h broad-gate row, not a single factor of the repaired-h 2x2.  P0 vs H0 isolates the broad-gate effect on the original h only.
- The gate is off at evaluation, so gate ON/OFF is a training-time reward-shaping difference, not an inference-time filter.
- B vs D isolates the actor h-input effect with the gate off; A vs C isolates it with the gate on; A vs B and C vs D isolate the gate effect at each h-input level.
