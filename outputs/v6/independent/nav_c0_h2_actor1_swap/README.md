# Post-hoc seed-0 h2 + actor1 closed-loop diagnostic

Status: complete, read-only diagnostic.  Not a C0/C1 result.

## Protocol

- actor: fixed final Stage-B actor1 seed 0
  (`outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth`)
- h: C0 h2 seed 0 (`outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl`)
  instead of h1
- 100 deterministic episodes, eval seeds `10000..10099`
- same environment/config as the formal Stage-B evaluation; the gate is not
  active during evaluation
- command/log:
  `outputs/v6/independent/nav_c0_h2_actor1_swap/seed0_100ep.log`

This is an off-policy h-swap: actor1 was trained with h1.  The diagnostic asks
whether the offline risk-region degradation translates to closed-loop
behavior; it does not identify the effect of h2 on a freshly trained actor2.

## Result

| metric | actor1 + h1 | actor1 + h2 | delta |
|---|---:|---:|---:|
| safe_episode | 0.840 | 0.900 | +0.060 |
| collision_episode | 0.160 | 0.100 | -0.060 |
| official_safety | 0.97375 | 0.98500 | +0.01125 |
| official_success | 0.42750 | 0.43875 | +0.01125 |
| official_reach | 0.43500 | 0.44625 | +0.01125 |
| episode_return | 60.91 | 61.40 | +0.49 |
| warning_episode | 0.210 | 0.110 | -0.100 |
| h_mean | 0.836 | 0.930 | +0.093 |
| h_min | 0.648 | 0.762 | +0.114 |
| cbf_violation | 0.0446 | 0.0381 | -0.0065 |

The differences are small and one-seed: safe episode +6 pp is about
`z = 1.27` under a binomial approximation, official safety/success/reach
+1.1 pp, return +0.49.  They are not statistically separated.

## Interpretation

- The closed-loop seed-0 diagnostic does **not** reproduce the offline
  17-32-step risk-region degradation.  If anything, actor1 + h2 is slightly
  safer on this seed.
- The swap is off-policy.  h2 shifts the actor's h input upward
  (`h_mean +0.093`, `h_min +0.114`).  A higher h can make actor1 more
  conservative; that is an input-distribution side effect, not evidence that
  h2 is a better-calibrated safety value.
- Offline 17-32 MAE weights states uniformly; closed-loop safety depends on
  the actor's response.  The two can diverge.  This is a limitation of using
  offline calibration alone as the C0 gate, and it is why the result is
  recorded as a diagnostic rather than a replacement criterion.
- The pre-registered C0a rule remains failed.  C0b and C1 remain
  unauthorized.  Do not use this diagnostic to promote h2 or to authorize
  actor2.  Pursuing h2 further would require a new pre-registration with a
  multi-seed closed-loop criterion and explicit approval.

## Artifacts

- `seed0_100ep.json` — full metrics
- `seed0_100ep.log` — evaluation log
- baseline: `../nav_stageb_repair_16k/seed0/eval_100ep_final.json`
