# s7 — Encoding pulses on an off-grid target

## Question

Can the encoding pulse scalers reach frequencies the unitary parameters
cannot, and what does that arm owe to which of its parts?

## Method

Every cell trains on an off-grid target (`offgrid_mode=generator`, `mts=4`)
through the `train_enc` flow, at `n_layers=2`.

**364 runs** in two blocks:

- **Ablation, 24 runs:** `Circuit_15`, ternary encoding, ranks off, 3 data
  seeds (1000–1002), and the training changed one thing at a time.

  | arm | changes |
  | --- | --- |
  | B0-floor-unitary | `gate_mode=unitary`, `offgrid_prob=0` (on-grid target) |
  | B0-floor-encpulse | `gate_mode=enc_pulse`, `offgrid_prob=0` (on-grid target) |
  | unitary | `gate_mode=unitary` |
  | encpulse-ones | `gate_mode=enc_pulse` |
  | B2-encpulse-target | `gate_mode=enc_pulse`, `enc_pulse_init=target` (scalers start at the target) |
  | B3-enc-params | `gate_mode=unitary`, `train_enc_params=true` (trainable frequencies instead) |
  | B4-lr1e-2 | `gate_mode=enc_pulse`, `pulse_learning_rate=1e-2` |
  | B6-hamming-encpulse | `gate_mode=enc_pulse`, `encoding_strategy=hamming` |

  The floor arms' target is on the comb, where the unitary parameters already
  reach every component, so whatever the pulse arm gains there is not
  reachability.

- **Ansatz grid, 340 runs:** 17 ansaetze × `gate_mode` ∈ {unitary, enc_pulse}
  × 10 data seeds (1000–1009), `decompose_circuit=false`: every ansatz trained
  from the same target with the encoding at gate level and at pulse level.

## Reproduce

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s7-ablation/run.py --dry-run   # the grid, submitting nothing
uv run python dev/s7-ablation/run.py --jobs 1    # submit it, resuming what finished
uv run python dev/s7-ablation/figures.py         # the ansatz block's figures, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
