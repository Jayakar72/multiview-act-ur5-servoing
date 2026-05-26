# Training Configs

AIC engine YAML configs used during ACT demonstration collection and evaluation. Each config seeds the eval engine with a deterministic distribution over rail / port-type / yaw / translation, so a collection run produces demos that match a known scene profile.

## Top-level configs

| File | Use |
|------|-----|
| `sample_config.yaml` | Default eval config (matches the portal's sampling). Use for unbiased data collection or eval. |
| `test_config.yaml` | Lightweight version of sample_config, fewer trials per run. Useful for fast iteration during development. |
| `multi_nic_test.yaml` | Spawns multiple NICs simultaneously to exercise the multi-NIC disambiguation logic (see [`../../docs/multi_view_perception.md`](../../docs/multi_view_perception.md)). |
| `custom_verify_v14.yaml` | A specific scene we kept around for verifying behavior across policy revisions. Not necessary for new contributors. |

## `diverse/` — per-rail and per-port-pair configs

These are the configs we used to **weight data collection toward the rails the eval portal samples most heavily**, while still maintaining coverage on the long-tail rails.

| File | Spawns |
|------|--------|
| `config_rail0.yaml` | NICs on rail 0 only |
| `config_rail1.yaml` | NICs on rail 1 only |
| `config_rail2.yaml` | NICs on rail 2 only |
| `config_rail3.yaml` | NICs on rail 3 only |
| `config_rail4.yaml` | NICs on rail 4 only |
| `config_rail3_4.yaml` | Mixed rail 3 + 4 |
| `config_rail2_port01.yaml` | Rail 2, ports 0 and 1 explicitly targeted |
| `config_rail34_port01.yaml` | Rails 3+4, ports 0 and 1 |
| `config_0_port01.yaml` | Rail 0, ports 0 and 1 |
| `config_1_port01.yaml` | Rail 1, ports 0 and 1 |
| `config_test_ood.yaml` | **Out-of-distribution stress test** — board placements and rail configurations the portal doesn't typically sample. Use for robustness evaluation. |
| `stress_config_hard.yaml` | High-randomization config: extreme yaw rotations, edge-of-range translations, multiple NICs. Use to validate that your policy doesn't overfit. |

## How they're used

For ACT data collection:

```bash
bash ../../data_collection/auto_collect.sh \
  --config diverse/config_rail0.yaml \
  --target 80
```

For YOLO-OBB data collection (with CheatCode as the driver):

```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true start_aic_engine:=true \
  aic_engine_config_file:=$PWD/diverse/config_rail0.yaml

# Terminal 2 (CheatCode)
# Terminal 3 (collect_port_data.py)
```

See [`../../data_collection/README.md`](../../data_collection/README.md) for the full sequence.

## The 302-episode distribution

Our final ACT dataset used these configs in roughly this proportion:

| Config | Target demos | Why |
|--------|------------:|-----|
| `diverse/config_rail0.yaml` | 77 | Rail 0 heavily sampled by eval portal |
| `diverse/config_rail1.yaml` | 75 | Same |
| `diverse/config_rail2.yaml` | 73 | Common in eval |
| `diverse/config_rail3_4.yaml` | ~50 | Less common, lighter weight |
| SC-port configs | 98 | SC trials, mix of `*_port01` and `sample_config.yaml` |

If your eval portal uses a different sampling distribution, retune these weights to match. The `sample_config.yaml` shows the canonical distribution.

## Schema reference

See [Intrinsic's aic_engine documentation](https://github.com/intrinsic-dev/aic/tree/main/aic_engine) for the full YAML schema. The most relevant keys for these configs:

- `trials` — number of trials per session
- `nic_rails` — list of allowed rail indices for NIC spawning
- `port_types` — `[sfp, sc]` or subsets
- `randomize_yaw` / `randomize_translation` — board pose randomization toggles
- `time_limit` — per-trial timeout in seconds
