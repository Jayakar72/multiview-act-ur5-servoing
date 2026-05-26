# Training the ACT Policy

This folder contains the training script and per-rail evaluation configs used to train the [Action Chunking Transformer](https://tonyzhaozh.github.io/aloha/) (ACT) policy that handles contact dynamics in the Tier 1 pipeline.

> **Note:** This repository does **not** ship pretrained checkpoints. You'll need to collect demonstrations and train your own. See [`../data_collection/README.md`](../data_collection/README.md) for data collection.

## What's in this folder

| File | Purpose |
|------|---------|
| `train_act.py` | Custom training driver. Reads HDF5 demos directly, uses LeRobot's `ACTPolicy` class for the model. |
| `configs/config_rail0.yaml` | AIC engine config for rail-0 demos (NICs spawned on rail 0). |
| `configs/config_rail1.yaml` | Same for rail 1. |
| `configs/config_rail2.yaml` | Same for rail 2. |
| `configs/config_rail3_4.yaml` | Combined rails 3+4 (rare in the eval distribution). |
| `configs/config_sc.yaml` | SC-port trials. |

> **Reproduction note:** You'll need to copy `train_act.py` from your local workspace. The training script is custom-built around our HDF5 schema (see [`../data_collection/README.md`](../data_collection/README.md)) and not directly tied to LeRobot's `datasets` interface.

## Dependencies

Add LeRobot to your Pixi environment if it isn't already:

```bash
pixi add --pypi lerobot
```

You'll also need:
- PyTorch with CUDA support
- h5py (for reading the HDF5 demo files)
- numpy

## Training command

```bash
cd <your-workspace>
pixi run python3 training/train_act.py \
  --demo_dir ~/aic_demos \
  --output_dir ~/aic_act_checkpoints \
  --steps 100000 \
  --batch_size 8 \
  --save_freq 10000 \
  --log_freq 100
```

## Hyperparameters that matter

The defaults below are the ones we landed on after exploration. See `train_act.py` for the full list.

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| `chunk_size` | 50 | Predicts ~2.5 seconds of action at 20 Hz, smooths the policy through contact transients. |
| `n_action_steps` | 50 | Match chunk size — no temporal aggregation. |
| `dim_model` | 512 | Standard ACT setting. |
| `n_heads` | 8 | Standard. |
| `dim_feedforward` | 3200 | Standard (6.25× dim_model). |
| `n_encoder_layers` | 4 | Standard. |
| `n_decoder_layers` | 1 | Standard ACT — one decoder layer is enough for the autoregressive chunk. |
| `n_vae_encoder_layers` | 4 | Standard. |
| `use_vae` | True | Captures multi-modal demonstrations (different teleoperators handle the same scene differently). |
| `kl_weight` | 10.0 | Standard. |
| `vision_backbone` | `resnet18` | Lightweight; we don't have enough data to need bigger. |
| `pretrained_backbone_weights` | `None` | We train from scratch — domain gap from ImageNet to wrist-camera scenes is large enough that ImageNet init doesn't help. |

### Observation structure

- 3× wrist cameras at 224 × 224 RGB
- 16-dim state vector:
  - 7-dim TCP pose (xyz + quaternion)
  - 6-dim wrist wrench (force + torque)
  - 3-dim port encoding (is_sfp, is_sc, port_index)

### Action space

- 7-dim TCP target pose (xyz + quaternion)

## Training cost

| Resource | Cost |
|----------|------|
| Hardware | Single NVIDIA RTX 4070 (12 GB VRAM) |
| Steps | 100,000 |
| Batch size | 8 |
| Wall time | ~3 hours |
| Dataset | 302 demos |
| Passes per demo | ~330 |

We scaled steps with data size: 50,000 steps when we had ~100 demos, doubled to 100,000 once we reached 302. The rough heuristic is to maintain ~300 passes per demonstration regardless of dataset size.

## Output

```
~/aic_act_checkpoints/
└── final/
    ├── policy.pt        ← LeRobot ACTPolicy state dict
    └── norm_stats.npy   ← state_mean, state_std, action_mean, action_std
```

These files are loaded by `MultiViewACTPolicy` at runtime. Place them where the Dockerfile expects (see [`../docker/README.md`](../docker/README.md)).

## What we learned

- **The dataset is what matters most.** Doubling our dataset from 150 → 302 episodes had a larger effect on task success than any hyperparameter sweep we ran.
- **Demonstration distribution beats demonstration count.** Of those 302 episodes, weighting rails 0 and 1 (which the eval portal samples from heavily, per Intrinsic's `sample_config.yaml`) gave more reliable scores than collecting uniform per-rail data.
- **The wrist wrench input matters.** Removing it from the state vector dropped task success noticeably — ACT uses contact forces to recognize "I'm against the rim" without explicit programming.
- **VAE helps for SC trials, hurts a bit for SFP.** SC ports have a wider variation in approach angles; SFP has tighter geometry. The VAE captures both modes but slightly dilutes precision on the easier task. We kept it on for cross-task robustness.

## Cross-references

- Data collection schema: [`../data_collection/README.md`](../data_collection/README.md)
- Where ACT fits in the larger pipeline: [`../docs/architecture.md`](../docs/architecture.md)
- Detailed training notes: [`../docs/act_training.md`](../docs/act_training.md)
