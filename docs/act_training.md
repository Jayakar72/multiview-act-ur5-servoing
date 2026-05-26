# ACT Training — Deep Dive

Detailed notes on the Action Chunking Transformer (ACT) component of the pipeline. The top-level training instructions are in [`training/README.md`](../training/README.md); this document covers the *why* behind the choices.

## What ACT is

[Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/) (ACT, Zhao et al. 2023) is a behavioral-cloning architecture designed for contact-rich bimanual manipulation. Two design choices distinguish it from naive behavioral cloning (BC):

### Action chunking

Standard BC predicts one action per timestep, conditioned on the current observation. This is brittle in two ways:

1. **Compounding error.** Small per-step errors accumulate. After 100 steps you're well outside the training distribution.
2. **Discretization mismatch.** Demonstrations naturally have multi-step structure (e.g., a teleoperator gives a single push, not 30 micro-corrections). Predicting one step at a time forces the policy to *re-discover* the multi-step plan every tick.

Action chunking predicts an entire trajectory of K future actions per inference. The policy commits to this chunk, executes it open-loop for some steps, then re-predicts. This dramatically reduces compounding error and matches the natural temporal structure of demonstrations.

In our setup: `chunk_size = 50`, executed at 20 Hz → each chunk plans **2.5 seconds ahead**.

### Conditional VAE

Humans demonstrate the same task in different ways. Approach angles vary, the order of micro-corrections varies, even the timing varies. If you train a deterministic policy on multi-modal data, it will average the modes — which often produces an action that's *not* in any of them. (Classic example: train on "go left around the obstacle" + "go right around the obstacle" and the average is "go through the obstacle".)

ACT adds a CVAE encoder that takes (observation, action_chunk) at training time and produces a latent style code `z`. At test time, you sample `z = 0` (the prior mean) — this gives you the "typical" behavior, not an averaged-mode artifact. The KL term keeps the latent space well-behaved.

In our setup: `use_vae = True`, `kl_weight = 10.0`, `n_vae_encoder_layers = 4`.

## Why we use LeRobot's implementation

[LeRobot](https://huggingface.co/lerobot) is the de facto open implementation of ACT (and several other manipulation policies). Reasons to use it:

- **Maintained.** The original ACT codebase is research-quality; LeRobot is maintained by Hugging Face with a small army of users finding bugs.
- **Self-contained model class.** `lerobot.policies.act.ACTPolicy` is just the model + a forward pass. It doesn't drag in their dataset loader, their trainer, or their environment wrapper. You can use the model in your own training loop.
- **Reasonable defaults.** Most hyperparameters in LeRobot's `ACTConfig` match the paper. We tweaked only a few.

What we *don't* use from LeRobot:
- Their `LeRobotDataset` format (we use our own HDF5 schema)
- Their `Trainer` (we wrote our own loop in `train_act.py`)
- Their environment wrappers (we use ROS 2 / AIC native)

## Custom HDF5 schema vs LeRobot dataset format

LeRobot's dataset format is JSONL-indexed parquet files, optimized for HuggingFace Hub distribution. For our use case it has two friction points:

1. **One-shot training**, not iterative dataset construction. We collect a fixed set of episodes and never modify them. The parquet/JSONL machinery is overhead.
2. **HDF5 is well-served by Python's h5py**, with built-in chunking and compression. No external dependency needed.

The custom schema:

```python
# Per-episode HDF5 file
{
    "images/cam_0": uint8 array (T, 224, 224, 3),
    "images/cam_1": uint8 array (T, 224, 224, 3),
    "images/cam_2": uint8 array (T, 224, 224, 3),
    "state":       float32 array (T, 16),
    "action":      float32 array (T, 7),
    "meta": {
        "port_type":  "sfp" or "sc",
        "rail_idx":   int,
        "success":    bool,
    }
}
```

`train_act.py` builds a `torch.utils.data.Dataset` over a directory of these files, with random sampling of `(observation, action_chunk)` windows.

## Hyperparameter notes

### `chunk_size = 50`

We tested `chunk_size` in [25, 50, 100]:
- 25 — too short to capture full approach motions; jittery
- 50 — smooth, captures full descent attempts
- 100 — slow to re-plan when surprised; risks executing stale plans through contact transients

50 was the sweet spot.

### `n_action_steps = chunk_size = 50`

Some ACT implementations execute fewer steps than the chunk length and average overlapping chunks ("temporal aggregation"). We tried this with `n_action_steps = 25` and saw no improvement on our task — possibly because our 20 Hz control rate is already aggressive enough that the policy gets re-conditioned often enough.

Skipping temporal aggregation also halves inference cost.

### `vision_backbone = "resnet18"` (no pretrained weights)

We tested:
- ResNet18 from scratch — what we ship
- ResNet18 with ImageNet pretrained weights — no improvement
- ResNet34 from scratch — slightly better but 30% slower; not worth the latency

The lack of benefit from ImageNet pretraining surprised us at first, but makes sense: wrist-camera scenes in the AIC simulator (sterile lab benchtop, neutral lighting, geometric primitives) are far from natural ImageNet imagery. Random init lets the network specialize fully on the domain.

### `kl_weight = 10.0`

We tested KL weights in [0.1, 1.0, 10.0, 100.0]:
- 0.1 — VAE latent collapses; equivalent to deterministic policy
- 1.0 — some mode mixing
- 10.0 — clean modes, stable training
- 100.0 — over-regularized; latent loses informativeness

The LeRobot default of 10.0 was correct.

## Training dynamics

A few things we noticed during training that might be useful:

- **Loss plateau at ~step 30,000.** L1 reconstruction loss stops dropping but KL loss continues evolving for another 20–30k steps. Stopping early at 30k gives a policy that *looks* converged but performs measurably worse than the full 100k. The CVAE needs the extra time to organize its latent space even when the action reconstruction is already good.

- **Lr scheduling didn't matter much** — we used constant LR = 1e-5 (LeRobot's default). Tested cosine annealing and warm restarts, neither beat constant LR for our dataset size.

- **Per-camera dropout helped a little** — randomly zeroing one of the three cameras 5% of the time during training acted as regularization and improved robustness when one camera view was occluded at test time.

- **State normalization matters.** Our 16-dim state includes pose (units: meters, dimensionless quaternion components), wrench (units: N, Nm — wildly different scales), and port encoding (one-hot-ish). Without per-dimension mean/std normalization (the `norm_stats.npy` file), training collapses immediately. With it, training is stable.

## What ACT does well (qualitatively)

- **Recognizing rim contact via wrench.** The policy "knows" what an empty descent feels like vs. a rim-hit feel like, and adjusts. This wasn't programmed — it emerged from the demonstrations.
- **Smooth approach trajectories.** Even with chunk re-prediction, motion is fluid; there's no visible chunk-boundary judder.
- **Quick recovery on cable tension.** When the cable pulls the wrist sideways during descent, the policy laterally corrects without losing Z progress.

## What ACT does poorly (the gap we target)

- **Multi-NIC scenes with the target on a back-rail.** When demonstrations were heavily biased to rails 0–1, performance on rails 3–4 degraded smoothly with distance from the demo distribution.
- **Initial localization.** From the spawn pose, ACT often doesn't *know* where the target NIC is. It's working from camera images alone with no explicit world model.
- **Recovery from bad alignment.** If the wrist arrives even 3–4 mm off the port, the policy doesn't reliably search for it — it tries to descend anyway.

These are exactly the failure modes our Tier 2/3 layers target. Tier 2 gives ACT (or any descent strategy) a well-conditioned starting pose. Tier 3 handles search and recovery so ACT doesn't need to.

## How ACT is invoked at runtime

In the final `MultiViewACTPolicy.py`, ACT is loaded but its `ACT_MODE` flag controls whether it's actively used:

- `ACT_MODE = "skip"` — never call ACT; use the geometric descent + recovery pipeline
- `ACT_MODE = "descent"` — invoke ACT during the descent phase (Phase 2) after visual servoing
- `ACT_MODE = "all"` — invoke ACT from homing onward

Our submitted policy ships with `ACT_MODE` set based on port type. SC trials use ACT for descent (the geometry-only pipeline is less tuned there); SFP trials use the geometric descent.

## Cross-references

- Where to actually run training: [`../training/README.md`](../training/README.md)
- How ACT fits in the pipeline: [architecture.md](architecture.md)
- The data ACT trains on: [`../data_collection/README.md`](../data_collection/README.md)
