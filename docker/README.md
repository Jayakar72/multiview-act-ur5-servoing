# Docker Setup

This folder contains the build files for packaging the `MultiViewACTPolicy` into an OCI image suitable for local testing and submission to the AIC evaluation portal.

## Prerequisites

Before building, ensure you have:

1. **Trained checkpoints** (see [`../training/README.md`](../training/README.md) and [`../perception/README.md`](../perception/README.md)):
   - ACT policy: `~/aic_act_checkpoints/final/policy.pt` + `norm_stats.npy`
   - YOLO-OBB: `~/aic_yolo_models/best.pt`

2. **A workspace structured for the AIC toolkit.** The Dockerfile expects an Intrinsic-style ROS 2 workspace at the build context root. See [Intrinsic's getting-started guide](https://github.com/intrinsic-dev/aic/blob/main/docs/getting_started.md).

3. **The policy file placed in `aic_example_policies/aic_example_policies/ros/`** inside your workspace. Copy or symlink:
   ```bash
   cp policy/MultiViewACTPolicy.py \
      <your-workspace>/src/aic/aic_example_policies/aic_example_policies/ros/
   ```

## Build

From the **workspace root** (one level above the `aic` source directory):

```bash
DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.yaml build model
```

BuildKit cache mounts make rebuilds fast — only changed layers reprocess. First build takes ~30 minutes on a typical workstation (downloads ROS Kilted + Pixi dependencies). Subsequent builds take 3–6 minutes.

## Local end-to-end test

```bash
docker compose -f docker/docker-compose.yaml up
```

This launches both:
- `eval` — the official AIC evaluation environment (pulled from `ghcr.io/intrinsic-dev/aic/aic_eval`)
- `model` — your built policy

Watch the `eval` container logs for trial scores. The default `aic_engine` config runs 3 trials per session.

## Cleanup after testing

```bash
docker compose -f docker/docker-compose.yaml down
```

## Submitting to ECR

If your team has an Amazon ECR repository for submission (per the AIC qualification process):

```bash
# 1. Tag for ECR
docker tag multiview-act-ur5-servoing:latest \
  <your-ecr-registry>/<your-team>:<version-tag>

# 2. Authenticate (replace with the AWS profile your team uses)
export AWS_PROFILE=<your-aws-profile>
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <your-ecr-registry>

# 3. Push
docker push <your-ecr-registry>/<your-team>:<version-tag>
```

> **Note:** ECR tags are immutable in the AIC submission infrastructure. Increment your version tag for each submission (`v1`, `v2`, …) — don't try to overwrite an existing tag.

## Troubleshooting

### `pixi install --locked` fails with "lock-file not up-to-date"

This means your local `pixi.toml` has diverged from `pixi.lock`. Either:

(a) Regenerate the lockfile on the host:
```bash
cd <your-workspace>/src/aic
pixi install   # no --locked → updates pixi.lock
```
Then rebuild.

(b) Temporarily remove `--locked` from the Dockerfile (less safe — versions may drift between builds).

### Build runs out of disk space

The build creates ~15 GB of new image layers. The pixi package cache mount adds another ~15 GB but is shared across builds.

If you're tight on disk, clean stale containers and images before building:
```bash
docker container prune -f
docker image prune -f
```

### Local test reports `ZENOH_ROUTER_CHECK_ATTEMPTS` timeout

The policy container is starting before the eval container's Zenoh router is ready. The Dockerfile sets `ZENOH_ROUTER_CHECK_ATTEMPTS=-1` to disable the check, but if you've overridden it elsewhere, restore the `-1` value.
