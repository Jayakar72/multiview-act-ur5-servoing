FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git curl && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"
ENV ZENOH_ROUTER_CHECK_ATTEMPTS=-1

# ── Copy source ──────────────────────────────────────────────────────
# This Dockerfile assumes the policy file has been placed inside an
# aic_example_policies-style ROS 2 package at the build context root.
# Adjust COPY paths to match your workspace layout.
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model            /ws_aic/src/aic/aic_model
COPY aic_interfaces       /ws_aic/src/aic/aic_interfaces
COPY aic_utils            /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock robostack_mapping.json /ws_aic/src/aic/
COPY pixi_env_setup.sh    /ws_aic/src/aic/

# ── Copy trained checkpoints ─────────────────────────────────────────
# Provide your own checkpoints. See:
#   - training/README.md for training the ACT policy
#   - perception/README.md for training the YOLO-OBB detector
COPY aic_act_checkpoints/final /root/aic_act_checkpoints/final
COPY aic_yolo_models/best.pt    /root/aic_yolo_models/best.pt

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked

WORKDIR /ws_aic/src/aic

# ── Entrypoint: configure Zenoh and launch aic_model ─────────────────
COPY --chmod=755 <<"EOF2" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ATOMIC_ACT_CHECKPOINT=/root/aic_act_checkpoints/final
export AIC_YOLO_CHECKPOINT=/root/aic_yolo_models/best.pt

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF2

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.MultiViewACTPolicy", "-p", "use_sim_time:=true"]
