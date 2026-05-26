#!/bin/bash
#
# auto_collect.sh — Fully automated demo collection
# Automatically restarts simulator between runs
#
# USAGE:
#   bash auto_collect.sh --config config_rail2.yaml --target 225
#

# ── Config ────────────────────────────────────────────────────────────────────
WS_DIR="$HOME/ws_aic_new/src/aic"
DEMO_DIR="$HOME/aic_demos"
CONFIG_DIR="$WS_DIR/aic_engine/config/diverse"
TARGET=225
CONFIG="config_rail0.yaml"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)  CONFIG="$2";  shift 2 ;;
        --target)  TARGET="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CONFIG_PATH="$CONFIG_DIR/$CONFIG"

# ── Helpers ───────────────────────────────────────────────────────────────────
count_episodes() {
    ls "$DEMO_DIR"/episode_*.h5 2>/dev/null | wc -l
}

kill_sim() {
    echo "  Stopping simulator..."
    # Kill distrobox/docker processes for aic_eval
    docker stop aic_eval 2>/dev/null || true
    sleep 3
    docker start aic_eval 2>/dev/null || true
    sleep 2
}

start_sim() {
    echo "  Starting simulator with $CONFIG..."
    export DBX_CONTAINER_MANAGER=docker
    distrobox enter -r aic_eval -- /entrypoint.sh \
        ground_truth:=true \
        start_aic_engine:=true \
        aic_engine_config_file:="$CONFIG_PATH" &
    SIM_PID=$!
    echo "  Simulator PID: $SIM_PID"

    # Wait for aic_engine to be ready
    echo "  Waiting for simulator to be ready..."
    local elapsed=0
    while true; do
        nodes=$(cd "$WS_DIR" && timeout 3 pixi run \
            ros2 node list 2>/dev/null | grep -c "aic_engine" || true)
        if [[ "$nodes" -gt 0 ]]; then
            echo "  ✓ Simulator ready!"
            sleep 5  # extra buffer
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        if [[ $((elapsed % 15)) -eq 0 ]]; then
            echo "  Still waiting for sim... (${elapsed}s)"
        fi
    done
}

run_recorder() {
    echo "  Running AtomicRecorder..."
    cd "$WS_DIR"
    pixi run ros2 run aic_model aic_model \
        --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.AtomicRecorder
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════╗"
echo "║   ATOMIC Auto Collector                ║"
printf "║   Config : %-28s ║\n" "$CONFIG"
printf "║   Target : %-4d episodes              ║\n" "$TARGET"
echo "╚════════════════════════════════════════╝"
echo ""

mkdir -p "$DEMO_DIR"
run_num=0

while true; do
    current=$(count_episodes)

    if [[ "$current" -ge "$TARGET" ]]; then
        echo "✓ Target reached: $current / $TARGET episodes"
        break
    fi

    run_num=$((run_num + 1))
    echo ""
    echo "─── Run $run_num | Episodes: $current / $TARGET ───"

    # Start sim
    start_sim

    # Run recorder
    run_recorder
    
    after=$(count_episodes)
    echo "  ✓ Run $run_num done | Episodes: $after / $TARGET"

    # Kill sim
    kill_sim

    sleep 2
done

echo ""
echo "✓ Collection complete! Total: $(count_episodes) episodes"
