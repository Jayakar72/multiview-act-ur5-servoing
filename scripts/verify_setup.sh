#!/usr/bin/env bash
# verify_setup.sh — Sanity check for the multiview-act-ur5-servoing pipeline.
# Runs a series of checks to ensure all prerequisites are in place
# before attempting to build the Docker image or run the policy.

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()    { echo -e "${GREEN}  ✓${NC} $1"; PASS=$((PASS+1)); }
bad()   { echo -e "${RED}  ✗${NC} $1"; FAIL=$((FAIL+1)); }
warn()  { echo -e "${YELLOW}  ⚠${NC} $1"; WARN=$((WARN+1)); }
section() { echo ""; echo -e "${YELLOW}── $1 ──${NC}"; }

# ─── 1. System requirements ─────────────────────────────────────────────
section "System requirements"

command -v docker &>/dev/null \
  && ok "docker installed: $(docker --version | head -1)" \
  || bad "docker not found — install Docker first"

command -v pixi &>/dev/null \
  && ok "pixi installed: $(pixi --version | head -1)" \
  || warn "pixi not found — needed for training. Install: curl -fsSL https://pixi.sh/install.sh | bash"

command -v nvidia-smi &>/dev/null \
  && ok "nvidia-smi available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  || warn "nvidia-smi not found — GPU access may be limited"

command -v distrobox &>/dev/null \
  && ok "distrobox installed (needed for aic_eval)" \
  || warn "distrobox not found — needed for running the eval container"

# ─── 2. Disk space ──────────────────────────────────────────────────────
section "Disk space"

AVAIL_GB=$(df --output=avail / | tail -1 | awk '{print int($1/1024/1024)}')
if (( AVAIL_GB >= 50 )); then
  ok "$AVAIL_GB GB free on /  (plenty for full pipeline)"
elif (( AVAIL_GB >= 20 )); then
  warn "$AVAIL_GB GB free on /  (tight — clean Docker if build fails)"
else
  bad "Only $AVAIL_GB GB free on / — need ≥ 20 GB for Docker build"
fi

# ─── 3. Required checkpoints ────────────────────────────────────────────
section "Trained checkpoints (you must train these yourself)"

ACT_DIR="${HOME}/aic_act_checkpoints/final"
ACT_PT="${ACT_DIR}/policy.pt"
ACT_NORM="${ACT_DIR}/norm_stats.npy"

[[ -f "$ACT_PT" ]]    && ok "ACT checkpoint: $ACT_PT"       || warn "Missing $ACT_PT  (see training/README.md)"
[[ -f "$ACT_NORM" ]]  && ok "ACT norm stats: $ACT_NORM"     || warn "Missing $ACT_NORM (see training/README.md)"

YOLO_PT="${HOME}/aic_yolo_models/best.pt"
[[ -f "$YOLO_PT" ]] && ok "YOLO checkpoint: $YOLO_PT" || warn "Missing $YOLO_PT (see perception/README.md)"

# ─── 4. Workspace structure ─────────────────────────────────────────────
section "Workspace integration (expects an Intrinsic AIC-style workspace)"

# Try to find a likely workspace root
WS_CANDIDATES=(
  "${HOME}/ws_aic"
  "${HOME}/aic_workspace"
  "${HOME}/aic_ws"
  "${HOME}/workspace"
)

WS_ROOT=""
for cand in "${WS_CANDIDATES[@]}"; do
  if [[ -d "$cand/src/aic/aic_example_policies" ]]; then
    WS_ROOT="$cand"
    break
  fi
done

if [[ -n "$WS_ROOT" ]]; then
  ok "Found AIC workspace at: $WS_ROOT"
  POLICY_TARGET="$WS_ROOT/src/aic/aic_example_policies/aic_example_policies/ros/MultiViewACTPolicy.py"
  if [[ -f "$POLICY_TARGET" ]]; then
    ok "Policy file present in workspace: $POLICY_TARGET"
  else
    warn "Policy not yet placed in workspace — copy policy/MultiViewACTPolicy.py to: $POLICY_TARGET"
  fi
else
  warn "Could not auto-detect AIC workspace under \$HOME — set up Intrinsic toolkit first"
fi

# ─── 5. Docker daemon ───────────────────────────────────────────────────
section "Docker daemon"

if docker info &>/dev/null; then
  ok "Docker daemon is running"
  # Check for GPU support
  if docker info 2>/dev/null | grep -q "nvidia"; then
    ok "NVIDIA Docker runtime detected"
  else
    warn "NVIDIA Docker runtime not detected — GPU may not be accessible inside container"
  fi
else
  bad "Docker daemon not running — sudo systemctl start docker"
fi

# ─── Summary ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════"
echo -e "  ${GREEN}Passed:${NC}  $PASS"
echo -e "  ${YELLOW}Warnings:${NC} $WARN"
echo -e "  ${RED}Failed:${NC}  $FAIL"
echo "═══════════════════════════════════════"

if (( FAIL > 0 )); then
  echo ""
  echo -e "${RED}Some checks failed — see messages above.${NC}"
  exit 1
elif (( WARN > 0 )); then
  echo ""
  echo -e "${YELLOW}All required checks passed, but some warnings — review before building.${NC}"
  exit 0
else
  echo ""
  echo -e "${GREEN}All checks passed — you're ready to build.${NC}"
  exit 0
fi
