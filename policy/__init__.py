"""MultiView ACT UR5 Servoing — policy module.

# Primary policy

`MultiViewACTPolicy` — the hybrid policy submitted for evaluation:
  - Multi-view scene registration via YOLO-OBB + PnP
  - Cable-aware approach planning
  - Hybrid YOLO/registered-pose visual servoing
  - Failure-aware descent and mode-gated spiral recovery
  - ACT (Action Chunking Transformer) for contact dynamics (optional)

The policy is loaded by the AIC engine via the aic_model framework.
See the repository README for the full architecture description.

# Auxiliary policies (not imported here — load on demand)

`CheatCode.py` — Intrinsic's reference policy that uses ground-truth TF data
to compute exact target poses. We use it as the *driver* during YOLO-OBB
data collection. It is NOT submitted for evaluation.

`WallPresser.py` — One of Intrinsic's example policies, kept here as a
minimal working reference for the aic_model framework.

Both are loaded explicitly by the eval engine via their module path
(e.g., `aic_example_policies.ros.CheatCode`), so they don't need to be
re-exported from this __init__.
"""

from .MultiViewACTPolicy import MultiViewACTPolicy

__all__ = ["MultiViewACTPolicy"]
