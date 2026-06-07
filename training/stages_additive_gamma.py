"""
Additive-gamma schedule: same stage layout as default, but ICT coeffs use
``ict_raw + delta`` instead of ``ict_raw ** gamma`` (exponent personalization).
"""

from copy import deepcopy

from training.stages import STAGE_SCHEDULE, StageSpec

STAGE_SCHEDULE_ADDITIVE_GAMMA: list[StageSpec] = []
for _spec in STAGE_SCHEDULE:
    spec = deepcopy(_spec)
    if spec.steps > 0 and not getattr(spec, "use_ict_raw_coeffs", False):
        spec.additive_gamma_correction = True
    STAGE_SCHEDULE_ADDITIVE_GAMMA.append(spec)
