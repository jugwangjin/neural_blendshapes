"""MediaPipe AU / iMotions emotion keyframe sequences for ``render_control_video.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

SequenceBuilder = Callable[..., list[dict[str, float]]]


def apply_mp_aus(
    mp_blendshape: torch.Tensor,
    name_to_idx: dict[str, int],
    aus: dict[str, float],
) -> torch.Tensor:
    """Return calibrated ``mp_blendshape`` copy with AU overrides (52-D cache columns)."""
    from utils.mediapipe_blendshapes import resolve_mp_blendshape_index

    out = mp_blendshape.clone()
    for name, val in aus.items():
        out[resolve_mp_blendshape_index(name, name_to_idx)] = float(val)
    return out


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _ramp(n: int, v0: float, v1: float) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [float(v1)]
    return [_lerp(v0, v1, i / (n - 1)) for i in range(n)]


def _hold(n: int, v: float) -> list[float]:
    return [float(v)] * int(n)


@dataclass(frozen=True)
class FacsAuSpec:
    au: int
    slug: str
    label: str
    mp_peak: dict[str, float]


# FACS AU → MediaPipe (iMotions main action unit table).
FACS_AU_SPECS: tuple[FacsAuSpec, ...] = (
    FacsAuSpec(1, "inner_brow_raiser", "Inner Brow Raiser", {"browInnerUpLeft": 1.0, "browInnerUpRight": 1.0}),
    FacsAuSpec(2, "outer_brow_raiser", "Outer Brow Raiser", {"browOuterUpLeft": 1.0, "browOuterUpRight": 1.0}),
    FacsAuSpec(4, "brow_lowerer", "Brow Lowerer", {"browDownLeft": 1.0, "browDownRight": 1.0}),
    FacsAuSpec(5, "upper_lid_raiser", "Upper Lid Raiser", {"eyeWideLeft": 1.0, "eyeWideRight": 1.0}),
    FacsAuSpec(6, "cheek_raiser", "Cheek Raiser", {"cheekSquintLeft": 1.0, "cheekSquintRight": 1.0}),
    FacsAuSpec(7, "lid_tightener", "Lid Tightener", {"eyeSquintLeft": 1.0, "eyeSquintRight": 1.0}),
    FacsAuSpec(9, "nose_wrinkler", "Nose Wrinkler", {"noseSneerLeft": 1.0, "noseSneerRight": 1.0}),
    FacsAuSpec(12, "lip_corner_puller", "Lip Corner Puller", {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0}),
    FacsAuSpec(14, "dimpler", "Dimpler", {"mouthDimpleLeft": 1.0, "mouthDimpleRight": 1.0}),
    FacsAuSpec(15, "lip_corner_depressor", "Lip Corner Depressor", {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0}),
    FacsAuSpec(
        16,
        "lower_lip_depressor",
        "Lower Lip Depressor",
        {"mouthLowerDownLeft": 1.0, "mouthLowerDownRight": 1.0},
    ),
    FacsAuSpec(20, "lip_stretcher", "Lip Stretcher", {"mouthStretchLeft": 1.0, "mouthStretchRight": 1.0}),
    FacsAuSpec(23, "lip_tightener", "Lip Tightener", {"mouthPressLeft": 1.0, "mouthPressRight": 1.0}),
    FacsAuSpec(26, "jaw_drop", "Jaw Drop", {"jawOpen": 1.0}),
)


def _canonical_au_mp() -> dict[int, dict[str, float]]:
    return {spec.au: dict(spec.mp_peak) for spec in FACS_AU_SPECS}


AU_MP = _canonical_au_mp()


@dataclass(frozen=True)
class EmotionSpec:
    """
    iMotions / Affectiva emotion = ordered FACS AUs (gifs listed in this order).

    Each AU is activated sequentially; earlier AUs stay on while the next ramps in.
    """

    slug: str
    label: str
    au_order: tuple[int, ...]
    au_steps: tuple[dict[str, float], ...] | None = None


# https://imotions.com/blog/.../facial-action-coding-system/ — Emotions and Action Units
EMOTION_SPECS: tuple[EmotionSpec, ...] = (
    EmotionSpec("happiness", "Happiness / Joy", (6, 12)),
    EmotionSpec("sadness", "Sadness", (1, 4, 15)),
    EmotionSpec("surprise", "Surprise", (1, 2, 5, 26)),
    # EmotionSpec("fear", "Fear", (1, 2, 4, 5, 7, 20, 26)),
    EmotionSpec("anger", "Anger", (4, 5, 7, 23)),
    # EmotionSpec("disgust", "Disgust", (9, 15, 16)),
    EmotionSpec(
        "contempt",
        "Contempt (one side)",
        (),
        au_steps=(
            {"mouthSmileLeft": 1.0},
            {"mouthSmileLeft": 1.0, "mouthDimpleLeft": 1.0},
        ),
    ),
)


def _au_steps_for_emotion(spec: EmotionSpec) -> list[dict[str, float]]:
    if spec.au_steps is not None:
        return [dict(s) for s in spec.au_steps]
    return [dict(AU_MP[n]) for n in spec.au_order]


def build_emotion_sequential_sequence(
    au_steps: list[dict[str, float]],
    *,
    peak: float = 0.75,
    n_ramp: int = 14,
    n_hold_step: int = 10,
    n_hold_full: int = 16,
    n_down: int = 16,
    n_rest: int = 10,
) -> list[dict[str, float]]:
    """
    Activate each AU in order (iMotions gif order). Previous AUs remain at peak
    while the next ramps in → full emotion → release.
    """
    frames: list[dict[str, float]] = []
    active: dict[str, float] = {}
    p = float(peak)

    for step_mp in au_steps:
        step_target = {k: float(v) * p for k, v in step_mp.items()}
        new_keys = [k for k in step_target if k not in active]
        for t in _ramp(n_ramp, 0.0, 1.0):
            frame = dict(active)
            for k in new_keys:
                frame[k] = step_target[k] * t
            frames.append(frame)
        active.update(step_target)
        for _ in range(int(n_hold_step)):
            frames.append(dict(active))

    for _ in range(int(n_hold_full)):
        frames.append(dict(active))
    for t in _ramp(n_down, 1.0, 0.0):
        frames.append({k: v * t for k, v in active.items()})
    frames.extend({} for _ in range(int(n_rest)))
    return frames


def _builder_for_emotion(spec: EmotionSpec) -> SequenceBuilder:
    steps = _au_steps_for_emotion(spec)

    def builder(
        *,
        peak: float = 0.75,
        n_ramp: int = 14,
        n_hold_step: int = 10,
        n_hold_full: int = 16,
        n_down: int = 16,
        n_rest: int = 10,
        **_ignored,
    ) -> list[dict[str, float]]:
        return build_emotion_sequential_sequence(
            steps,
            peak=peak,
            n_ramp=n_ramp,
            n_hold_step=n_hold_step,
            n_hold_full=n_hold_full,
            n_down=n_down,
            n_rest=n_rest,
        )

    builder.__doc__ = f"{spec.label}: AUs {spec.au_order or 'custom'} one-by-one"
    return builder


def build_smile_wink_sequence(
    *,
    smile_peak: float = 0.75,
    brow_peak: float = 0.75,
    blink_peak: float = 0.75,
    n_ramp: int = 18,
    n_hold: int = 12,
    n_brow_ramp: int = 16,
    n_brow_hold: int = 10,
    n_blink_ramp: int = 10,
    n_blink_hold: int = 14,
    **_ignored,
) -> list[dict[str, float]]:
    """Legacy composite demo (not iMotions emotion table)."""
    frames: list[dict[str, float]] = []
    for v in _ramp(n_ramp, 0.0, smile_peak):
        frames.append({"mouthSmileRight": v})
    for v in _hold(n_hold, smile_peak):
        frames.append({"mouthSmileRight": v})
    carry_smile = {"mouthSmileRight": smile_peak}
    for v in _ramp(n_brow_ramp, 0.0, brow_peak):
        frames.append({**carry_smile, "browInnerUpLeft": v, "browInnerUpRight": v})
    for v in _hold(n_brow_hold, brow_peak):
        frames.append({**carry_smile, "browInnerUpLeft": v, "browInnerUpRight": v})
    for v in _ramp(n_brow_ramp, brow_peak, 0.0):
        frames.append({**carry_smile, "browInnerUpLeft": v, "browInnerUpRight": v})
    carry_wink = {**carry_smile, "browInnerUpLeft": 0.0, "browInnerUpRight": 0.0}
    for v in _ramp(n_blink_ramp, 0.0, blink_peak):
        frames.append({**carry_wink, "eyeBlinkRight": v})
    for v in _hold(n_blink_hold, blink_peak):
        frames.append({**carry_wink, "eyeBlinkRight": v})
    return frames


def build_emotions_all_sequence(
    *,
    peak: float = 0.75,
    n_ramp: int = 14,
    n_hold_step: int = 10,
    n_hold_full: int = 16,
    n_down: int = 16,
    n_rest: int = 10,
    **_ignored,
) -> list[dict[str, float]]:
    """All iMotions emotion clips back-to-back."""
    kw = dict(
        peak=peak,
        n_ramp=n_ramp,
        n_hold_step=n_hold_step,
        n_hold_full=n_hold_full,
        n_down=n_down,
        n_rest=n_rest,
    )
    frames: list[dict[str, float]] = []
    for name in list_emotion_sequence_names():
        frames.extend(CONTROL_SEQUENCES[name](**kw))
    return frames


CONTROL_SEQUENCES: dict[str, SequenceBuilder] = {
    "smile_wink": build_smile_wink_sequence,
    "emotions_all": build_emotions_all_sequence,
}

for _espec in EMOTION_SPECS:
    CONTROL_SEQUENCES[f"emotion_{_espec.slug}"] = _builder_for_emotion(_espec)


def list_emotion_sequence_names() -> list[str]:
    return sorted(k for k in CONTROL_SEQUENCES if k.startswith("emotion_") and k != "emotion_all")


def list_control_sequences() -> list[str]:
    return sorted(CONTROL_SEQUENCES.keys())


def emotion_catalog() -> list[dict[str, str]]:
    rows = []
    for spec in EMOTION_SPECS:
        if spec.au_order:
            aus = "+".join(str(n) for n in spec.au_order)
        else:
            aus = " → ".join(
                "+".join(k.replace("Left", "L").replace("Right", "R") for k in step.keys())
                for step in spec.au_steps or ()
            )
        rows.append(
            {
                "sequence": f"emotion_{spec.slug}",
                "label": spec.label,
                "aus": aus,
            }
        )
    return rows


def build_control_sequence(name: str, **kwargs) -> list[dict[str, float]]:
    if name not in CONTROL_SEQUENCES:
        raise KeyError(f"unknown sequence {name!r}; choose from {list_control_sequences()}")
    return CONTROL_SEQUENCES[name](**kwargs)
