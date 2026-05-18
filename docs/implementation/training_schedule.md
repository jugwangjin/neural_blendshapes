# Training schedule (2-stage)

Expression deformer is **off in Stage 1**, on in Stage 2 (2A warmup → 2B full Gaussian).

## Stages

| Stage | Steps | Train | Freeze |
|-------|-------|-------|--------|
| 0 | 0 | — | precompute caches |
| **1** `coarse_geometry` | 15k | tracker γ, pose, pose weight, **template deformer** | expr deformer; Gaussian uv/h/scale/rot |
| **2A** `expression_warmup` | 5k | expr deformer; color/opacity; uv/h **low LR** | tracker, template, pose |
| **2B** `gaussian_detail` | 35k | expr (low LR) + full Gaussian + semantic + gaze | template / tracker frozen |

**Total:** 55k steps — see `training/stages.py`.

## Roles

```text
Template deformer (template_offset):
  subject canonical shape — jaw, face width, scalp anchor

Expression deformer (ExpressionDeformMLP):
  per-expression personalization — lip, blink, jaw
  masked by expr_region_weight (hair/accessory = 0)

Gaussian h/scale/opacity/color:
  appearance + hair volume (Stage 2B)
```

## Stage 1 Gaussian policy

Only **color + opacity** train; **uv / h / scale / rotation** frozen so RGB/seg gradients flow to mesh/template.

## H prior (`gaussian_splatting/semantic.py`)

Per-class `sigma` + `weight` (Charbonnier on `|h|/sigma`):

- skin / eye / iris: strong (σ≈0.002, w=1)
- lip: medium
- hair: weak (w=0.1)
- accessory / bg: off (w=0)

## Code map

- `training/stages.py` — `STAGE_SCHEDULE`
- `training/apply.py` — granular Gaussian freeze + optimizers
- `model/expr_regions.py` — `build_expr_region_weight(ict)`
- `train.py` — stage loop
