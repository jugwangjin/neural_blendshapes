# Documentation index

| Directory | Purpose |
|-----------|---------|
| [`guides/`](guides/) | How to run training, CLI examples, experiment notes |
| [`implementation/`](implementation/) | Code-level design for agents and future refactors |

## Guides

- [Training & CLI](guides/training.md) — `train.py` arguments, output paths, densify options

## Implementation

- [Opacity regularization](implementation/opacity_regularization.md) — stage weights, sigmoid fix, decay removal
- [Densification & grow signal](implementation/densify_grow_signal.md) — `grad2d` vs `gradrgb`, config ↔ `training/densify.py`
- [Train CLI wiring](implementation/train_cli.md) — `parse_train_cli` / `apply_train_cli` ↔ `Config`
- [Blurry rendering analysis](implementation/blurry_rendering_analysis.md) — current blur hypotheses and server-log checks
- [Distribution sampling weights](implementation/distribution_sampling_weights.md) — MP+pose variance weights and RGB EMA merge
- [Geometry LR decay](implementation/geometry_lr_decay.md) — GB-style decay on h/bary_uv + template/expr MLPs
- [loss_log densify grads](implementation/loss_log_densify_grad2d_gradrgb.md) — grad2d/gradrgb JSONL fields and known logging bugs
