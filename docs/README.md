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
