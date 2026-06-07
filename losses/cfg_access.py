"""Config / loss_cfg weight lookup (no imports from other losses.* modules)."""


def get_loss_weight(cfg, key, default=0.0):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
