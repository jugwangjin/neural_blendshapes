"""Safe restart for multiprocessing DataLoader iterators."""


def loader_persistent_workers(loader) -> bool:
    return bool(getattr(loader, "persistent_workers", False))


def shutdown_loader_iter(loader_iter):
    """
    Shut down worker processes (non-persistent loaders only).

    Do **not** call between training stages when ``persistent_workers=True`` — workers
    must stay alive for the next ``iter(loader)``.
    """
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def restart_loader_iter(loader, loader_iter=None):
    """New epoch: reset iterator; keep workers if ``persistent_workers``."""
    if not loader_persistent_workers(loader):
        shutdown_loader_iter(loader_iter)
    return iter(loader)
