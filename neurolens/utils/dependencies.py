"""Dependency checks that distinguish missing packages from broken imports."""
from importlib import import_module


def check_modules(modules):
    """Return available, missing, and broken imports separately."""
    available, missing, broken = [], [], []
    for name in modules:
        try:
            import_module(name)
            available.append(name)
        except ModuleNotFoundError as exc:
            if exc.name == name or name.startswith(f"{exc.name}."):
                missing.append(name)
            else:
                broken.append((name, f"missing transitive module {exc.name}: {exc}"))
        except Exception as exc:
            broken.append((name, f"{type(exc).__name__}: {exc}"))
    return available, missing, broken
