"""Check training dependencies without installing or upgrading packages."""
import sys
from neurolens.utils.dependencies import check_modules


def main():
    required = ("torch", "numpy", "nibabel", "yaml")
    optional = ("sklearn", "matplotlib", "scipy", "PIL")
    failures = False
    for title, modules in (("Required", required), ("Optional", optional)):
        available, missing, broken = check_modules(modules)
        print(f"{title} available: {', '.join(available) or 'none'}")
        print(f"{title} missing: {', '.join(missing) or 'none'}")
        for module, error in broken:
            print(f"{title} import error for {module}: {error}")
        if title == "Required" and (missing or broken):
            failures = True
    return int(failures)


if __name__ == "__main__":
    sys.exit(main())
