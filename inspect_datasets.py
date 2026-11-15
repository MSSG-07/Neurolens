"""Read-only inspection of BraTS cases, split manifests, and BRISC metadata."""
import argparse
import hashlib
import json
from pathlib import Path
from neurolens.datasets.brats import discover_patients
from neurolens.datasets.manifests import inspect_manifest


def inspect_brisc(root):
    root = Path(root)
    print(f"BRISC root: {root} (exists={root.is_dir()})")
    if not root.is_dir():
        return
    readmes = list(root.rglob("README.md"))
    manifests = list(root.rglob("manifest.csv")) + list(root.rglob("manifest.json"))
    for path in readmes:
        print(f"\nREADME: {path}\n" + path.read_text(encoding="utf-8", errors="replace")[:12000])
    for path in manifests:
        try:
            print("\nManifest inspection:", json.dumps(inspect_manifest(str(path)), indent=2, default=str)[:12000])
            checksum = path.with_suffix(path.suffix + ".sha256")
            if checksum.is_file():
                expected = checksum.read_text(encoding="utf-8").split()[0]
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"SHA256 {path.name}: expected={expected} actual={actual} match={expected == actual}")
        except Exception as exc:
            print(f"Could not inspect {path}: {type(exc).__name__}: {exc}")
    print("\nDirectory and file sample (inspection only; no labels inferred):")
    for path in sorted(root.rglob("*"))[:100]:
        if path.is_dir():
            print("DIR ", path.relative_to(root))
        elif path.is_file():
            print("FILE", path.relative_to(root), path.stat().st_size, "bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", default="")
    parser.add_argument("--brisc-root", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--id-column", default="patient_id")
    args = parser.parse_args()
    if args.brats_root:
        cases = discover_patients(args.brats_root, ("t1n", "t1c", "t2w", "t2f"))
        print(f"Complete BraTS cases: {len(cases)} under {args.brats_root}")
    for name in ("train", "val", "test"):
        path = getattr(args, f"{name}_manifest")
        if path:
            print(f"\n{name} manifest: {json.dumps(inspect_manifest(path), indent=2, default=str)[:12000]}")
    if args.brisc_root:
        inspect_brisc(args.brisc_root)


if __name__ == "__main__":
    main()
