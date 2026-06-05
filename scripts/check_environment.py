#!/usr/bin/env python
"""Check that all required packages are installed and importable."""

import sys

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "scanpy", "anndata",
    "torch", "matplotlib", "seaborn", "tqdm", "yaml", "h5py",
    "umap", "einops",
]

OPTIONAL = [
    "scrublet", "harmonypy", "scgpt", "transformers",
]


def main():
    ok = 0
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg)
            ok += 1
        except ImportError:
            missing.append(pkg)

    for pkg in OPTIONAL:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[OPTIONAL] {pkg} not found (required for some features)")

    print(f"\nRequired packages: {ok}/{len(REQUIRED)} available")
    if missing:
        print(f"Missing required: {missing}")
        sys.exit(1)
    else:
        print("All required packages available.")
        sys.exit(0)


if __name__ == "__main__":
    main()