"""Keep the embodied gen_0 module tree and the installed package in sync.

Why this exists: the evolution editor can only READ files inside the staged
module tree, while editor-authored variants subclass the PACKAGE import path
(`harbor.agents.terminus_2_modular.modules.tools.baseline`). The tools baseline
must therefore be byte-identical in both places.

Source of truth: the installed package. Run this after editing the package
baseline and before starting an evolution run.

Usage:
    python embodied/sync_modules.py           # copy package -> tree
    python embodied/sync_modules.py --check   # verify only (exit 1 on drift)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_MODULES = (
    REPO_ROOT / "src" / "harbor" / "agents" / "terminus_2_modular" / "modules"
)
TREE_MODULES = Path(__file__).resolve().parent / "modules"

# (package-relative path, tree-relative path)
SYNCED = [
    ("tools/baseline.py", "tools/baseline.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()

    drifted: list[str] = []
    for pkg_rel, tree_rel in SYNCED:
        src = PACKAGE_MODULES / pkg_rel
        dst = TREE_MODULES / tree_rel
        if not src.is_file():
            print(f"missing package file: {src}", file=sys.stderr)
            return 2
        if args.check:
            if not dst.is_file() or src.read_bytes() != dst.read_bytes():
                drifted.append(f"{tree_rel} != package {pkg_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"synced {pkg_rel} -> {dst.relative_to(REPO_ROOT)}")

    if args.check and drifted:
        print("drift detected:", file=sys.stderr)
        for row in drifted:
            print(f"  - {row}", file=sys.stderr)
        print("run: python embodied/sync_modules.py", file=sys.stderr)
        return 1
    if args.check:
        print("in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
