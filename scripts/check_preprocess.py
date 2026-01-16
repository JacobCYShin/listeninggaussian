#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


REQUIRED_FILES = [
    "aud.wav",
    "au.csv",
    "track_params.pt",
    "transforms_train.json",
    "transforms_val.json",
    "flame_params.npy",
    "flame_params_stats.npz",
]

REQUIRED_DIRS = [
    "ori_imgs",
    "parsing",
    "gt_imgs",
    "torso_imgs",
]


def check_path(path: Path) -> bool:
    if not path.exists():
        print(f"[MISS] {path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        print(f"[ZERO] {path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check preprocess outputs for training.")
    parser.add_argument("--data", required=True, help="Dataset folder name under data/, e.g. test")
    args = parser.parse_args()

    base = Path("data") / args.data
    if not base.exists():
        print(f"[MISS] {base}")
        raise SystemExit(1)

    ok = True

    for d in REQUIRED_DIRS:
        ok = check_path(base / d) and ok

    for f in REQUIRED_FILES:
        ok = check_path(base / f) and ok

    if ok:
        print("[OK] All required preprocess outputs found.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
