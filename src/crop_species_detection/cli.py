"""Command-line entry point: ``crop-species-detect``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import SUPPORTED_EXTENSIONS
from .pipeline import CropSpeciesDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop-species-detect",
        description="Run the hierarchical crop species detection pipeline on one or more images.",
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image file(s) (.jpg, .jpeg, .png, .webp).",
    )

    parser.add_argument(
        "--stage1",
        default=None,
        help=(
            "Path to the Stage 1 checkpoint. "
            "If omitted, it is automatically downloaded from Hugging Face."
        ),
    )

    parser.add_argument(
        "--stage2",
        default=None,
        help=(
            "Path to the Stage 2 checkpoint. "
            "If omitted, it is automatically downloaded from Hugging Face."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON results instead of a human-readable summary.",
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bad_images = [
        img
        for img in args.images
        if not str(img).lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if bad_images:
        parser.error(
            f"Unsupported file type(s): {bad_images}. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )

    for img in args.images:
        if not Path(img).exists():
            parser.error(f"Image not found: {img}")

    detector = CropSpeciesDetector.from_pretrained(
        stage1_path=args.stage1,
        stage2_path=args.stage2,
        device=args.device,
    )

    results = {
        img: detector.predict(img)
        for img in args.images
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for img, result in results.items():
        print(f"\n{img}")
        print(
            f"  Stage 1: "
            f"{result['stage1_prediction']} "
            f"({result['stage1_confidence'] * 100:.2f}%)"
        )

        if result["stage2_used"]:
            print(
                f"  Stage 2: "
                f"{result['stage2_prediction']} "
                f"({result['stage2_confidence'] * 100:.2f}%)"
            )

        print(
            f"  Final:   "
            f"{result['final_prediction']} "
            f"({result['final_confidence'] * 100:.2f}%)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
