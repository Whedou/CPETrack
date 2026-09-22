#!/usr/bin/env python3
"""Validate VTUAV sparse training annotations against RGB/IR image timestamps."""

import argparse
import os
import sys

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_numbers(folder):
    numbers = set()
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS and stem.isdigit():
            numbers.add(int(stem))
    return numbers


def annotation_count(path):
    boxes = np.loadtxt(path, dtype=np.float64)
    return len(np.atleast_2d(boxes))


def validate(dataset_root, init_frame_path):
    offsets = np.load(init_frame_path, allow_pickle=True).item()
    errors = []
    sequence_count = 0

    for sequence_name in sorted(os.listdir(dataset_root)):
        sequence_dir = os.path.join(dataset_root, sequence_name)
        if not os.path.isdir(sequence_dir):
            continue
        sequence_count += 1
        rgb_dir = os.path.join(sequence_dir, "rgb")
        ir_dir = os.path.join(sequence_dir, "ir")
        anno_path = os.path.join(sequence_dir, "rgb.txt")
        if not all(os.path.exists(path) for path in (rgb_dir, ir_dir, anno_path)):
            errors.append("{}: missing rgb/, ir/, or rgb.txt".format(sequence_name))
            continue

        rgb_numbers = frame_numbers(rgb_dir)
        ir_numbers = frame_numbers(ir_dir)
        offset = int(offsets.get(sequence_name, 0))
        try:
            count = annotation_count(anno_path)
        except Exception as exc:
            errors.append("{}: cannot read rgb.txt ({})".format(sequence_name, exc))
            continue

        missing = [
            offset + index * 10
            for index in range(count)
            if offset + index * 10 not in rgb_numbers or offset + index * 10 not in ir_numbers
        ]
        if missing:
            errors.append(
                "{}: {} sparse timestamps missing (first: {})".format(
                    sequence_name, len(missing), missing[:5]
                )
            )

    unused_offsets = sorted(set(offsets) - set(os.listdir(dataset_root)))
    print("VTUAV training sequences: {}".format(sequence_count))
    print("Offset entries: {} (unused for this split: {})".format(len(offsets), len(unused_offsets)))
    if errors:
        print("Validation errors: {}".format(len(errors)))
        for error in errors[:50]:
            print("  " + error)
        if len(errors) > 50:
            print("  ... {} more".format(len(errors) - 50))
        return 1

    print("VTUAV sparse frame mapping: OK")
    return 0


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="/data/pudata/VTUAV/train_ST/train_ST")
    parser.add_argument(
        "--init_frame",
        default=os.path.join(project_root, "lib", "train", "dataset", "init_frame.npy"),
    )
    args = parser.parse_args()
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError("VTUAV training directory not found: {}".format(args.dataset_root))
    sys.exit(validate(args.dataset_root, args.init_frame))


if __name__ == "__main__":
    main()
