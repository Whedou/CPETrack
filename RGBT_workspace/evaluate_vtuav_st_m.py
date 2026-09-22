#!/usr/bin/env python3
"""Evaluate VTUAV-ST results with the RGBT MPR/MSR protocol."""

import argparse
import os

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_count(sequence_dir):
    rgb_dir = os.path.join(sequence_dir, "rgb")
    return sum(
        os.path.splitext(name)[1].lower() in IMAGE_EXTS
        for name in os.listdir(rgb_dir)
    )


def discover_sequences(dataset_root):
    sequences = []
    for current_root, dirs, _ in os.walk(dataset_root):
        if "rgb" in dirs and "ir" in dirs and os.path.isfile(os.path.join(current_root, "rgb.txt")):
            sequences.append(os.path.relpath(current_root, dataset_root).replace("\\", "/"))
    return sorted(sequences)


def load_boxes(path):
    boxes = np.loadtxt(path, dtype=np.float64)
    boxes = np.atleast_2d(boxes)
    if boxes.shape[1] < 4:
        raise ValueError("Expected xywh boxes in {}, got {}".format(path, boxes.shape))
    return boxes[:, :4]


def align_predictions(pred, gt, frame_count):
    if len(pred) == len(gt):
        return pred
    if len(pred) != frame_count:
        raise ValueError(
            "Prediction length {} matches neither frames {} nor annotations {}".format(
                len(pred), frame_count, len(gt)
            )
        )

    # VTUAV-ST annotations are provided every 10 frames.
    indices = np.arange(len(gt), dtype=np.int64) * 10
    if indices[-1] >= len(pred):
        raise ValueError(
            "Sparse annotation index {} exceeds prediction length {}".format(indices[-1], len(pred))
        )
    return pred[indices]


def center_error(pred, gt):
    pred_center = pred[:, :2] + 0.5 * pred[:, 2:]
    gt_center = gt[:, :2] + 0.5 * gt[:, 2:]
    return np.linalg.norm(pred_center - gt_center, axis=1)


def overlap_ratio(pred, gt):
    pred_tl = pred[:, :2]
    pred_br = pred[:, :2] + pred[:, 2:]
    gt_tl = gt[:, :2]
    gt_br = gt[:, :2] + gt[:, 2:]
    inter_tl = np.maximum(pred_tl, gt_tl)
    inter_br = np.minimum(pred_br, gt_br)
    inter_wh = np.maximum(0.0, inter_br - inter_tl)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def evaluate(dataset_root, results_root):
    sequences = discover_sequences(dataset_root)
    if not sequences:
        raise RuntimeError("No VTUAV-ST sequences found under {}".format(dataset_root))

    precision_thresholds = np.linspace(0.0, 50.0, 51)
    success_thresholds = np.linspace(0.0, 1.0, 21)
    precision_curves = []
    success_curves = []
    annotated_frames = 0
    missing = []
    for sequence_name in sequences:
        sequence_dir = os.path.join(dataset_root, *sequence_name.split("/"))
        result_name = sequence_name.rsplit("/", 1)[-1] + ".txt"
        result_path = os.path.join(results_root, result_name)
        if not os.path.isfile(result_path):
            missing.append(sequence_name)
            continue

        gt_rgb = load_boxes(os.path.join(sequence_dir, "rgb.txt"))
        gt_ir_path = os.path.join(sequence_dir, "ir.txt")
        if not os.path.isfile(gt_ir_path):
            raise FileNotFoundError("Missing infrared annotations: {}".format(gt_ir_path))
        gt_ir = load_boxes(gt_ir_path)
        if len(gt_rgb) != len(gt_ir):
            raise ValueError(
                "RGB/IR annotation length mismatch for {}: {} vs {}".format(
                    sequence_name, len(gt_rgb), len(gt_ir)
                )
            )

        pred = load_boxes(result_path)
        pred = align_predictions(pred, gt_rgb, image_count(sequence_dir))
        if len(pred) != len(gt_rgb):
            raise ValueError(
                "Length mismatch for {}: {} vs {}".format(
                    sequence_name, len(pred), len(gt_rgb)
                )
            )

        valid_rgb = (gt_rgb[:, 2] > 0) & (gt_rgb[:, 3] > 0)
        valid_ir = (gt_ir[:, 2] > 0) & (gt_ir[:, 3] > 0)
        valid = valid_rgb | valid_ir
        if not np.any(valid):
            continue

        rgb_errors = np.full(len(pred), np.inf, dtype=np.float64)
        ir_errors = np.full(len(pred), np.inf, dtype=np.float64)
        rgb_errors[valid_rgb] = center_error(pred[valid_rgb], gt_rgb[valid_rgb])
        ir_errors[valid_ir] = center_error(pred[valid_ir], gt_ir[valid_ir])
        max_precision_errors = np.minimum(rgb_errors, ir_errors)[valid]

        rgb_overlaps = np.full(len(pred), -np.inf, dtype=np.float64)
        ir_overlaps = np.full(len(pred), -np.inf, dtype=np.float64)
        rgb_overlaps[valid_rgb] = overlap_ratio(pred[valid_rgb], gt_rgb[valid_rgb])
        ir_overlaps[valid_ir] = overlap_ratio(pred[valid_ir], gt_ir[valid_ir])
        max_success_overlaps = np.maximum(rgb_overlaps, ir_overlaps)[valid]

        precision_curves.append(
            np.array([(max_precision_errors <= threshold).mean() for threshold in precision_thresholds])
        )
        success_curves.append(
            np.array([(max_success_overlaps > threshold).mean() for threshold in success_thresholds])
        )
        annotated_frames += int(valid.sum())

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            "Missing results for {}/{} sequences (first: {})".format(len(missing), len(sequences), preview)
        )
    if not precision_curves:
        raise RuntimeError("No valid VTUAV-ST annotations were evaluated")

    mean_precision_curve = np.mean(np.stack(precision_curves), axis=0)
    mean_success_curve = np.mean(np.stack(success_curves), axis=0)
    mpr = mean_precision_curve[np.where(precision_thresholds == 20.0)[0][0]]
    msr = mean_success_curve.mean()

    print("VTUAV-ST sequences: {}".format(len(sequences)))
    print("VTUAV-ST annotated frames: {}".format(annotated_frames))
    print("VTUAV-ST MPR@20: {:.4f}".format(mpr))
    print("VTUAV-ST MSR(AUC): {:.4f}".format(msr))


def main():
    parser = argparse.ArgumentParser(description="Evaluate VTUAV-ST with MPR/MSR")
    parser.add_argument("--dataset_root", default="/data/pudata/VTUAV/test_ST")
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()
    evaluate(args.dataset_root, args.results_root)


if __name__ == "__main__":
    main()
