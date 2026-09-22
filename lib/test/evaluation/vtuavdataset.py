import os

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _image_key(name):
    stem = os.path.splitext(name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


class VTUAVDataset(BaseDataset):
    """VTUAV-ST/LT test data in the RGB reference coordinate system."""

    def __init__(self, subset):
        super().__init__()
        if subset not in {"st", "lt"}:
            raise ValueError("VTUAV has no {} subset".format(subset))

        subset_dir = "test_ST" if subset == "st" else "test_LT"
        self.base_path = os.path.join(self.env_settings.vtuav_path, subset_dir)
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError("VTUAV subset directory not found: {}".format(self.base_path))
        self.subset = subset
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = os.path.join(self.base_path, *sequence_name.split("/"))
        rgb_gt = load_text(
            os.path.join(sequence_path, "rgb.txt"),
            delimiter=[" ", "\t", ","],
            dtype=np.float64,
        )
        rgb_gt = np.atleast_2d(rgb_gt)[:, :4]

        rgb_frames = self._frames(sequence_path, "rgb")
        ir_frames = self._frames(sequence_path, "ir")
        if len(rgb_frames) != len(ir_frames):
            raise ValueError(
                "RGB/IR frame-count mismatch in {}: {} vs {}".format(
                    sequence_name, len(rgb_frames), len(ir_frames)
                )
            )

        frames = list(zip(rgb_frames, ir_frames))
        frame_gt = self._expand_sparse_gt(rgb_gt, len(frames))
        return Sequence(sequence_name, frames, "vtuav_{}".format(self.subset), frame_gt)

    @staticmethod
    def _frames(sequence_path, modality):
        folder = os.path.join(sequence_path, modality)
        names = sorted(
            [name for name in os.listdir(folder) if os.path.splitext(name)[1].lower() in IMAGE_EXTS],
            key=_image_key,
        )
        return [os.path.join(folder, name) for name in names]

    @staticmethod
    def _expand_sparse_gt(gt, frame_count):
        if len(gt) == frame_count:
            return gt
        if (len(gt) - 1) * 10 >= frame_count:
            raise ValueError(
                "Sparse VTUAV annotations ({}) do not fit {} image frames".format(len(gt), frame_count)
            )
        # Ground truth is only used for initialization/debug in the tracker loop.
        # Repeat each sparse RGB box until the next annotated timestamp.
        return np.repeat(gt, 10, axis=0)[:frame_count]

    def _get_sequence_list(self):
        sequences = []
        for current_root, dirs, _ in os.walk(self.base_path):
            if "rgb" in dirs and "ir" in dirs and os.path.isfile(os.path.join(current_root, "rgb.txt")):
                relative = os.path.relpath(current_root, self.base_path).replace("\\", "/")
                sequences.append(relative)
        sequences.sort()
        if not sequences:
            raise RuntimeError("No VTUAV sequences found under {}".format(self.base_path))
        return sequences

    def __len__(self):
        return len(self.sequence_list)
