import os
import random
from collections import OrderedDict

import numpy as np
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class VTUAV(BaseVideoDataset):
    """VTUAV-ST training set with one annotation for every 10 video frames."""

    def __init__(self, root=None, dtype="rgbrgb", seq_ids=None, data_fraction=None):
        root = env_settings().vtuav_train_dir if root is None else root
        super().__init__("VTUAV", root)
        self.dtype = dtype
        init_frame_path = os.path.join(os.path.dirname(__file__), "init_frame.npy")
        if not os.path.isfile(init_frame_path):
            raise FileNotFoundError("Missing VTUAV frame-offset file: {}".format(init_frame_path))
        self.init_idx = np.load(init_frame_path, allow_pickle=True).item()
        self._frame_maps = {}
        self.sequence_list = self._get_sequence_list()

        if seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        if data_fraction is not None:
            count = int(len(self.sequence_list) * data_fraction)
            self.sequence_list = random.sample(self.sequence_list, count)

    def get_name(self):
        return "vtuav"

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError("VTUAV training directory not found: {}".format(self.root))
        sequences = [
            name for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]
        sequences.sort()
        if not sequences:
            raise RuntimeError("No VTUAV sequences found under {}".format(self.root))
        return sequences

    def _image_map(self, folder):
        if folder in self._frame_maps:
            return self._frame_maps[folder]

        names = [
            name for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        frame_map = {}
        for name in names:
            stem = os.path.splitext(name)[0]
            if not stem.isdigit():
                continue
            frame_number = int(stem)
            if frame_number in frame_map:
                raise ValueError("Duplicate VTUAV frame number {} in {}".format(frame_number, folder))
            frame_map[frame_number] = name
        if not frame_map:
            raise RuntimeError("No numerically named VTUAV images found in {}".format(folder))
        self._frame_maps[folder] = frame_map
        return frame_map

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    @staticmethod
    def _read_bb_anno(seq_path):
        anno_file = os.path.join(seq_path, "rgb.txt")
        if not os.path.isfile(anno_file):
            raise FileNotFoundError("Missing VTUAV annotation: {}".format(anno_file))
        boxes = np.atleast_2d(np.loadtxt(anno_file, dtype=np.float32))
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError("Invalid VTUAV annotation shape {} in {}".format(boxes.shape, anno_file))
        return torch.tensor(boxes[:, :4])

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(self._get_sequence_path(seq_id))
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {"bbox": bbox, "valid": valid, "visible": valid.clone().byte()}

    def _get_frame_path(self, seq_path, frame_id):
        rgb_dir = os.path.join(seq_path, "rgb")
        ir_dir = os.path.join(seq_path, "ir")
        rgb_frames = self._image_map(rgb_dir)
        ir_frames = self._image_map(ir_dir)
        sequence_name = os.path.basename(os.path.normpath(seq_path))
        offset = int(self.init_idx.get(sequence_name, 0))
        image_number = frame_id * 10 + offset
        if image_number not in rgb_frames or image_number not in ir_frames:
            raise IndexError(
                "Sparse VTUAV annotation {} in {} maps to image number {}, "
                "which is missing from RGB or IR".format(frame_id, sequence_name, image_number)
            )
        return (
            os.path.join(rgb_dir, rgb_frames[image_number]),
            os.path.join(ir_dir, ir_frames[image_number]),
        )

    def _get_frame(self, seq_path, frame_id):
        rgb_path, ir_path = self._get_frame_path(seq_path, frame_id)
        return get_x_frame(rgb_path, ir_path, dtype=self.dtype)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        sequence_name = self.sequence_list[seq_id]
        frame_list = [self._get_frame(seq_path, frame_id) for frame_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {
            key: [value[frame_id, ...].clone() for frame_id in frame_ids]
            for key, value in anno.items()
        }
        object_meta = OrderedDict({
            "object_class_name": None,
            "sequence_name": sequence_name,
            "motion_class": None,
            "major_class": None,
            "root_class": None,
            "motion_adverb": None,
        })
        return frame_list, anno_frames, object_meta
