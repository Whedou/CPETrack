import math

from lib.models.cpetrack import build_cpetrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.text_semantic import TextFeatureCache, clean_sequence_name


class CPETrack(BaseTracker):
    def __init__(self, params):
        super(CPETrack, self).__init__(params)
        network = build_cpetrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
        try:
            network.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print("Strict checkpoint loading failed: {}".format(e))
            print("Retry loading with strict=False for optional text-router compatibility.")
            missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
            print("missing_keys: ", missing_keys)
            print("unexpected_keys:", unexpected_keys)

        depth = 12
        self.keep_rate = [x for x in torch.linspace(0.7, 1, depth // 4)][::-1]
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM(mean=self.cfg.DATA.MEAN, std=self.cfg.DATA.STD)
        self.state = None
        text_cfg = getattr(getattr(self.cfg, "MODEL", None), "TEXT", None)
        self.text_feature_cache = TextFeatureCache.from_cfg(text_cfg)
        self.text_feature = None
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True
        ).cuda()

        self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS
        self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD

        # for debug
        self.debug = getattr(params, 'debug', 0)
        self.use_visdom = getattr(params, 'use_visdom', False)
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        self.z_dict = {}
        self.z_patch_arr = None
        self.box_mask_z = None

    def initialize(self, image, info: dict):
        text_name = (
            info.get('sequence_name')
            or info.get('object_class_name')
            or info.get('target_name')
        )
        if text_name is not None and 'object_class_name' not in info:
            info['object_class_name'] = clean_sequence_name(text_name)
        if self.text_feature_cache.enabled:
            device = next(self.network.parameters()).device
            self.text_feature = self.text_feature_cache.get_batch(
                [text_name],
                device=device,
                training=False,
            )
        else:
            self.text_feature = None

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size
        )

        self.z_patch_arr = z_patch_arr

        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            # PreprocessorMM 返回纯 tensor，因此这里直接存 template
            self.z_dict = [template] * self.num_template

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.box_mask_z = None

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, input_state=None, prev_output=None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.params.search_factor,
            output_sz=self.params.search_size
        )  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            # PreprocessorMM 返回纯 tensor，因此这里直接用 search
            x_dict = [search]

            out_dict = self.network.forward(
                template=self.z_dict,
                search=x_dict,
                ce_template_mask=self.box_mask_z,
                keep_rate=self.keep_rate,
                training=False,
                text_feature=self.text_feature,
            )

            # 兼容某些版本 forward(training=False) 仍返回 list[dict]
            if isinstance(out_dict, list):
                out_dict = out_dict[0]

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(
            response,
            out_dict['size_map'],
            out_dict['offset_map']
        )
        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (
            pred_boxes.mean(dim=0) * self.params.search_size / resize_factor
        ).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor),
            H, W, margin=10
        )

        conf_score = None
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):

                z_patch_arr, resize_factor, z_amask_arr = sample_target(
                    image,
                    self.state,
                    self.params.template_factor,
                    output_sz=self.params.template_size
                )
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(
                    image_BGR,
                    (int(x1), int(y1)),
                    (int(x1 + w), int(y1 + h)),
                    color=(0, 0, 255),
                    thickness=2
                )
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                gt_box = info['gt_bbox'].tolist() if (info is not None and 'gt_bbox' in info) else self.state
                self.visdom.register((image, gt_box, self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(
                    torch.from_numpy(x_patch_arr).permute(2, 0, 1),
                    'image', 1, 'search_region'
                )
                self.visdom.register(
                    torch.from_numpy(self.z_patch_arr).permute(2, 0, 1),
                    'image', 1, 'template'
                )
                self.visdom.register(
                    pred_score_map.view(self.feat_sz, self.feat_sz),
                    'heatmap', 1, 'score_map'
                )
                self.visdom.register(
                    (pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz),
                    'heatmap', 1, 'score_map_hann'
                )

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(
                        torch.from_numpy(masked_search).permute(2, 0, 1),
                        'image', 1, 'masked_search'
                    )

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor
            )
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N,)
            if conf_score is not None:
                return {
                    "target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": conf_score.cpu().numpy()[0][0]
                }
            else:
                return {
                    "target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": None
                }
        else:
            if conf_score is not None:
                return {
                    "target_bbox": self.state,
                    "best_score": conf_score.cpu().numpy()[0][0]
                }
            else:
                return {
                    "target_bbox": self.state,
                    "best_score": None
                }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) -> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return CPETrack
