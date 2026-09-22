import math
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.cpetrack.vit_cpetrack import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class CPETrack(nn.Module):
    """
    ViT backbone + prediction head.
    """

    def __init__(
        self,
        transformer,
        box_head,
        cfg,
        aux_loss=False,
        head_type="CORNER",
    ):
        super().__init__()
        hidden_dim = transformer.embed_dim

        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type

        self.template_number = cfg.DATA.TEMPLATE.NUMBER

        if head_type in ["CORNER", "CENTER"]:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def _build_search_memory_from_cpetrack(
        self,
        x: torch.Tensor,
        num_template_token: int,
        num_search_token: int,
    ):
        """
        Keep only search tokens from the single fused branch layout.

        x: [B, template_tokens + search_tokens, C]
        """
        B, N, C = x.shape

        num_prefix_token = N - num_template_token - num_search_token
        assert num_prefix_token == 0, (
            f"invalid token layout: N={N}, "
            f"template={num_template_token}, search={num_search_token}"
        )
        search_start = num_template_token
        search_end = search_start + num_search_token
        assert search_end <= N, (
            f"slice out of range: template={num_template_token}, "
            f"search={num_search_token}, N={N}"
        )

        feat_last = x[:, search_start:search_end, :]  # [B, HW, C]

        feat_sz = int(math.sqrt(num_search_token))
        assert feat_sz * feat_sz == num_search_token, (
            f"num_search_token={num_search_token} cannot form square feature map"
        )

        return feat_last.view(B, feat_sz, feat_sz, C).permute(0, 3, 1, 2).contiguous()

    def _forward_head_with_feature(self, feat: torch.Tensor, batch_size=None, gt_score_map=None):
        """
        Run box head directly on search feature.
        """
        bfeat = feat.shape[0]

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)

            if batch_size is None:
                outputs_coord_new = outputs_coord.view(bfeat, 1, 4)
            else:
                num_queries = bfeat // batch_size
                outputs_coord_new = outputs_coord.view(batch_size, num_queries, 4)

            return {
                "pred_boxes": outputs_coord_new,
                "score_map": score_map,
            }

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(feat, gt_score_map)

            if batch_size is None:
                outputs_coord_new = bbox.view(bfeat, 1, 4)
            else:
                num_queries = bfeat // batch_size
                outputs_coord_new = bbox.view(batch_size, num_queries, 4)

            return {
                "pred_boxes": outputs_coord_new,
                "score_map": score_map_ctr,
                "size_map": size_map,
                "offset_map": offset_map,
            }

        raise NotImplementedError(f"Unknown head type {self.head_type}")

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        ce_template_mask=None,
        return_last_attn=False,
        tgt_pre=None,
        keep_rate=None,
        training=True,
        text_feature=None,
    ):
        if not isinstance(search, list):
            search = [search]

        out_dict_list = []

        for i in range(len(search)):
            B = search[i].shape[0]

            x, aux_dict, len_zx = self.backbone(
                z=template,
                x=search[i],
                track_query_before=None,
                keep_rate=keep_rate,
                ce_template_mask=ce_template_mask,
                return_last_attn=return_last_attn,
                text_feature=text_feature,
            )

            num_template_token = len_zx[0]
            num_search_token = len_zx[1]

            search_feat = self._build_search_memory_from_cpetrack(
                x, num_template_token, num_search_token
            )

            out = self._forward_head_with_feature(search_feat, batch_size=B)

            out.update(aux_dict)
            out["backbone_feat"] = x

            out_dict_list.append(out)

        if training:
            return out_dict_list
        return out_dict_list[0] if len(out_dict_list) > 0 else {}

    def forward_head(self, cat_feature, gt_score_map=None):
        raise RuntimeError(
            "CPETrack runs the prediction head inside forward(). "
            "Please call forward() instead of forward_head()."
        )


def build_cpetrack(cfg, training=True):
    pretrained_path = "/data/wsk/project/STTrack/pretrained/"

    if cfg.MODEL.PRETRAIN_FILE and ("STTrack" not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print("Load pretrained model from: " + pretrained)
    else:
        pretrained = ""

    if cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224":
        text_cfg = getattr(cfg.MODEL, "TEXT", None)
        text_feature_dim = 0
        if text_cfg is not None and getattr(text_cfg, "ENABLE", False):
            text_feature_dim = int(getattr(text_cfg, "FEATURE_DIM", 512))
        backbone = vit_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            cross_loc=cfg.MODEL.BACKBONE.CROSS_LOC,
            drop_path=cfg.TRAIN.CROSS_DROP_PATH,
            text_feature_dim=text_feature_dim,
        )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = CPETrack(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if "OSTrack" in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys:", unexpected_keys)
        print("Load pretrained model from: " + cfg.MODEL.PRETRAIN_FILE)

    return model
