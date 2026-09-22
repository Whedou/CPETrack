import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu
from lib.utils.text_semantic import TextFeatureCache


class CPETrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        text_cfg = getattr(getattr(cfg, "MODEL", None), "TEXT", None)
        self.text_feature_cache = TextFeatureCache.from_cfg(text_cfg)

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 6, H, W)
            search_images: (N_s, batch, 6, H, W)
        returns:
            loss    - the training loss
            status  - dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and multiple search regions

        template_list = []
        search_list = []

        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:]
            )  # (batch, 6, H, W)
            template_list.append(template_img)

        for i in range(self.settings.num_search):
            search_img = data['search_images'][i].view(
                -1, *data['search_images'].shape[2:]
            )  # (batch, 6, H, W)
            search_list.append(search_img)

        box_mask_z = None

        drop_start_epoch = self.cfg.TRAIN.LR_DROP_EPOCH
        total_epoch = self.cfg.TRAIN.EPOCH
        keep_rate = adjust_keep_rate(
            data['epoch'],
            warmup_epochs=drop_start_epoch,
            total_epochs=total_epoch
        )

        text_feature = None
        if self.text_feature_cache.enabled:
            text_names = data.get('sequence_name', None)
            if text_names is None:
                text_names = data.get('test_class', None)
            text_feature = self.text_feature_cache.get_batch(
                text_names,
                device=search_list[0].device,
                training=self.net.training,
            )

        out_dict = self.net(
            template=template_list,
            search=search_list,
            ce_template_mask=box_mask_z,
            keep_rate=keep_rate,
            return_last_attn=False,
            training=True,
            text_feature=text_feature,
        )

        return out_dict

    def _build_search_token_mask(self, gt_bbox, num_tokens, device):
        if gt_bbox is None:
            return None

        gt_bbox = gt_bbox.to(device=device)
        boxes = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)
        dtype = boxes.dtype

        grid_size = int(self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE)
        grid_h = grid_w = grid_size
        if grid_h * grid_w != num_tokens:
            side = int(num_tokens ** 0.5)
            if side * side != num_tokens:
                return None
            grid_h = grid_w = side

        y = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h
        x = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w
        centers_y = y[:, None].expand(grid_h, grid_w).reshape(1, -1)
        centers_x = x[None, :].expand(grid_h, grid_w).reshape(1, -1)

        mask = (
            (centers_x >= boxes[:, 0:1])
            & (centers_x <= boxes[:, 2:3])
            & (centers_y >= boxes[:, 1:2])
            & (centers_y <= boxes[:, 3:4])
        )

        empty = mask.sum(dim=1) == 0
        if empty.any():
            box_cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).unsqueeze(1)
            box_cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).unsqueeze(1)
            dist = (centers_x - box_cx).pow(2) + (centers_y - box_cy).pow(2)
            nearest = dist.argmin(dim=1)
            fallback = F.one_hot(nearest, num_classes=num_tokens).to(torch.bool)
            mask = torch.where(empty.unsqueeze(1), fallback, mask)

        return mask

    def _collect_layer_similarity_stats(self, pred, gt_bbox, frame_idx):
        stats = {}
        layer_tokens = pred.get("layer_search_tokens", None)
        if layer_tokens is None or layer_tokens.dim() != 4:
            return stats

        with torch.no_grad():
            layer_tokens = layer_tokens.float()
            _, num_layers, num_tokens, _ = layer_tokens.shape
            final_tokens = layer_tokens[:, -1]
            final_norm = F.normalize(final_tokens, dim=-1)

            target_mask = self._build_search_token_mask(
                gt_bbox,
                num_tokens,
                layer_tokens.device,
            )
            if target_mask is not None:
                target_mask_f = target_mask.to(dtype=layer_tokens.dtype)
                target_token_count = target_mask_f.sum().clamp_min(1.0)
                bg_mask = ~target_mask
                bg_mask_f = bg_mask.to(dtype=layer_tokens.dtype)
                target_count_per_sample = target_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                bg_count_per_sample = bg_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                target_mask_f = None
                bg_mask_f = None
                target_token_count = None
                target_count_per_sample = None
                bg_count_per_sample = None

            for layer_idx in range(num_layers):
                current = layer_tokens[:, layer_idx]
                current_norm = F.normalize(current, dim=-1)
                sim_to_final = (current_norm * final_norm).sum(dim=-1)
                stats[f"{frame_idx}frame_LayerSim/all_to_final_L{layer_idx:02d}"] = sim_to_final.mean().item()

                if target_mask_f is not None:
                    target_sim = (sim_to_final * target_mask_f).sum() / target_token_count
                    stats[f"{frame_idx}frame_LayerSim/target_to_final_L{layer_idx:02d}"] = target_sim.item()

                    target_feat = (current * target_mask_f.unsqueeze(-1)).sum(dim=1) / target_count_per_sample
                    bg_feat = (current * bg_mask_f.unsqueeze(-1)).sum(dim=1) / bg_count_per_sample
                    target_bg_cos = (
                        F.normalize(target_feat, dim=-1)
                        * F.normalize(bg_feat, dim=-1)
                    ).sum(dim=-1).mean()
                    target_bg_l2 = (target_feat - bg_feat).norm(dim=-1).mean()
                    stats[f"{frame_idx}frame_LayerSep/target_bg_cos_L{layer_idx:02d}"] = target_bg_cos.item()
                    stats[f"{frame_idx}frame_LayerSep/target_bg_l2_L{layer_idx:02d}"] = target_bg_l2.item()

            for layer_idx in range(1, num_layers):
                previous = layer_tokens[:, layer_idx - 1]
                current = layer_tokens[:, layer_idx]
                adjacent_delta = (current - previous).norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-6)
                stats[f"{frame_idx}frame_LayerDelta/all_L{layer_idx - 1:02d}_to_L{layer_idx:02d}"] = adjacent_delta.mean().item()

                if target_mask_f is not None:
                    target_delta = (adjacent_delta * target_mask_f).sum() / target_token_count
                    stats[f"{frame_idx}frame_LayerDelta/target_L{layer_idx - 1:02d}_to_L{layer_idx:02d}"] = target_delta.item()

        return stats
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()

        gt_gaussian_maps_list = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )

        # 这里按“每个 search frame 对应一个输出 dict”计算损失
        for i in range(len(pred_dict)):
            # GT
            gt_bbox = gt_dict['search_anno'][i]  # (batch, 4), x1y1wh
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)  # (B,1,H,W)

            # Pred boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")

            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) -> (BN,4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
            gt_boxes_vec = gt_boxes_vec.clamp(min=0.0, max=1.0)

            # giou / iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except Exception:
                giou_loss = torch.tensor(0.0, device=pred_boxes_vec.device)
                iou = torch.tensor(0.0, device=pred_boxes_vec.device)
            loss_dict['giou'] = giou_loss

            # l1
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            loss_dict['l1'] = l1_loss

            # focal / score map
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss

            # weighted sum
            loss = sum(
                loss_dict[k] * self.loss_weight[k]
                for k in loss_dict.keys()
                if k in self.loss_weight
            )
            total_loss += loss

            if return_status:
                mean_iou = iou.detach().mean()
                status = {
                    f"{i}frame_Loss/total": loss.item(),
                    f"{i}frame_Loss/giou": giou_loss.item(),
                    f"{i}frame_Loss/l1": l1_loss.item(),
                    f"{i}frame_Loss/location": location_loss.item(),
                    f"{i}frame_IoU": mean_iou.item(),
                }
                status.update(self._collect_layer_similarity_stats(pred_dict[i], gt_bbox, i))
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
