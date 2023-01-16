import torch
from torch.nn import functional as F
import numpy as np

def _nms(heat, kernel=1):
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feature(feature, ind):
    feature = feature.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] => [B, H, W, C]
    feature = feature.view(feature.size(0), -1, feature.size(3))  # [B, H, W, C] => [B, H x W, C]
    ind = ind[:, :, None].expand(ind.shape[0], ind.shape[1], feature.shape[-1])  # [B, num_obj] => [B, num_obj, C]
    feature = feature.gather(1, ind)  # [B, H x W, C] => [B, num_obj, C]
    return feature

def _topk(score_map, K=20):
    batch, cat, height, width = score_map.size()

    topk_scores, topk_inds = torch.topk(score_map.view(batch, -1), K)

    topk_classes = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_classes, topk_ys, topk_xs

def decode(hmap_tl, hmap_br, hmap_ct,
           embd_tl, embd_br,
           regs_tl, regs_br, regs_ct,
           K, kernel, ae_threshold, num_dets=1000):
    batch, _, _, _ = hmap_tl.shape

    hmap_tl = torch.sigmoid(hmap_tl)
    hmap_br = torch.sigmoid(hmap_br)
    hmap_ct = torch.sigmoid(hmap_ct)

    # perform nms on heatmaps
    hmap_tl = _nms(hmap_tl, kernel=kernel)
    hmap_br = _nms(hmap_br, kernel=kernel)
    hmap_ct = _nms(hmap_ct, kernel=kernel)

    scores_tl, inds_tl, clses_tl, ys_tl, xs_tl = _topk(hmap_tl, K=K)
    scores_br, inds_br, clses_br, ys_br, xs_br = _topk(hmap_br, K=K)
    scores_ct, inds_ct, clses_ct, ys_ct, xs_ct = _topk(hmap_ct, K=K)

    xs_tl = xs_tl.view(batch, K, 1).expand(batch, K, K)
    ys_tl = ys_tl.view(batch, K, 1).expand(batch, K, K)
    xs_br = xs_br.view(batch, 1, K).expand(batch, K, K)
    ys_br = ys_br.view(batch, 1, K).expand(batch, K, K)
    xs_ct = xs_ct.view(batch, 1, K).expand(batch, K, K)
    ys_ct = ys_ct.view(batch, 1, K).expand(batch, K, K)

    if regs_tl is not None and regs_br is not None:
        regs_tl = _tranpose_and_gather_feature(regs_tl, inds_tl)
        regs_br = _tranpose_and_gather_feature(regs_br, inds_br)
        regs_ct = _tranpose_and_gather_feature(regs_ct, inds_ct)
        regs_tl = regs_tl.view(batch, K, 1, 2)
        regs_br = regs_br.view(batch, 1, K, 2)
        regs_ct = regs_ct.view(batch, 1, K, 2)

        xs_tl = xs_tl + regs_tl[..., 0]
        ys_tl = ys_tl + regs_tl[..., 1]
        xs_br = xs_br + regs_br[..., 0]
        ys_br = ys_br + regs_br[..., 1]
        xs_ct = xs_ct + regs_ct[..., 0]
        ys_ct = ys_ct + regs_ct[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((xs_tl, ys_tl, xs_br, ys_br), dim=3)

    embd_tl = _tranpose_and_gather_feature(embd_tl, inds_tl)
    embd_br = _tranpose_and_gather_feature(embd_br, inds_br)
    embd_tl = embd_tl.view(batch, K, 1)
    embd_br = embd_br.view(batch, 1, K)
    dists = torch.abs(embd_tl - embd_br)

    scores_tl = scores_tl.view(batch, K, 1).expand(batch, K, K)
    scores_br = scores_br.view(batch, 1, K).expand(batch, K, K)
    scores = (scores_tl + scores_br) / 2

    # reject boxes based on classes
    clses_tl = clses_tl.view(batch, K, 1).expand(batch, K, K)
    clses_br = clses_br.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (clses_tl != clses_br)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (xs_br < xs_tl)
    height_inds = (ys_br < ys_tl)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    classes = clses_tl.contiguous().view(batch, -1, 1)
    classes = _gather_feat(classes, inds).float()

    scores_tl = scores_tl.contiguous().view(batch, -1, 1)
    scores_br = scores_br.contiguous().view(batch, -1, 1)
    scores_tl = _gather_feat(scores_tl, inds).float()
    scores_br = _gather_feat(scores_br, inds).float()

    xs_ct = xs_ct[:, 0, :]
    ys_ct = ys_ct[:, 0, :]

    center = torch.stack([xs_ct, ys_ct, clses_ct.float(), scores_ct], dim=-1)
    detections = torch.cat([bboxes, scores, scores_tl, scores_br, classes], dim=2)

    return detections, center

    
def _rescale_dets(detections, centers, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]

    tx_inds = xs[:, :, 0] <= -5
    bx_inds = xs[:, :, 1] >= sizes[0, 1] + 5
    ty_inds = ys[:, :, 0] <= -5
    by_inds = ys[:, :, 1] >= sizes[0, 0] + 5

    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

    detections[:, tx_inds[0, :], 4] = -1
    detections[:, bx_inds[0, :], 4] = -1
    detections[:, ty_inds[0, :], 4] = -1
    detections[:, by_inds[0, :], 4] = -1

    centers[..., [0]] /= ratios[:, 1][:, None, None]
    centers[..., [1]] /= ratios[:, 0][:, None, None]
    centers[..., [0]] -= borders[:, 2][:, None, None]
    centers[..., [1]] -= borders[:, 0][:, None, None]
    np.clip(centers[..., [0]], 0, sizes[:, 1][:, None, None], out=centers[..., [0]])
    np.clip(centers[..., [1]], 0, sizes[:, 0][:, None, None], out=centers[..., [1]])
    
def center_match(detections, centers):
    valid_ind = detections[:, 4] > -1
    valid_detections = detections[valid_ind]

    box_width = valid_detections[:, 2] - valid_detections[:, 0]
    box_height = valid_detections[:, 3] - valid_detections[:, 1]

    small_box_ind = (box_width * box_height <= 22500)
    large_box_ind = (box_width * box_height > 22500)

    small_detections = valid_detections[small_box_ind]
    large_detections = valid_detections[large_box_ind]

    small_left_x = ((2 * small_detections[:, 0] + small_detections[:, 2]) / 3)[None, :]
    small_right_x = ((small_detections[:, 0] + 2 * small_detections[:, 2]) / 3)[None, :]
    small_top_y = ((2 * small_detections[:, 1] + small_detections[:, 3]) / 3)[None, :]
    small_bottom_y = ((small_detections[:, 1] + 2 * small_detections[:, 3]) / 3)[None, :]

    small_temp_score = small_detections[:, 4].copy()
    # small_temp_score = copy.copy(small_detections[:, 4])
    small_detections[:, 4] = -1

    center_x = centers[:, 0][:, None]
    center_y = centers[:, 1][:, None]

    ind_lx = (center_x - small_left_x) > 0
    ind_rx = (center_x - small_right_x) < 0
    ind_ty = (center_y - small_top_y) > 0
    ind_by = (center_y - small_bottom_y) < 0
    ind_cls = centers[:, 2][:, None] == small_detections[:, -1][None, :]
    ind_small_new_score = np.max(((ind_lx + 0) &
                                    (ind_rx + 0) &
                                    (ind_ty + 0) &
                                    (ind_by + 0) &
                                    (ind_cls + 0)), axis=0) == 1
    index_small_new_score = np.argmax(((ind_lx + 0) &
                                        (ind_rx + 0) &
                                        (ind_ty + 0) &
                                        (ind_by + 0) &
                                        (ind_cls + 0))[:, ind_small_new_score], axis=0)
    small_detections[:, 4][ind_small_new_score] = \
        (small_temp_score[ind_small_new_score] * 2 + centers[index_small_new_score, 3]) / 3

    large_left_x = ((3 * large_detections[:, 0] + 2 * large_detections[:, 2]) / 5)[None, :]
    large_right_x = ((2 * large_detections[:, 0] + 3 * large_detections[:, 2]) / 5)[None, :]
    large_top_y = ((3 * large_detections[:, 1] + 2 * large_detections[:, 3]) / 5)[None, :]
    large_bottom_y = ((2 * large_detections[:, 1] + 3 * large_detections[:, 3]) / 5)[None, :]

    large_temp_score = large_detections[:, 4].copy()
    # large_temp_score = copy.copy(large_detections[:, 4])
    large_detections[:, 4] = -1

    center_x = centers[:, 0][:, None]
    center_y = centers[:, 1][:, None]

    ind_lx = (center_x - large_left_x) > 0
    ind_rx = (center_x - large_right_x) < 0
    ind_ty = (center_y - large_top_y) > 0
    ind_by = (center_y - large_bottom_y) < 0
    ind_cls = centers[:, 2][:, None] == large_detections[:, -1][None, :]
    ind_large_new_score = np.max(((ind_lx + 0) &
                                    (ind_rx + 0) &
                                    (ind_ty + 0) &
                                    (ind_by + 0) &
                                    (ind_cls + 0)), axis=0) == 1
    index_l_new_score = np.argmax(((ind_lx + 0) &
                                    (ind_rx + 0) &
                                    (ind_ty + 0) &
                                    (ind_by + 0) &
                                    (ind_cls + 0))[:, ind_large_new_score], axis=0)
    large_detections[:, 4][ind_large_new_score] = \
        (large_temp_score[ind_large_new_score] * 2 + centers[index_l_new_score, 3]) / 3

    detections = np.concatenate([large_detections, small_detections], axis=0)
    detections = detections[np.argsort(-detections[:, 4])]
    clses = detections[..., -1]
    return detections, clses