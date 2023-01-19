import torch
from torch.nn import functional as F
from typing import Tuple, Optional
import numpy as np


def _nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Perform a non-maximum suppression (NMS) operation
    on a 2D tensor "heat" using a square kernel of size "kernel".

    Args:
        heat (torch.Tensor): 2D tensor
        kernel (int, optional): kernel size. Defaults to 3.

    Returns:
        torch.Tensor: Result of the NMS operation
    """
    
    padding = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=padding)
    
    return heat * (hmax == heat).float()

def _topk(score_map: torch.Tensor, K: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the top K scores and their corresponding indices, classes, y-coordinates,
    and x-coordinates from a 4D tensor "score_map".
    
    Args:
        score_map (torch.Tensor): A 4D tensor of shape (batch, category, height, width)
                                  representing the scores for each class at each location.
        K (int): The number of top scores to extract from the score map. Default value is 20.
    
    Returns:
        topk_scores (torch.Tensor) : A tensor of shape (batch, K) representing the top K scores.
        topk_inds (torch.Tensor) : A tensor of shape (batch, K) representing the indices of top K scores in the original score map.
        topk_classes (torch.Tensor) : A tensor of shape (batch, K) representing the class of top K scores.
        topk_ys (torch.Tensor) : A tensor of shape (batch, K) representing the y-coordinates of top K scores.
        topk_xs (torch.Tensor) : A tensor of shape (batch, K) representing the x-coordinates of top K scores.
    """

    batch, _, height, width = score_map.size()

    topk_scores, topk_inds = torch.topk(score_map.view(batch, -1), K)

    topk_classes = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    return topk_scores, topk_inds, topk_classes, topk_ys, topk_xs

    
def _gather_feat(feat: torch.Tensor, ind: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gather and extract a specific subset of feature values from a 3D
    tensor "feat" based on the indices in a 2D tensor "ind", and optionally
    applies a mask to the extracted features.
    
    Args:
        feat (torch.Tensor): A 3D tensor representing the feature map.
        ind (torch.Tensor): A 2D tensor representing the indices of the elements in "feat" to be extracted.
        mask (Optional[torch.Tensor]): A 2D tensor representing the mask to be applied on the extracted features. Default value is None.
    
    Returns:
        torch.Tensor: A 2D tensor representing the gathered and extracted features with the mask applied.
    """

    dim = feat.size(2)
    feat = feat.gather(1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim))

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask].view(-1, dim)
    return feat

def _tranpose_and_gather_feature(feature: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """
    Perform a transpose operation on a 4D tensor "feature" followed
    by gathering specific subset of features based on the indices in a 2D tensor "ind".
    
    Args:
        feature (torch.Tensor): A 4D tensor representing the feature map.
        ind (torch.Tensor): A 2D tensor representing the indices of the elements in "feature" to be extracted.
    
    Returns:
        torch.Tensor: A 2D tensor representing the gathered and extracted features after applying transpose operation on the feature map.
    """
    
    feature = feature.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] => [B, H, W, C]
    feature = feature.view(feature.size(0), -1, feature.size(3))  # [B, H, W, C] => [B, H x W, C]
    ind = ind[:, :, None].expand(ind.shape[0], ind.shape[1], feature.shape[-1])  # [B, num_obj] => [B, num_obj, C]
    
    return feature.gather(1, ind)  # [B, H x W, C] => [B, num_obj, C]

    
def decode(hmap_tl, hmap_br, hmap_ct,
           embd_tl, embd_br,
           regs_tl, regs_br, regs_ct,
           K, kernel, ae_threshold, num_dets=1000):
    batch, _, _, width = hmap_tl.shape

    hmap_tl = torch.sigmoid(hmap_tl)
    hmap_br = torch.sigmoid(hmap_br)
    hmap_ct = torch.sigmoid(hmap_ct)

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

def bbox_center(detection: np.array, n: int = 3) -> np.array:
    """Calculate the bounding box for the center keypoint

    Args:
        detection (np.array): Detection array containing the top left and bottom right coordinates
        n (int, optional): Odd number 3 or 5. Determines the scale of the central region. Defaults to 3.

    Returns:
        np.array: Array containing the top left and bottom right coordinates of the central region
    """
    
    array = np.zeros((detection.shape[0], detection.shape[1], 4), dtype=np.float32)
    array[:, :, 0] = ((n + 1) * detection[:, :, 0] + (n - 1) * detection[:, :, 2]) / (2 * n)
    array[:, :, 1] = ((n + 1) * detection[:, :, 1] + (n - 1) * detection[:, :, 3]) / (2 * n)
    array[:, :, 2] = ((n - 1) * detection[:, :, 0] + (n + 1) * detection[:, :, 2]) / (2 * n)
    array[:, :, 3] = ((n - 1) * detection[:, :, 1] + (n + 1) * detection[:, :, 3]) / (2 * n)
    
    return array


def filter_detections(detections: torch.Tensor, centers: torch.Tensor, n: int = 3) -> np.array:
    """Filter and combine bounding box detections and their corresponding
    centers by checking if the centers fall within the center of the bounding boxes.

    Args:
        detections (torch.Tensor): Tensor of shape (num_detections, 8) containing: (x1, y1, x2, y2, score, scoretl, scorebr, class)
        centers (torch.Tensor): Tensor of shape (num_centers, 4) containing: (x, y, score, class)
        n (int, optional): Odd number 3 or 5. Determines the scale of the central region. Defaults to 3.

    Returns:
        np.array: A 2D array representing the filtered detections with centers.
        Contains: (tlx, tly, brx, bry, cx, cy, score, class)
    """
    
    detections_centers = []
    
    for classe in range(10):
        dets = detections[np.where((detections[:, -1] == classe) & (detections[:, 4] >0))]
        cets = centers[np.where(centers[:, 2] == classe)]
        
        if (len(dets) == 0) or (len(cets) == 0):
            continue
        
        dets = dets[None, :, :]
        cets = cets[None, :, :]

        ct_bb = bbox_center(dets, n)
        
        for i in range(cets.shape[1]):
            for j in range(ct_bb.shape[1]):
                if (cets[0, i, 0] >= ct_bb[0, j, 0]) and (cets[0, i, 0] <= ct_bb[0, j, 2]) and (cets[0, i, 1] >= ct_bb[0, j, 1]) and (cets[0, i, 1] <= ct_bb[0, j, 3]):
                    bbox = dets[:, j, :4][0]
                    cent = cets[:, i, :2][0]
                    score = dets[0, j, 4]
                    classe = dets[0, j, -1]
                    detections_centers.append(np.array([*bbox, *cent, score, classe]))
                    break
    
    detections_centers = np.array(detections_centers)
    detections_centers = detections_centers[detections_centers[:, 6].argsort()[::-1]]

    return detections_centers


def rescale_detection(detections: np.array,
                      ratios: np.array = np.array([[0.25, 0.25]]),
                      borders: np.array = np.array([[0., 300., 0., 300.]]),
                      sizes: np.array = np.array([[300, 300]])) -> None:
    """Rescale in place the detections to the original image size

    Args:
        detections (np.array): Array of detections of shape (num_detections, 8) containing: (tlx, tly, brx, bry, cx, cy, score, class)
        ratios (np.array, optional): Ratios used to rescale the image
        borders (np.array, optional): Borders used to rescale the image
        sizes (np.array, optional): Size of the original image
    """
    
    xs = detections[..., 0:6:2]
    ys = detections[..., 1:6:2]
    
    xs /= ratios[:, 1]
    ys /= ratios[:, 0]
    xs -= borders[:, 2]
    ys -= borders[:, 0]

    tx_inds = xs[:, 0] <= -5
    bx_inds = xs[:, 1] >= sizes[0, 1] + 5
    ty_inds = ys[:, 0] <= -5
    by_inds = ys[:, 1] >= sizes[0, 0] + 5

    np.clip(xs, 0, sizes[:, 1], out=xs)
    np.clip(ys, 0, sizes[:, 0], out=ys)
    
    detections[tx_inds, 6] = -1
    detections[bx_inds, 6] = -1
    detections[ty_inds, 6] = -1
    detections[by_inds, 6] = -1
    
    detections[:, 0:6] = np.floor(detections[:, 0:6])

    _, idx = np.unique(detections[:, :4], axis=0, return_index=True)
    
    return detections[idx]