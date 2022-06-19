import torch
import cv2
import numpy as np


def is_correct_hit(bbox_annot, heatmap, orig_img_shape):
    h_orig, w_orig = orig_img_shape
    h, w = heatmap.shape

    if isinstance(heatmap, np.ndarray):
        if not (h == h_orig and w == w_orig):
            heatmap = cv2.resize(heatmap, (w_orig, h_orig))
        max_loc = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    else:
        if not (h == h_orig and w == w_orig):
            heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (h_orig, w_orig),
                                                      mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        index = torch.argmax(heatmap)
        y = index / w_orig  # (integer division in pytorch tensors is just `/` not `//`)
        x = index % w_orig
        max_loc = [y, x]
    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1, heatmap
    return 0, heatmap


def check_percent(bboxes):
    for bbox in bboxes:
        x_length = bbox[2]-bbox[0]
        y_length = bbox[3]-bbox[1]
        if x_length*y_length < .05:
            return False
    return True


def union(bbox):
    if len(bbox) == 0:
        return []
    if isinstance(bbox[0], type(0.0)) or isinstance(bbox[0], type(0)):
        bbox = [bbox]
    maxes = np.max(bbox, axis=0)
    mins = np.min(bbox, axis=0)
    return [[mins[0], mins[1], maxes[2], maxes[3]]]


def calc_correctness(annot, heatmap, orig_img_shape):

    bbox_annot = annot['bbox']

    hit_correctness, heatmap_resized = is_correct_hit(bbox_annot, heatmap, orig_img_shape)

    return hit_correctness
