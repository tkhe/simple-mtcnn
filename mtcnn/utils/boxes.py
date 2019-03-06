import numpy as np


def bbox_transform(ex_rois, gt_rois, type='rcnn'):
    assert type in ['rcnn', 'mtcnn'], f'Unknown transform type: {type}'
    if type == 'rcnn':
        return rcnn_transform(ex_rois, gt_rois)
    else:
        return mtcnn_transform(ex_rois, gt_rois)


def bbox_transform_inv(boxes, deltas, type='rcnn'):
    assert type in ['rcnn', 'mtcnn'], f'Unknown transform type: {type}'
    if type == 'rcnn':
        return rcnn_transform_inv(boxes, deltas)
    else:
        return mtcnn_transform_inv(boxes, deltas)


def mtcnn_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0

    targets_x1 = (gt_rois[:, 0] - ex_rois[:, 0]) / ex_widths
    targets_y1 = (gt_rois[:, 1] - ex_rois[:, 1]) / ex_widths
    targets_x2 = (gt_rois[:, 2] - ex_rois[:, 2]) / ex_heights
    targets_y2 = (gt_rois[:, 3] - ex_rois[:, 3]) / ex_heights

    targets = np.vstack((
        targets_x1,
        targets_y1,
        targets_x2,
        targets_y2
    )).transpose()
    return targets


def mtcnn_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    dx1 = deltas[:, 0::4]
    dy1 = deltas[:, 1::4]
    dx2 = deltas[:, 2::4]
    dy2 = deltas[:, 3::4]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)

    pred_boxes[:, 0::4] = dx1 * widths[:, np.newaxis] + boxes[:, 0::4]
    pred_boxes[:, 1::4] = dy1 * widths[:, np.newaxis] + boxes[:, 1::4]
    pred_boxes[:, 2::4] = dx2 * heights[:, np.newaxis] + boxes[:, 2::4]
    pred_boxes[:, 3::4] = dy2 * heights[:, np.newaxis] + boxes[:, 3::4]

    return pred_boxes


def rcnn_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((
        targets_dx,
        targets_dy,
        targets_dw,
        targets_dh
    )).transpose()
    return targets


def rcnn_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1.0
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1.0

    return pred_boxes
