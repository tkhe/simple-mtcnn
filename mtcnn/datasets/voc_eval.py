import os
from collections import defaultdict

import numpy as np

from mtcnn.config import cfg


def do_voc_evaluation(dataset, predictions, iou=0.5, use_07_metric=True):
    assert dataset in ('train', 'test'), "Unknown dataset: {}".format(dataset)
    if dataset == 'train':
        gt_file = os.path.join(cfg.DATA_DIR, 'annotations', 'train.txt')
    else:
        gt_file = os.path.join(cfg.DATA_DIR, 'annotations', 'test.txt')

    with open(gt_file, 'r') as f:
        annotations = f.readlines()

    gt_rois = dict()
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes).reshape((-1, 4))
        gt_rois[im_path] = boxes
    
    pred_rois = list()
    for im_path, boxes in predictions.items():
        if boxes is not None:
            for box in boxes:
                pred_rois.append([im_path, box[4], box[0], box[1], box[2], box[3]])
        else:
            continue

    prec, rec, ap, fp = eval_detections_voc(
        pred_rois,
        gt_rois,
        iou=iou,
        use_07_metric=use_07_metric
    )
    return prec, rec, ap, fp

def eval_detections_voc(predictions, ground_truth, iou=0.5, use_07_metric=True):
    prec, rec, fp = calc_detection_voc_prec_rec(
        predictions, ground_truth, iou=iou)
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return prec, rec, ap, fp


def calc_detection_voc_prec_rec(predictions, ground_truth, iou=0.5):
    image_ids = [pred[0] for pred in predictions]
    confidence = np.array([float(pred[1]) for pred in predictions])
    BB = np.array([[float(x) for x in pred[2:]] for pred in predictions])

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[ind] for ind in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        BBGT = ground_truth[image_ids[d]]
        det = [False] * BBGT.shape[0]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf

        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] -
                                                                 BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou:
            if not det[jmax]:
                tp[d] = 1
                det[jmax] = True
            else:
                fp[d] = 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    n_pos = 0
    for image_id in ground_truth:
        n_pos += ground_truth[image_id].shape[0]
    rec = tp / float(n_pos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return prec, rec, fp


def calc_detection_voc_ap(prec, rec, use_07_metric):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
