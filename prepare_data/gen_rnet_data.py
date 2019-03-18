import argparse
import os
import pickle
import pprint
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from mtcnn.config import cfg
from mtcnn.datasets.imdb import get_imdb
from mtcnn.engine.inference import inference
from mtcnn.modeling.detector import Detector
from mtcnn.modeling.model_builder import build_model
from mtcnn.utils.box_coder import build_box_coder
from mtcnn.utils.cython_bbox import bbox_overlaps
from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)


def gen_rnet_data(predictions):
    logger.info("Start generating")

    with open(os.path.join(cfg.DATA_DIR, 'annotations', 'train.txt'), 'r') as f:
        annotations = [line.strip() for line in f.readlines()]

    pos_rois_index = 0
    part_rois_index = 0
    neg_rois_index = 0
    im_index = 0

    box_coder = build_box_coder(cfg.MODEL.TRANSFORM, cfg.MODEL.BBOX_REG_WEIGHTS)

    rois = defaultdict(list)
    for annotation in annotations:
        annotation = annotation.split(' ')
        im_path = os.path.join(cfg.DATA_DIR, 'train', annotation[0])
        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im = cv2.imread(im_path)

        pred_rois = predictions[annotation[0]]
        if pred_rois is None:
            continue
        for roi in pred_rois:
            roi = roi[:4]
            iou = bbox_overlaps(roi.reshape((-1, 4)), boxes)
            max_index = np.argmax(iou)
            target = box_coder.encode(roi.reshape((-1, 4)), boxes[max_index].reshape(-1, 4))
            if np.max(iou) > 0.65:
                rois[im_path].append(
                    {
                        'bbox': [roi[i] for i in range(4)],
                        'target': target,
                        'label': 1
                    }
                )
                pos_rois_index += 1
            elif np.max(iou) > 0.4:
                if part_rois_index < 3 * pos_rois_index:
                    rois[im_path].append(
                        {
                            'bbox': [roi[i] for i in range(4)],
                            'target': target,
                            'label': -1
                        }
                    )
                    part_rois_index += 1
            elif np.max(iou) < 0.3:
                if neg_rois_index < 3 * pos_rois_index:
                    rois[im_path].append(
                        {
                            'bbox': [roi[i] for i in range(4)],
                            'target': np.zeros((1, 4), dtype=np.float32),
                            'label': 0
                        }
                    )
                    neg_rois_index += 1
        im_index += 1
        print(
            '{} images done, pos: {} part: {} neg {}'
            .format(im_index, pos_rois_index, part_rois_index, neg_rois_index)
        )
    cache_dir = os.path.join(cfg.DATA_DIR, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, 'anno_rnet.pkl'), 'wb') as f:
        pickle.dump(rois, f)

    logger.info('Finish')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for generating P-Net training data',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info('Called with args:')
    logger.info(pprint.pformat(args))
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    logger.info('Using configs:')
    logger.info(pprint.pformat(cfg))

    device = torch.device(cfg.MODEL.DEVICE)
    pnet = build_model('pnet')
    pnet.to(device)
    params = os.path.join(cfg.OUTPUT_DIR, 'pnet', 'model_final.pth')
    pnet.load_state_dict(torch.load(params))
    pnet.eval()

    detector = Detector(pnet=pnet)
    data_dir = os.path.join(cfg.DATA_DIR, 'train')
    imdb = get_imdb(data_dir, 'train')

    # predictions = inference(detector, imdb)
    # with open(os.path.join(cfg.DATA_DIR, 'cache', 'dets_pnet.pkl'), 'wb') as f:
        # pickle.dump(predictions, f)
    with open(os.path.join(cfg.DATA_DIR, 'cache', 'dets_pnet.pkl'), 'rb') as f:
        predictions = pickle.load(f)
    gen_rnet_data(predictions)


if __name__ == "__main__":
    main()
