import argparse
import os
import pickle
import pprint
import sys
from collections import defaultdict

import cv2
import numpy as np

from mtcnn.config import cfg
from mtcnn.utils.box_coder import build_box_coder
from mtcnn.utils.cython_bbox import bbox_overlaps
from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)


def gen_pnet_data():
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

        height, width, _ = im.shape

        neg_num = 0
        while neg_num < 50:
            size = np.random.randint(12, min(height, width) / 2)
            x = np.random.randint(0, width - size)
            y = np.random.randint(0, height - size)
            bbox = np.array([[x, y, x + size, y + size]], dtype=np.float32)
            iou = bbox_overlaps(bbox, boxes)
            if np.max(iou) < 0.3:
                rois[im_path].append(
                    {
                        'bbox': [x, y, x + size, y + size],
                        'target': np.zeros((1, 4), dtype=np.float32),
                        'label': 0
                    }
                )
                neg_num += 1
                neg_rois_index += 1

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if w < 12 or h < 12:
                continue

            for _ in range(12):
                size = np.random.randint(
                    int(min(w, h) * 0.8),
                    int(max(w, h) * 1.25) + 1
                )
                delta_x = np.random.randint(-int(w * 0.2), int(w * 0.2))
                delta_y = np.random.randint(-int(h * 0.2), int(h * 0.2))

                nx1 = int(max(0, x1 + w / 2 + delta_x - size / 2))
                ny1 = int(max(0, y1 + h / 2 + delta_y - size / 2))
                nx2 = int(min(nx1 + size, width - 1))
                ny2 = int(min(ny1 + size, height - 1))

                bbox = np.array([[nx1, ny1, nx2, ny2]], dtype=np.float32)
                iou = bbox_overlaps(bbox, boxes)
                max_index = np.argmax(iou)
                target = box_coder.encode(bbox, boxes[max_index].reshape(-1, 4))
                if np.max(iou) > 0.65:
                    rois[im_path].append(
                        {
                            'bbox': [nx1, ny1, nx2, ny2],
                            'target': target,
                            'label': 1
                        }
                    )
                    pos_rois_index += 1
                elif np.max(iou) > 0.4:
                    rois[im_path].append(
                        {
                            'bbox': [nx1, ny1, nx2, ny2],
                            'target': target,
                            'label': -1
                        }
                    )
                    part_rois_index += 1
        im_index += 1
        print(
            '{} images done, pos: {} part: {} neg {}'
            .format(im_index, pos_rois_index, part_rois_index, neg_rois_index)
        )
    cache_dir = os.path.join(cfg.DATA_DIR, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, 'anno_pnet.pkl'), 'wb') as f:
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

    gen_pnet_data()


if __name__ == "__main__":
    main()
