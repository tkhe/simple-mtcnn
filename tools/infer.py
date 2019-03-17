import argparse
import os
import pprint
import sys

import torch
import cv2

from mtcnn.config import cfg
from mtcnn.modeling.detector import Detector
from mtcnn.modeling.model_builder import build_model
from mtcnn.utils.logger import setup_logging
from mtcnn.utils.visualize import visualize_boxes

logger = setup_logging(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--im',
        dest='im_file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--pnet',
        dest='pnet',
        action='store_true'
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
    if args.pnet:
        pnet = build_model('pnet')
        pnet.to(device)
        params = os.path.join(cfg.OUTPUT_DIR, 'pnet', 'model_final.pth')
        pnet.load_state_dict(torch.load(params))

    detector = Detector(pnet)

    im = cv2.imread(args.im_file)
    boxes = detector.detect(im)
    im = visualize_boxes(im, boxes)
    cv2.imshow('demo', im)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
