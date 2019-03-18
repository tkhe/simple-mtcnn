import argparse
import os
import pprint
import sys

import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mtcnn.config import cfg
from mtcnn.datasets.imdb import get_imdb
from mtcnn.datasets.voc_eval import do_voc_evaluation
from mtcnn.modeling.detector import Detector
from mtcnn.engine.inference import inference
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
        '--ds',
        dest='dataset',
        default='test',
        type=str
    )
    parser.add_argument(
        '--pnet',
        dest='pnet',
        action='store_true'
    )
    parser.add_argument(
        '--rnet',
        dest='rnet',
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

    pnet = None
    rnet = None
    if args.rnet:
        args.pnet = True
        rnet = build_model('rnet')
        rnet.to(device)
        params = os.path.join(cfg.OUTPUT_DIR, 'rnet', 'model_final.pth')
        rnet.load_state_dict(torch.load(params))
        rnet.eval()
    if args.pnet:
        pnet = build_model('pnet')
        pnet.to(device)
        params = os.path.join(cfg.OUTPUT_DIR, 'pnet', 'model_final.pth')
        pnet.load_state_dict(torch.load(params))
        pnet.eval()

    detector = Detector(pnet=pnet, rnet=rnet)
    data_dir = os.path.join(cfg.DATA_DIR, args.dataset)
    imdb = get_imdb(data_dir, args.dataset)
    predictions = inference(detector, imdb)
    prec, rec, ap, fp = do_voc_evaluation(args.dataset, predictions)
    logger.info('ap: {}'.format(ap))



if __name__ == '__main__':
    main()
