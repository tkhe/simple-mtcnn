import argparse
import pprint
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mtcnn.config import cfg
from mtcnn.datasets.batch_sampler import build_batch_sampler
from mtcnn.datasets.roidb import get_roidb
from mtcnn.engine.trainer import do_train
from mtcnn.modeling.model_builder import build_model
from mtcnn.solver.lr_scheduler import make_optimizer
from mtcnn.solver.lr_scheduler import make_scheduler
from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)


def train():
    model = build_model(cfg.MODEL.TYPE)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    transform = transforms.ToTensor()

    roidb = get_roidb(transforms=transform)
    batch_sampler = build_batch_sampler(
        roidb,
        cfg.TRAIN.BATCH_SIZE,
        shuffle=True
    )
    data_loader = DataLoader(roidb, batch_sampler=batch_sampler, num_workers=1)

    do_train(model, data_loader, optimizer, scheduler, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
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

    train()


if __name__ == '__main__':
    main()
