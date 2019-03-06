import os

import torch

from mtcnn.config import cfg
from mtcnn.modeling.loss import Loss
from mtcnn.utils.logger import log_json_state
from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)


def do_train(model, data_loader, optimizer, scheduler, device):
    logger.info('Start training')

    model.train()

    loss = Loss()
    loss_ratio = cfg.TRAIN.LOSS_RATIO
    max_iters = len(data_loader)

    for iteration, (im, label, target) in enumerate(data_loader):
        iteration += 1

        scheduler.step()

        im = im.to(device)
        label = label.to(device)
        target = target.to(device)

        cls, reg = model(im)
        cls_loss, reg_loss = loss(cls, reg, label, target)
        losses = loss_ratio[0] * cls_loss + loss_ratio[1] * reg_loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iteration % cfg.TRAIN.DISPLAY == 0 or iteration == max_iters:
            training_state = {
                'iters': iteration,
                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                'cls_loss': cls_loss.cpu().tolist(),
                'reg_loss': reg_loss.cpu().tolist(),
                'total_loss': losses.cpu().tolist()
            }
            log_json_state(training_state)

        if iteration % cfg.TRAIN.SNAPSHOT == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    get_output_dir(cfg.MODEL.TYPE),
                    'model_iter{}.pth'.format(iteration)
                )
            )
    logger.info('Training finished. Saving model...')
    final_path = os.path.join(get_output_dir(cfg.MODEL.TYPE), 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    logger.info('Model saved in {}'.format(final_path))


def get_output_dir(net_type):
    dirname = os.path.join(cfg.ROOT_DIR, 'output', net_type)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname
