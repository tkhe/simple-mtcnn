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

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.TYPE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss = Loss()
    loss_ratio = cfg.TRAIN.LOSS_RATIO
    max_iters =len(data_loader)

    for iteration, (im, label, target) in enumerate(data_loader):
        iteration += 1

        pos = label[label > 0].sum()
        part = (-label[label < 0]).sum()
        neg = cfg.TRAIN.BATCH_SIZE - pos - part

        scheduler.step()

        im = im.to(device)
        label = label.to(device)
        target = target.to(device)

        cls, reg = model(im)
        cls_loss = loss.cls_loss(cls, label)
        reg_loss = loss.reg_loss(reg, target, label)
        losses = loss_ratio[0] * cls_loss + loss_ratio[1] * reg_loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iteration % cfg.TRAIN.DISPLAY == 0 or iteration == max_iters:
            cls = torch.squeeze(cls)
            label = torch.squeeze(label)
            pred = torch.argmax(cls, dim=1)

            pred = pred[label >= 0]
            label = label[label >= 0]
            acc = (pred == label).float().mean()
            training_state = {
                'iters': iteration,
                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                'cls_loss': cls_loss.cpu().tolist(),
                'reg_loss': reg_loss.cpu().tolist(),
                'total_loss': losses.cpu().tolist(),
                'pos': pos.tolist(),
                'part': part.tolist(),
                'neg': neg.tolist(),
                'acc': acc.cpu().tolist()
            }
            log_json_state(training_state)
        
        if iteration % cfg.TRAIN.SNAPSHOT == 0:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, 'model_iter{}.pth'.format(iteration))
            )
    logger.info('Training finished. Saving model...')
    final_path = os.path.join(output_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    logger.info('Model saved in {}'.format(final_path))
