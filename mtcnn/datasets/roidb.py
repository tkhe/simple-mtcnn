import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset

from mtcnn.config import cfg
from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)


class RoIDB(Dataset):
    def __init__(self, transforms=None):
        super(RoIDB, self).__init__()
        self.transforms = transforms
        self.rois = self._load_rois()

    def _load_rois(self):
        path = os.path.join(
            cfg.DATA_DIR,
            'cache',
            'anno_{}.pkl'.format(cfg.MODEL.TYPE)
        )
        assert os.path.exists(path), "{} doesn't exist".format(path)

        with open(path, 'rb') as f:
            rois_dict = pickle.load(f)

        rois = self._read_im(rois_dict)
        return rois

    def _read_im(self, rois_dict):
        rois = []
        num = 0
        logger.info('Loading rois:')
        for im_path, roi_list in rois_dict.items():
            im = Image.open(im_path).convert('RGB')
            for roi in roi_list:
                crop_im = im.crop(roi['bbox'])
                crop_im = crop_im.resize((cfg.MODEL.SIZE, cfg.MODEL.SIZE))
                rois.append(
                    {
                        'im': crop_im,
                        'target': roi['target'],
                        'label': roi['label']
                    }
                )
            im.close()
            num += 1
            if num % 100 == 0:
                logger.info('{} images loaded'.format(num))
        return rois
    
    def __getitem__(self, index):
        roi = self.rois[index]
        im = roi['im']
        label = torch.LongTensor([roi['label']])
        target = torch.FloatTensor([roi['target']])
        if self.transforms:
            return self.transforms(im), label, target
        else:
            return im, label, target

    def __len__(self):
        return len(self.rois)
    
def get_roidb(transforms=None):
    return RoIDB(transforms=transforms)
