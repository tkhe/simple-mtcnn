import os

import cv2
from torch.utils.data import Dataset


class imdb(Dataset):
    def __init__(self, data_dir, mode):
        self.root = data_dir
        self.mode = mode
        self.ids = os.listdir(self.root)
        self.im_path = os.path.join(self.root, '%s')

    def __getitem__(self, index):
        im_id = self.ids[index]
        im = cv2.imread(self.im_path % im_id)
        return im, im_id


def get_imdb(data_dir, mode='test'):
    return imdb(data_dir, mode=mode)
