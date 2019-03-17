import cv2
import numpy as np
from PIL import Image

import torchvision.transforms as transforms


def cv_image_to_tensor(im):
    pil_im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    transform = transforms.ToTensor()
    tensor = transform(pil_im)
    return tensor


def clip_boxes_to_image(boxes, width, height):
    boxes[:, [0, 2]] = np.minimum(width - 1, np.maximum(0, boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1, np.maximum(0, boxes[:, [1, 3]]))
    return boxes


def filter_small_boxes(boxes, min_size=12):
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w > min_size) & (h > min_size))[0]
    boxes = boxes[keep]
    return boxes
