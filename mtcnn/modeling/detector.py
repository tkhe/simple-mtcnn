import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mtcnn.config import cfg
from mtcnn.utils.box_coder import build_box_coder
from mtcnn.utils.cython_nms import nms
from mtcnn.utils.image import cv_image_to_tensor
from mtcnn.utils.image import clip_boxes_to_image
from mtcnn.utils.image import filter_small_boxes

box_coder = build_box_coder(cfg.MODEL.TRANSFORM, cfg.MODEL.BBOX_REG_WEIGHTS)


class Detector(object):
    def __init__(self, pnet, stride=2, min_size=12, scale_factor=0.707):
        self.pnet = pnet
        self.stride = stride
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.device = torch.device(cfg.MODEL.DEVICE)

    def generate_boxes(self, cls, reg, scale, threshold):
        stride = self.stride
        cell_size = 12
        cls = torch.squeeze(cls, 0)
        reg = torch.squeeze(reg, 0)
        reg = reg.permute(1, 2, 0)
        mask = torch.nonzero(cls > threshold)
        if mask.size(0) == 0:
            return torch.tensor([])
        reg = reg[mask[:, 0], mask[:, 1], :]
        score = cls[cls > threshold].view(-1, 1)
        mask = mask.float()
        x1 = torch.round(stride * mask[:, 1:] / scale)
        y1 = torch.round(stride * mask[:, :1] / scale)
        x2 = torch.round((stride * mask[:, 1:] + cell_size) / scale) - 1
        y2 = torch.round((stride * mask[:, :1] + cell_size) / scale) - 1

        boxes = torch.cat((x1, y1, x2, y2, score, reg), 1)
        return boxes

    def detect_pnet(self, im):
        h, w, _ = im.shape
        short_side = min(h, w)
        scales = list()
        net_size = 12
        scale = 12 / self.min_size
        while scale * short_side > net_size:
            scales.append(scale)
            scale *= self.scale_factor

        all_boxes = list()
        for scale in scales:
            img = cv2.resize(im, (0, 0), fx=scale, fy=scale)
            tensor = cv_image_to_tensor(img)
            input = torch.unsqueeze(tensor, 0)
            input.to(self.device)
            cls, reg = self.pnet(input)
            cls = F.softmax(cls, dim=1)
            boxes = self.generate_boxes(
                cls[:, 1, :, :],
                reg,
                scale,
                cfg.TEST.PNET.THRESHOLD
            )
            boxes = boxes.cpu().detach().numpy()
            if boxes.size == 0:
                continue
            deltas = boxes[:, 5:]
            boxes[:, 0:4] = box_coder.decode(deltas, boxes[:, 0:4])
            boxes = clip_boxes_to_image(boxes, w, h)
            boxes = filter_small_boxes(boxes)
            keep = nms(boxes[:, :5].astype(np.float32), cfg.TEST.PNET.NMS[0])
            boxes = boxes[keep]
            boxes = boxes[boxes[:, 4].argsort()[::-1]][:300, :]
            all_boxes.append(boxes[:, :5])
        if len(all_boxes) == 0:
            return None
        all_boxes = np.vstack(all_boxes)
        keep = nms(all_boxes.astype(np.float32), cfg.TEST.PNET.NMS[1])
        all_boxes = all_boxes[keep]
        all_boxes = all_boxes[all_boxes[:, 4].argsort()[::-1]][:300, :]
        return all_boxes

    def detect(self, im):
        if self.pnet:
            boxes = self.detect_pnet(im)
        return boxes