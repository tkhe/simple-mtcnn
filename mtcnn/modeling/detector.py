import cv2
import numpy as np
import torch
import torch.nn.functional as F


from mtcnn.config import cfg
from mtcnn.utils.boxes import bbox_transform_inv
from mtcnn.utils.cython_nms import nms
from mtcnn.utils.image import cv_image_to_tensor


class Detector(object):
    def __init__(self,
                 pnet=None,
                 rnet=None,
                 min_size=24,
                 stride=2,
                 scale_factor=0.79):
        super(Detector, self).__init__()
        self.pnet = pnet
        self.rnet = rnet
        self.min_size = min_size
        self.stride = stride
        self.scale_factor = scale_factor

    def generate_bbox(self, cls, reg, scale, threshold):
        stride = self.stride
        cell_size = 12
        cls = torch.squeeze(cls)
        reg = torch.squeeze(reg)
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
        y2 = torch.round((stride * mask[:, 1:] + cell_size) / scale) - 1

        boxes = torch.cat((x1, y1, x2, y2, score, reg), 1)
        return boxes

    def detect_pnet(self, im):
        net_size = 12

        current_scale = float(net_size) / self.min_size

        im_resized = cv2.resize(
            im,
            (0, 0),
            fx=current_scale,
            fy=current_scale,
            interpolation=cv2.INTER_LINEAR
        )
        current_height, current_width, _ = im_resized.shape
        all_boxes = list()
        tensor = cv_image_to_tensor(im_resized)
        input = torch.unsqueeze(tensor, 0)
        while min(current_height, current_width) > net_size:
            if self.pnet.is_cuda:
                input = input.cuda()
            cls, reg = self.pnet(input)
            cls = F.softmax(cls, dim=1)
            boxes = self.generate_bbox(
                cls[:, 1, :, :],
                reg,
                current_scale,
                cfg.TEST.PNET.THRESHOLD
            )
            current_scale *= self.scale_factor
            im_resized = cv2.resize(
                im,
                (0, 0),
                fx=current_scale,
                fy=current_scale,
                interpolation=cv2.INTER_LINEAR
            )
            tensor = cv_image_to_tensor(im_resized)
            input = torch.unsqueeze(tensor, 0)
            current_height, current_width, _ = im_resized.shape
            if boxes.size(0) == 0:
                continue
            boxes = boxes.cpu().numpy()
            keep = nms(boxes[:, :5].astype(np.float32), cfg.TEST.PNET.NMS[0])
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None
        all_boxes = np.vstack(all_boxes)
        boxes = all_boxes[:4]
        deltas = all_boxes[5:]
        boxes = bbox_transform_inv(boxes, deltas, type=cfg.MODEL.TRANSFORM)
        all_boxes[:, 0:4] = boxes
        keep = nms(all_boxes[:, 0:5].astype(np.float32), cfg.TEST.PNET.NMS[1])
        all_boxes = all_boxes[keep]
        return all_boxes

    def detect(self, im):
        boxes = None
        if self.pnet:
            boxes = self.detect_pnet(im)
        return boxes
