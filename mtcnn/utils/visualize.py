import cv2
import numpy as np


def visualize_boxes(im, boxes):
    for box in boxes:
        score = float(box[4])
        box = list(map(int, box[:4]))
        box = list(map(lambda x: 0 if x < 0 else x, box))
        cv2.rectangle(
            im,
            (box[0], box[1]),
            (box[2], box[3]),
            (18, 127, 15),
            thickness=1
        )
        im = visualize_text(im, (box[0], box[1]),
                            class_str='{:.2f}'.format(score))
    return im


def visualize_text(im, pos, font_scale=0.35, class_str='airplane'):
    im = im.astype(np.uint8)
    x, y = int(pos[0]), int(pos[1])
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    back_tl = x, y - int(1.3 * txt_h)
    back_br = x + txt_w, y
    cv2.rectangle(im, back_tl, back_br, (18, 127, 15), -1)
    txt_tl = x, y - int(0.3 * txt_h)
    cv2.putText(
        im,
        txt,
        txt_tl,
        font,
        font_scale,
        (218, 227, 218),
        lineType=cv2.LINE_AA
    )
    return im
