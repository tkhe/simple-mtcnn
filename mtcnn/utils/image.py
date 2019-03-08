import cv2
import torchvision.transforms as transforms
from PIL import Image


def cv_image_to_tensor(im):
    pil_img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    transform = transforms.ToTensor()
    tensor = transform(pil_img)
    return tensor
