from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np

from torch import float32
def process_pil(image):
    image = image.resize((224,224))
    image = np.array(image)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 1.0)
    image = cv2.addWeighted(image, 3, gaussian_3, -2.5, 0)
    image = to_pil_image(image)
    return image

def process_tensor(X):
    X = X.to(float32)
    mean, std = X.mean(), X.std()
    return (X - mean) / std