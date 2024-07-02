import os
import shutil
from PIL import Image

def rotate(path, degrees=[90, 180, 270], image_ext='jpg'):
    classes = [c for c in os.listdir(path) if os.path.isdir(os.path.join(path,c))]
    path_class = [os.path.join(path, c) for c in classes]
    path_class

    for pc in path_class:
        path_image = [os.path.join(pc, image) for image in os.listdir(pc) if image.split('.')[-1] == f'{image_ext}']
        index = len(path_image)
        for pi in path_image:
            im = Image.open(pi)
            for degree in degrees:
                im = im.rotate(degree)
                im.save(os.path.join(pc, f"{index:04}.{image_ext}"))
                index += 1


def copy(src_path, target_path):
    index = len(os.listdir(target_path)) - 1
    for image in os.listdir(src_path):
        ext = image.split('.')[-1]
        src = os.path.join(src_path, image)
        target = os.path.join(target_path, f"{index:04}.{ext}")
        shutil.copy(src, target)
        index += 1
