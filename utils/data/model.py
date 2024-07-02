from pathlib import Path
from PIL import Image
from functools import partial

import os


class VisionDataset():
    def __init__(
            self, 
            data_path:str|Path, 
            split:str, 
            class_map:dict, 
            transform:callable=None, 
            target_transform:callable= None,
        ) -> dict:

        self.transform = transform
        self.target_transform = target_transform
        self.class_map = class_map
        allowed_ext = ['jpg', 'png', 'jpeg']
        self.path = os.path.join(data_path, split)
        self.images = []
        self.classes = [f for f in os.listdir(self.path) if not os.path.isfile(os.path.join(self.path, f))]
        for cls in self.classes:
            cls_images = [x for x in os.listdir(os.path.join(self.path, cls)) if x.split('.')[-1] in allowed_ext]
            self.images += list(map(partial(os.path.join, cls), cls_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file_name = self.images[index]
        if isinstance(index, slice):
            labels = [
                self.target_transform(self.class_map[Path(fn).parts[0]]) if self.target_transform 
                else self.class_map[Path(fn).parts[0]] 
                for fn in file_name
            ]
            images = [
                self.transform(Image.open(os.path.join(self.path, fn))) if self.transform
                else Image.open(os.path.join(self.path, fn))
                for fn in file_name
            ]
        else:
            labels = self.target_transform(self.class_map[Path(file_name).parts[0]]) if self.target_transform else self.class_map[Path(file_name).parts[0]]
            images = self.transform(Image.open(os.path.join(self.path, file_name))) if self.transform else Image.open(os.path.join(self.path, file_name))
        return {'image': images, 'label': labels}