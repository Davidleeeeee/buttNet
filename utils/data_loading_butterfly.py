import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from img_expansion import resize16, patches
import matplotlib.pyplot as plt
from skimage.util import random_noise


class BasicDatasetBF(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if file.endswith('.bmp')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, is_mask, not_origen):
        w, h = pil_img.size
        pil_img = resize16(pil_img)
        # if (not is_mask) and not_origen:
        if not is_mask and not_origen:
            sigma = 0.155
            # pil_img = (random_noise(np.asarray(pil_img), var=sigma ** 2)*255).astype(np.uint8)
            pil_img = patches(pil_img, 50)
            # pil_img = Image.fromarray(pil_img)
            # pil_img.show()
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        if is_mask:
            img_ndarray = img_ndarray/255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.bmp'))
        img_file = list(self.images_dir.glob(name + '.bmp'))

        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # print(mask_file,img_file)
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        # print(name,img.size,mask.size)
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        origen = self.preprocess(img, is_mask=False, not_origen=False)
        img = self.preprocess(img, is_mask=False, not_origen=True)
        mask = self.preprocess(mask, is_mask=True, not_origen=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'image_origen': torch.as_tensor(origen.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
        }


class CarvanaDatasetBF(BasicDatasetBF):
    def __init__(self, images_dir, masks_dir):
        super().__init__(images_dir, masks_dir, mask_suffix='')
