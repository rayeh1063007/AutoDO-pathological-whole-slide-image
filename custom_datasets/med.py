import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random

import glob
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
class MedDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, n_train: int=-1, transform=None):
        self.images_list = glob.glob(os.path.join(images_dir,'*.png'))
        assert len(self.images_list) > n_train, \
            f'n_train {n_train} is lager then avaliable image number {len(self.images_list)}'
        self.images_list = sorted(self.images_list)
        self.masks_list = glob.glob(os.path.join(masks_dir,'*.png'))
        self.masks_list = sorted(self.masks_list)
        if n_train>0:
            self.images_list = self.images_list[:n_train]
            self.masks_list = self.masks_list[:n_train]
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def patch_size(self):
        mask = self.load(self.masks_list[0])
        mask = np.asarray(mask)
        return mask.shape[0]

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        if scale != 1:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img).astype(np.float32)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        
        img = self.load(self.images_list[idx])
        mask = self.load(self.masks_list[idx])

        assert img.size == mask.size, \
            f'Image and mask {self.images_list[idx]} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        if self.transform is not None:
            img = img.transpose(1,2,0)
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            img = img.transpose(2, 0, 1)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'idx': idx
        }

class Med_MultDirDataset(Dataset):
    def __init__(self, images_dir: list, masks_dir: list, scale: float = 1.0, n_train: int=-1, transform=None):
        self.images_list = list()
        for dir in images_dir:
            img_list = glob.glob(os.path.join(dir,'*.png'))
            self.images_list.extend(img_list)
        assert len(self.images_list) > n_train, \
            f'n_train {n_train} is lager then avaliable image number {len(self.images_list)}'
        self.images_list = sorted(self.images_list)

        self.masks_list = list()
        for dir in masks_dir:
            mask_list = glob.glob(os.path.join(dir,'*.png'))
            self.masks_list.extend(mask_list)
        self.masks_list = sorted(self.masks_list)
        if n_train>0:
            self.images_list = self.images_list[:n_train]
            self.masks_list = self.masks_list[:n_train]

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.transform = transform

        logging.info(f'Creating dataset with {len(self.images_list)} examples')

    def __len__(self):
        return len(self.images_list)

    def patch_size(self):
        mask = self.load(self.masks_list[0])
        mask = np.asarray(mask)
        return mask.shape[0]

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        if scale != 1:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img).astype(np.float32)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        
        img = self.load(self.images_list[idx])
        mask = self.load(self.masks_list[idx])

        assert img.size == mask.size, \
            f'Image and mask {self.images_list[idx]} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        if self.transform is not None:
            img = img.transpose(1,2,0)
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            img = img.transpose(2, 0, 1)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'idx': idx
        }

class MedAllDataset(Dataset):
    def __init__(self, data_path: str, scale: float = 1.0, n_train: int=-1,n_normal: float=0.25, n_tumor: float=0.25):
        self.data_path = {
            "good":os.path.join(data_path,"Good"),
            "good_mask":os.path.join(data_path,"Good_mask"),
            "normal":os.path.join(data_path,"normal"),
            "normal_mask":os.path.join(data_path,"normal_mask"),
            "tumor":os.path.join(data_path,"Tumor"),
            "tumor_mask":os.path.join(data_path,"Tumor_mask")
        }
        self.data_list = {
            "good": os.listdir(self.data_path['good']),
            "normal": os.listdir(self.data_path['normal']),
            "tumor": os.listdir(self.data_path['tumor'])
        }
        random.shuffle(self.data_list['good'])
        random.shuffle(self.data_list['normal'])
        random.shuffle(self.data_list['tumor'])
        
        assert n_normal+n_tumor < 1, \
            f'n_normal+n_tumor {n_normal+n_tumor} is lager then avaliable image size 1'
        
        self.n_good = int(n_train*(1-n_normal-n_tumor))
        self.n_normal = int(n_train*n_normal)
        self.n_tumor = int(n_train*n_tumor)
        assert len(self.data_list['good']) > self.n_good, \
            f'n_train {self.n_good} is lager then avaliable image number'
        assert len(self.data_list['normal']) > self.n_normal, \
            f'n_train {self.n_normal} is lager then avaliable image number'
        assert len(self.data_list['good']) > self.n_tumor, \
            f'n_train {self.n_tumor} is lager then avaliable image number'
        self.images_list = []
        self.masks_list = []
        for i in range(self.n_good):
            self.images_list.append(os.path.join(self.data_path['good'],self.data_list['good'][i]))
            self.masks_list.append(os.path.join(self.data_path['good_mask'],self.data_list['good'][i]))
        for i in range(self.n_normal):
            self.images_list.append(os.path.join(self.data_path['normal'],self.data_list['normal'][i]))
            self.masks_list.append(os.path.join(self.data_path['normal_mask'],self.data_list['normal'][i]))
        for i in range(self.n_tumor):
            self.images_list.append(os.path.join(self.data_path['tumor'],self.data_list['tumor'][i]))
            self.masks_list.append(os.path.join(self.data_path['tumor_mask'],self.data_list['tumor'][i]))
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        logging.info(f'Creating dataset with {len(self.images_list)} examples')

    def __len__(self):
        return len(self.images_list)

    
    def patch_size(self):
        mask = self.load(self.masks_list[0])
        mask = np.asarray(mask)
        return mask.shape[0]

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        
        img = self.load(self.images_list[idx])
        mask = self.load(self.masks_list[idx])

        assert img.size == mask.size, \
            f'Image and mask {self.images_list[idx]} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'idx': idx
        }
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
