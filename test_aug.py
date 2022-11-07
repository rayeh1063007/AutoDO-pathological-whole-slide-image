from typing import Tuple
import torch
import numpy as np
import kornia
import matplotlib.pyplot as plt
from custom_models import *
from custom_datasets import *
temp_wsi_Good_patch_path = '/home/rayeh/workspace/project/med/data/data_512/imgs'
temp_mask_Good_patch_path = '/home/rayeh/workspace/project/med/data/data_512/masks'
train_data = MedDataset(temp_wsi_Good_patch_path, temp_mask_Good_patch_path, 1)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True)
data_len = len(train_data)
trainAugModel = Med_AugmentModel_2(N=data_len, magn=5, apply=True,  mode=1, grad=True, device='cpu')
loader = iter(train_loader)
data = next(loader)
idx = data['idx']
aug_data = trainAugModel(idx,data)