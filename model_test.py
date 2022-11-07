from custom_models import *
from custom_datasets import *
from custom_transforms import *
from utils import *
import os
import logging
import torch
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
experiment_name = f'autodo_(lr-decay-epochs)_e100_med_run1_ir_1_sr_1.0_nr_0.0_SEP_NONE_HES'
#設定log
log_file = f"./Log/{experiment_name}.log"
logger = Log(__name__, log_file).getlog()
test_wsi_Good_patch_path = '/mnt/Nami/Med_test_patch_2_512/Good'
test_mask_Good_patch_path = '/mnt/Nami/Med_test_patch_2_512/Good_mask'
test_data = MedDataset(test_wsi_Good_patch_path, test_mask_Good_patch_path, 1)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=False, num_workers=8)
Dnn_model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
load_file = './local_data/med/run1/best_UNet_e100_opt_HES_est_True_aug_model_SEP_los_model_NONE_ir_1_sr_1.0_nr_0.0.pt'
checkpoint = torch.load(load_file)
Dnn_model.load_state_dict(checkpoint['Dnn_model_state_dict'])
logger.info('Loading pretrained model...', load_file)
logger.info('test new WSI')
loss_fn = nn.CrossEntropyLoss()
amp = False
cnt = 0
steps = 0
test_loss = 0
test_score = 0
iter_count = 0
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
for batch in test_loader:
    steps += 1
    images = batch['image']
    true_masks = batch['mask']

    assert images.shape[1] == Dnn_model.n_channels, \
        f'Network has been defined with {Dnn_model.n_channels} input channels, ' \
        f'but loaded images have {images.shape[1]} channels. Please check that ' \
        'the images are loaded correctly.'

    images = images.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.long)

    with torch.cuda.amp.autocast(enabled=amp):
        masks_pred = Dnn_model(images)
        dice = dice_loss(F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, Dnn_model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True)
        loss = loss_fn(masks_pred, true_masks) + dice
    test_score += (1-dice.item())
    test_loss += loss.item()
    iter_count += 1

    del masks_pred, images, true_masks
test_loss /= iter_count
test_score /= iter_count
logger.info(f'test loss: {test_loss}, test score: {test_score}')