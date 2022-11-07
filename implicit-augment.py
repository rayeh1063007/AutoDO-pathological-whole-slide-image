from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset, random_split
#
from custom_models import *
from custom_datasets import *
from custom_transforms import *
from utils import *
import logging
import matplotlib.pyplot as plt
import wandb

# conda activate autodo
# python implicit-augment.py -r run1 --gpu 0 -nr 0.0 -ir 1 --dataset med --epochs 20
# python implicit-augment.py -r run1 --gpu 1 -nr 0.0 -ir 1 --dataset med --epochs 100 --aug-model SEP --los-model NONE --hyper-opt HES
# python implicit-augment.py -r run0 --gpu 1 -nr 0.0 -ir 1 --dataset med --epochs 20 --aug-model SEP --los-model NONE --hyper-opt HES
# python implicit-augment.py -r run2 --gpu 0 -nr 0.0 -ir 1 --dataset med --epochs 20 --aug-model SEP --los-model NONE --hyper-opt HES

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# def seed_it(seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.enabled = True
# seed_it(0)

def get_args():
    parser = argparse.ArgumentParser(description='AutoDO using Implicit Differentiation')
    parser.add_argument('--data', default='./local_data', type=str, metavar='NAME',
                        help='folder to save all data')
    parser.add_argument('--dataset', default='med', type=str, metavar='NAME',
                        help='dataset MNIST/CIFAR10/CIFAR100/SVHN/SVHN_extra/ImageNet/med')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--workers', default=4, type=int, metavar='NUM',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, metavar='NUM',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=str, default='30,55,80', metavar='LR',
                        help='learning rate decay epochs (default: 150,175,195')
    parser.add_argument('--lr-warm-epochs', type=int, default=5, metavar='LR',
                        help='number using cosine annealing (default: False')
    parser.add_argument("--gpu", default='0', type=str, metavar='NUM',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=500, metavar='NUM',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-debug', action='store_true', default=False,
                        help='plot train images for debugging purposes')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of [1:C/2] to [C/2+1:C] labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-sr', '--subsample-ratio', type=float, default=1.0, metavar='N',
                        help='ratio of selected to total labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noise-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy (randomly flipped) labels (default: 0.0)')
    parser.add_argument('-r', '--run-folder', default='run0', type=str,
                        help='dir to save run')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='ablation: estimate DA from test data (default: False)')
    parser.add_argument('--oversplit', action='store_true', default=False,
                        help='ablation: train on all data (default: False)')
    parser.add_argument('--aug-model', default='NONE', type=str,
                        help='type of augmentation model NONE/RAND/AUTO/DADA/SHAred/SEParate parameters (default: NONE)')
    parser.add_argument('--los-model', default='NONE', type=str,
                        help='type of model for other loss hyperparams NONE/SOFT/WGHT/BOTH (default: NONE)')
    parser.add_argument('--hyper-opt', default='NONE', type=str,
                        help='type of bilevel optimization NONE/HES (default: NONE)')
    parser.add_argument('--hyper-steps', type=int, default=0, metavar='NUM',
                        help='number of gradient calculations to achieve grad(L_train)=0 (default: 0)')
    parser.add_argument('--hyper-iters', type=int, default=5, metavar='NUM',
                        help='number of approxInverseHVP iterations inside hyperparameter estimation loop (default: 5)')
    parser.add_argument('--hyper-alpha', type=float, default=0.01, metavar='HO',
                        help='hyperparameter learning rate (default: 0.01)')
    parser.add_argument('--hyper-beta', type=int, default=0, metavar='HO',
                        help='hyperparameter beta (default: 0)')
    parser.add_argument('--hyper-gamma', type=int, default=0, metavar='HO',
                        help='hyperparameter gamma (default: 0)')

    args = parser.parse_args()
    
    return args

def main(args):
    patch_path = '/mnt/Nami/Med_patch/'
    #提取args
    args.hyper_est = True
    args.lr_warm = True
    args.lr_cosine = True
    dataset = args.dataset
    img_scale = args.scale
    overfit = args.overfit
    oversplit = args.oversplit
    hyper_est = args.hyper_est
    hyper_opt = args.hyper_opt
    imbalance_ratio = args.imbalance_ratio
    subsample_ratio = args.subsample_ratio
    noise_ratio = args.noise_ratio
    model_postfix = 'ir_{}_sr_{}_nr_{}'.format(imbalance_ratio, subsample_ratio, noise_ratio)
    run_folder = args.run_folder
    experiment_name = f'autodo_e{args.epochs}_{dataset}_{run_folder}_{model_postfix}_{args.aug_model}_{args.los_model}_{hyper_opt}'
    #設定log
    log_file = f"./Log/{experiment_name}.log"
    logger = Log(__name__, log_file).getlog()
    logger.info(args)
    os.environ["WANDB_API_KEY"] = '816d619d917f02d5ff37c113e8630b5474b1ceaa'
    if hyper_est and hyper_opt=='HES':
        print(experiment_name)
        experiment = wandb.init(project='autodo2', resume='allow', anonymous='must', name=experiment_name)
        experiment.config.update(dict(epochs=args.epochs, batch_size=4, learning_rate=0.00001,run_folder=run_folder,
                                        imbalance_ratio=imbalance_ratio, subsample_ratio=subsample_ratio, noise_ratio=noise_ratio,
                                        aug_model=args.aug_model, los_model=args.los_model, hyper_opt=hyper_opt))

    #設定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    # create folders
    if not os.path.isdir(args.data):
        os.mkdir(args.data)
    save_folder = '{}/{}'.format(args.data, dataset)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    long_run_folder = '{}/{}'.format(save_folder, run_folder)
    print('long_run_folder:', long_run_folder)
    if not os.path.isdir(long_run_folder):
        os.mkdir(long_run_folder)
    model_folder = '{}/{}'.format(save_folder, run_folder)
    print('model_folder:', model_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    aug_mode = 0
    #create dataset
    aug_K, aug_M = 2, 5
    valid_images = None
    train_images = None
    if dataset == 'med':
        # data:
        if run_folder == 'run0':
            #train data: WSI 1~5
            train_data = MedDataset(wsi_Good_patch_path, mask_Good_patch_path, img_scale)
            total_images = len(train_data)
            #test data: WSI 6
            test_data = MedDataset(val_wsi_Good_patch_path, val_mask_Good_patch_path, img_scale)
            valid_images = 0.2
        elif run_folder == 'run1':
            train_data = Med_MultDirDataset([wsi_Good_patch_path,val_wsi_Good_patch_path],[mask_Good_patch_path,val_mask_Good_patch_path])
            total_images = len(train_data)
            test_images = int(total_images*0.1)
            train_images = total_images - test_images
            train_data, test_data = random_split(train_data, [train_images,test_images])
            valid_images = 0.2
        elif run_folder == 'run4':
            train_data = Med_MultDirDataset([wsi_Good_patch_path,val_wsi_Good_patch_path],[mask_Good_patch_path,val_mask_Good_patch_path])
            total_images = len(train_data)
            test_images = int(total_images*0.1)
            train_images = total_images - test_images
            train_data, test_data = random_split(train_data, [train_images,test_images])
            valid_images = 0.2
        # elif run_folder == 'run2': #for test
        #     #train data: WSI 1~5
        #     train_data = MedDataset(temp_wsi_Good_patch_path, temp_mask_Good_patch_path, img_scale)
        #     total_images = len(train_data)
        #     #test data: WSI 6
        #     test_data = MedDataset(temp_wsi_Good_patch_path, temp_mask_Good_patch_path, img_scale)
        #     valid_images = 0.2
        num_classes = 2
        num_channels = 3
        hyperEpochStart = 5
        # WideResNet model:
        task_lr = 0.00001
        train_batch_size = 4 #32
        hyper_batch_size = 2 #32
        args.hyper_theta = ['cls']
        model_name = 'UNet'#'wresnet28_10'
        Dnn_model = UNet(n_channels=3, n_classes=num_classes, bilinear=False).to(device)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(dataset))
    # dataloaders: 根據設定的ir、sr、nr去建立dataloader
    data_file = '{}/data_{}.pt'.format(model_folder, model_postfix)
    if os.path.isfile(data_file):
        if dataset == 'med':
            valid_sub_indices, train_sub_indices = torch.load(data_file) # load saved indices
            valid_images = len(valid_sub_indices)
            train_images = len(train_sub_indices)
        else:
            valid_sub_indices, train_sub_indices, train_targets = torch.load(data_file) # load saved indices
    else:
        if dataset == 'med':
            rs = ShuffleSplit(n_splits=5, test_size=valid_images, random_state=0)
            rs = rs.split(list(range(len(train_data))))
            for _ in range(random.randint(1,5)):
                train_indices, valid_indices = next(rs)
            valid_images = len(valid_indices)
            train_images = len(train_indices)
            train_indices, valid_indices = list(train_indices), list(valid_indices)
            valid_sub_indices = valid_indices
            # save targets for soft label estimation
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False, **kwargs)
            MLEN = len(train_loader.dataset) # dataset size
            BLEN = len(train_loader) # number of batches
            # train_targets = torch.zeros(MLEN, dtype=torch.long)
            # for batch_idx, data in enumerate(train_loader):
            #     if batch_idx % args.log_interval == 0:
            #         logger.info('Reading train batch {}/{}'.format(batch_idx, BLEN))
            #     _, train_target, train_index = data
            #     train_targets[train_index] = train_target
            # subsampling
            SR = int(1.0 * train_images * subsample_ratio) # number of subsampled examples
            train_sr_indices = random.sample(train_indices, SR)
            #
            train_sub_data = torch.utils.data.Subset(train_data, train_sr_indices)
            train_sub_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=False, **kwargs)
            SUB = len(train_sub_loader.dataset)
            logger.info('Train dataset/subset: {}->{}'.format(MLEN, SUB))
            # imbalance
            if imbalance_ratio == 1:
                train_sub_indices = train_sr_indices # use all train subsampled data
            else: # imbalance distort dataset
                # for batch_idx, data in enumerate(train_sub_loader):
                #     image, target, index = data
                #     if batch_idx == 0:
                #         targets = target
                #         indices = index
                #     else:
                #         targets = torch.cat([targets, target])
                #         indices = torch.cat([indices, index])
                # #
                # mskL = targets.lt(num_classes//2) # 0...4
                # indL = mskL.nonzero(as_tuple=False).squeeze()
                # indicesL = torch.index_select(indices, 0, indL)
                # L = indicesL.size(0)
                # #
                # mskU = targets.ge(num_classes//2) # 5...9
                # indU = mskU.nonzero(as_tuple=False).squeeze()
                # indicesU = torch.index_select(indices, 0, indU)
                # U = indicesU.size(0)
                # #
                # S = int(1.0 * L / imbalance_ratio) # number of U examples
                # indS = torch.tensor(random.sample(range(U), S), dtype=torch.long)
                # indicesS = torch.index_select(indicesU, 0, indS)
                # #
                # train_sub_indices = torch.cat([indicesL, indicesS])
                # train_sub_indices = train_sub_indices.tolist()
                # logger.info('Imbalance =', L, U, ':', S, '->', L+S)
                pass
            # label noise
            if noise_ratio > 0.0:
                # num_noisy_labels = round(noise_ratio*len(train_sub_indices))
                # noisy_sub_indices = random.sample(train_sub_indices, num_noisy_labels)
                # train_targets[noisy_sub_indices] = torch.randint(num_classes, (num_noisy_labels,), dtype=torch.long)
                # logger.info('Noisy labels: {:.0f}% ({}/{})'.format(100.0*len(noisy_sub_indices)/len(train_sub_indices), len(noisy_sub_indices), len(train_sub_indices)))\
                pass
            # save indices
            with open(data_file, 'wb') as f:
                torch.save((valid_sub_indices, train_sub_indices), f)
        else:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=valid_images, random_state=0)
            sss = sss.split(list(range(total_images)), train_data.targets)
            for _ in range(random.randint(1,5)):
                train_indices, valid_indices = next(sss)
            
            train_indices, valid_indices = list(train_indices), list(valid_indices)
            valid_sub_indices = valid_indices
            # save targets for soft label estimation
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False, **kwargs)
            MLEN = len(train_loader.dataset) # dataset size
            BLEN = len(train_loader) # number of batches
            train_targets = torch.zeros(MLEN, dtype=torch.long)
            for batch_idx, data in enumerate(train_loader):
                if batch_idx % args.log_interval == 0:
                    logger.info('Reading train batch {}/{}'.format(batch_idx, BLEN))
                _, train_target, train_index = data
                train_targets[train_index] = train_target
            # subsampling
            SR = int(1.0 * train_images * subsample_ratio) # number of subsampled examples
            train_sr_indices = random.sample(train_indices, SR)
            #
            train_sub_data = torch.utils.data.Subset(train_data, train_sr_indices)
            train_sub_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=False, **kwargs)
            SUB = len(train_sub_loader.dataset)
            logger.info('Train dataset/subset: {}->{}'.format(MLEN, SUB))
            # imbalance
            if imbalance_ratio == 1:
                train_sub_indices = train_sr_indices # use all train subsampled data
            else: # distort dataset
                for batch_idx, data in enumerate(train_sub_loader):
                    image, target, index = data
                    if batch_idx == 0:
                        targets = target
                        indices = index
                    else:
                        targets = torch.cat([targets, target])
                        indices = torch.cat([indices, index])
                #
                mskL = targets.lt(num_classes//2) # 0...4
                indL = mskL.nonzero(as_tuple=False).squeeze()
                indicesL = torch.index_select(indices, 0, indL)
                L = indicesL.size(0)
                #
                mskU = targets.ge(num_classes//2) # 5...9
                indU = mskU.nonzero(as_tuple=False).squeeze()
                indicesU = torch.index_select(indices, 0, indU)
                U = indicesU.size(0)
                #
                S = int(1.0 * L / imbalance_ratio) # number of U examples
                indS = torch.tensor(random.sample(range(U), S), dtype=torch.long)
                indicesS = torch.index_select(indicesU, 0, indS)
                #
                train_sub_indices = torch.cat([indicesL, indicesS])
                train_sub_indices = train_sub_indices.tolist()
                logger.info('Imbalance =', L, U, ':', S, '->', L+S)
            # label noise
            if noise_ratio > 0.0:
                num_noisy_labels = round(noise_ratio*len(train_sub_indices))
                noisy_sub_indices = random.sample(train_sub_indices, num_noisy_labels)
                train_targets[noisy_sub_indices] = torch.randint(num_classes, (num_noisy_labels,), dtype=torch.long)
                logger.info('Noisy labels: {:.0f}% ({}/{})'.format(100.0*len(noisy_sub_indices)/len(train_sub_indices), len(noisy_sub_indices), len(train_sub_indices)))
            # save indices
            with open(data_file, 'wb') as f:
                torch.save((valid_sub_indices, train_sub_indices, train_targets), f)
    # samplers
    logger.info('Valid/Train Split: {}/{}'.format(len(valid_sub_indices), len(train_sub_indices)))
    # loaders
    train_sub_data = torch.utils.data.Subset(train_data, train_sub_indices)
    valid_sub_data = torch.utils.data.Subset(train_data, valid_sub_indices)
    if overfit:
        test_loader  = torch.utils.data.DataLoader(test_data,      batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(test_data,      batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)        
    elif oversplit:
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)            
    else:
        test_loader  = torch.utils.data.DataLoader(test_data,      batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
    # train data augmentation model
    # hyperGradEnable 是否要開啟超參數的梯度更新
    if hyper_opt in ['NONE', 'RAND']:
        hyperGradEnable = False
    elif (hyper_opt in ['HES']) and hyper_est:
        hyperGradEnable = True
    elif (hyper_opt in ['HES']) and not(hyper_est):
        hyperGradEnable = False
    else:
        raise NotImplementedError('{} is not supported hyper optimization model!'.format(hyper_opt))
    # save other hyperparameters to arguments
    args.hyper_lr = 0.05
    if dataset == 'ImageNet':
        args.hyper_lr = 0.01
    else:
        args.hyper_lr = 0.05
    args.hyper_start = hyperEpochStart #第幾個epoch開始更新超參數
    args.lr = task_lr
    args.train_batch_size = train_batch_size
    args.num_classes = num_classes
    # task optimizer
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list()
    args.hyper_lr_decay_epochs = list()
    for i in iterations:
        args.lr_decay_epochs.append(int(i))
        args.hyper_lr_decay_epochs.append(int(i)-args.hyper_start)
    args.hyper_epochs = args.epochs-args.hyper_start
    if args.lr_warm:
        args.lr_warmup_from = args.lr/10.0
        args.hyper_lr_warmup_from = args.hyper_lr/10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
            hyper_eta_min = args.hyper_lr * (args.lr_decay_rate ** 3)
            args.hyper_lr_warmup_to = hyper_eta_min + (args.hyper_lr - hyper_eta_min) * (1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr
            args.hyper_lr_warmup_to = args.hyper_lr
    #
    optimizer = optim.RMSprop(Dnn_model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, min_lr=1e-8)  # goal: maximize Dice score
    # hyper models
    T = total_images
    L = len(test_loader.dataset)
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    logger.info('Test/Valid/Train Split: {}/{}/{} out of total {} train images'.format(L,M,N,T))
    # validation data loss/augmentation model
    if hyper_est:
        validLosModel = Med_LossModel(N=1, C=num_classes, apply=False, model='NONE', grad=False, sym=False, device=device).to(device)
        validAugModel = Med_AugmentModel_2(N=1, magn=aug_M, apply=False, mode=aug_mode, grad=False, device=device).to(device)
    # train data loss/augmentation models
    symmetricKlEnable = False if (imbalance_ratio == 1) and (noise_ratio == 0.0) else True
    trainLosModel = Med_LossModel(N=T, C=num_classes, apply=True, model=args.los_model, grad=hyperGradEnable, sym=symmetricKlEnable, device=device).to(device)
    # select model
    if   args.aug_model in ['NONE', 'RAND', 'AUTO', 'DADA']:
        trainAugModel = Med_AugmentModel_2(N=1, magn=aug_M, apply=False, mode=aug_mode, grad=False,           device=device).to(device)
    elif args.aug_model == 'SHA':
        trainAugModel = Med_AugmentModel_2(N=1, magn=aug_M, apply=True,  mode=aug_mode, grad=hyperGradEnable, device=device).to(device)
    elif args.aug_model == 'SEP':
        trainAugModel = Med_AugmentModel_2(N=T, magn=aug_M, apply=True,  mode=aug_mode, grad=hyperGradEnable, device=device).to(device)
    else:
        raise NotImplementedError('{} is not supported train augmentation model!'.format(args.aug_model))
    # hyperoptimizer
    hyperParams = list(trainLosModel.parameters()) + list(trainAugModel.parameters())
    hyperOptimizer = optim.RMSprop(hyperParams, lr=args.hyper_lr)
    hyperScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hyperOptimizer, args.epochs-args.hyper_start)
    # initial step to save pretrained model
    best_test_score = 0.0
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if overfit:
        model_name = 'overfit_' + model_name
    if oversplit:
        model_name = 'oversplit_' + model_name
    run_name = '{}_e{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
        model_name, args.epochs, hyper_opt, hyper_est, args.aug_model, args.los_model, model_postfix)
    # writer = SummaryWriter('./logs/{}/{}_{}_{}'.format(dataset, run_folder, run_name, run_date))
    checkpoint_file = '{}/best_{}.pt'.format(model_folder, run_name)
    # load hypermodel with estimated hyperparameters
    if not(hyper_est):
        load_name = '{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
            model_name, hyper_opt, 'True', args.aug_model, args.los_model, model_postfix)
        load_file = '{}/best_{}.pt'.format(model_folder, load_name)
        checkpoint = torch.load(load_file)
        Dnn_model.load_state_dict(checkpoint['model_state_dict'])
        trainLosModel.load_state_dict(checkpoint['reweight_state_dict'])
        trainAugModel.load_state_dict(checkpoint['augment_state_dict'])
        logger.info('Loading pretrained model...', load_file)
    #logger.info('Run: {}/{} - {}\n'.format(model_folder, run_name, run_date))
    logger.info('Run: {}/{}'.format(model_folder, run_name))
    dDivs = 4*[0.0]
    exp_history={
        'train': [],
        'test': [],
    }
    global_img_step = 0
    for epoch in range(0, args.epochs):
        logger.info('{:.0f}% ({}/{})'.format(100.0*epoch/args.epochs, epoch, args.epochs))
        adjust_learning_rate(args, optimizer, epoch)
        testEnable  = True #if  (epoch >= hyperEpochStart) else False
        hyperEnable = True if ((epoch >  hyperEpochStart) and hyperGradEnable)  else False
        if not(hyper_est): # train classifier only
            pass
            # train_loss = classTrain(args, encoder, decoder, optimizer, device, train_loader, epoch, trainLosModel, trainAugModel, logger)
        else:
            # train hyperparameters
            if hyper_opt == 'HES' and hyperEnable:
                hyper_adjust_learning_rate(args, hyperOptimizer, epoch-hyperEpochStart)
                dDivs, global_img_step = Med_hyperHesTrain(args, Dnn_model, optimizer, device, valid_loader, hyper_loader, epoch, hyperEpochStart,
                            trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer, logger, experiment, global_img_step)
            # train encoder and classifier
            train_loss, train_score, global_img_step = Med_innerTrain(args, Dnn_model, optimizer, scheduler, device, train_loader, epoch, trainLosModel, trainAugModel, logger, experiment, global_img_step, hyperEnable)
            exp_history['train'].append((train_loss,train_score))
        # test
        if testEnable:
            test_loss, test_score = Med_innerTest(args, Dnn_model, device, test_loader, epoch, logger)
            exp_history['test'].append((test_loss,test_score))
            # save checkpoint (test_score-based)
            if test_score >= best_test_score:
                logger.info('SAVING trained model at epoch {} with {:.4f} Dice score'.format(epoch, test_score))
                Med_save(Dnn_model, trainLosModel, trainAugModel, test_score, epoch, checkpoint_file)
                best_test_score = test_score
        else:
            test_score, test_loss = 0.0, 0.0
        # save log
        # writer.add_scalar('Dice score', test_score, epoch)
        # writer.add_scalar('Train Loss', train_loss, epoch)
        # writer.add_scalar('Test Loss', test_loss, epoch)
    script_file = os.path.realpath(sys.argv[0])
    path = os.path.dirname(script_file)
    if not os.path.exists(f"{path}/history"):
        os.makedirs(f"{path}/history")
    history_save_path = f"{path}/history/{run_folder}_{run_name}.npz"
    np.savez(history_save_path, 
            train = np.array(exp_history['train']),
            test = np.array(exp_history['test']))
    if not os.path.exists(f"{path}/picture"):
        os.makedirs(f"{path}/picture")
    picture_name = f"{path}/picture/{run_folder}_{run_name}.jpg"
    load_and_plot(history_save_path, picture_name)
    logger.info(f'save train history at: {picture_name}')
    #
    logger.info('BEST trained model has {:.4f} Dice score'.format(best_test_score))
    # writer.flush()
    # writer.close()

def load_and_plot(history_path, picture_name):

    model_history = np.load(history_path)
    epoch = range(1,len(model_history['train'])+1)
    plt.figure(figsize=(10,7))

    plt.subplot(2, 1, 1)
    plt.plot(epoch, model_history["train"][:,1], '.-',color="green",label="train score")
    plt.plot(epoch, model_history["test"][:,1], '.-',color="black",label="test score")
    plt.title("score vs. epoches")
    plt.ylabel("Dice score")
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    
    plt.plot(epoch, model_history["train"][:,0], '.-',color="green",label="train loss")
    plt.plot(epoch, model_history["test"][:,0], '.-',color="black",label="test loss")
    plt.title("loss vs. epoches")
    plt.ylabel("loss")
    plt.legend(loc='upper left')
    plt.savefig(picture_name)

if __name__ == '__main__':
    args = get_args()
    main(args)
