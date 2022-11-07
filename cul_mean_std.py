import random
import warnings
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import numpy as np

def main():
    
    random.seed(999)
    torch.manual_seed(999)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker()


def main_worker():
    
    def compute_mean_and_std(dataset):
        # 输入PyTorch的dataset，输出均值和标准差
        mean_r = 0
        mean_g = 0
        mean_b = 0

        for img, _ in tqdm(dataset):
            img = np.asarray(img) # change PIL Image to numpy array
            mean_b += np.mean(img[:, :, 0])
            mean_g += np.mean(img[:, :, 1])
            mean_r += np.mean(img[:, :, 2])

        mean_b /= len(dataset)
        mean_g /= len(dataset)
        mean_r /= len(dataset)

        diff_r = 0
        diff_g = 0
        diff_b = 0

        N = 0

        for img, _ in tqdm(dataset):
            img = np.asarray(img)

            diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
            diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
            diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

            N += np.prod(img[:, :, 0].shape)

        std_b = np.sqrt(diff_b / N)
        std_g = np.sqrt(diff_g / N)
        std_r = np.sqrt(diff_r / N)

        mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
        std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
        print("mean:", mean)
        print("std:", std)
        return mean, std

    total_trainset = torchvision.datasets.ImageFolder(root='/home/rayeh/workspace/project/med/data/chest_xray'+'/train')
    testset = torchvision.datasets.ImageFolder(root='/home/rayeh/workspace/project/med/data/chest_xray'+'/test')
    
    print('train')
    compute_mean_and_std(total_trainset)
    print('test')
    compute_mean_and_std(testset)
    return

if __name__ == '__main__':    
    main()