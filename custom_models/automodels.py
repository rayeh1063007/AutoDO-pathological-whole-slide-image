import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import kornia
import math
from utils import *
from custom_models.utils import *
from custom_transforms import plot_debug_images, aug_operator, Aug_list, Aug_list_tensor
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
import random
import albumentations as A
from tqdm import tqdm
import wandb
from typing import Callable, Tuple, Union, List, Optional, Dict, cast
from kornia.augmentation.base import AugmentationBase2D
class RandomGaussianNoise(AugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    Args:
        mean (float): The mean of the gaussian distribution. Default: 0.
        std (float): The standard deviation of the gaussian distribution. Default: 1.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> RandomGaussianNoise(mean=0., std=1., p=1.)(img)
        tensor([[[[ 2.5410,  0.7066],
                  [-1.1788,  1.5684]]]])
    """

    def __init__(self,
                 mean: float = 0.,
                 std: float = 1.,
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(RandomGaussianNoise, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return input + params['noise'].to(input.device) * self.std + self.mean

_SOFTPLUS_UNITY_ = 1.4427
_SIGMOID_UNITY_ = 2.0
_EPS_ = 1e-8 # small regularization constant

__all__ = ['innerTest', 'innerTrain', 'classTrain', 'vizStat', 'Med_hyperHesTrain', 'Med_innerTrain', 'Med_innerTest','Med_AugmentModel_2',
            'LossModel', 'AugmentModelNONE', 'AugmentModel', 'Med_AugmentModel', 'Med_LossModel', 'hyperHesTrain']


def metricCE(logit, target):
    return F.cross_entropy(logit, target)

def insert_policy(policies, transform):
    for _ in range(1):
        policy = random.choice(policies)
        for name, pr, level in policy:
            transform.transforms.insert(0, aug_operator(name,pr,level))
    return transform

class LossModel(nn.Module):
    def __init__(self, N, C, init_targets, apply, model, grad, sym, device):
        """
        N : totla image size
        C : num_classes
        model : los_model (WGHT=update loss-reweighting) (SOFT : update soft-labeling) (BOTH=update loss-reweighting and soft-labeling)
        """
        super(LossModel, self).__init__()
        self.alpha = 0.1
        self.apply = apply
        self.sym = sym
        self.act = nn.Softplus()
        self.actSoft     = nn.Softmax(dim=1)
        self.actLogSoft  = nn.LogSoftmax(dim=1)
        if self.apply:
            initWeights = torch.zeros(N)
            #soft label
            eps = 0.05 # alpha label-smoothing constant [0.05-0.2]
            diff = math.log(1.0-C+C/eps) # solution for softmax
            initSoftTargets = diff*(F.one_hot(init_targets, num_classes=C)-0.5)
            if model == 'NONE':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                self.soft = False
                self.cls_loss = nn.CrossEntropyLoss(reduction='none')
            elif model == 'WGHT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                self.soft = False
                self.cls_loss = nn.CrossEntropyLoss(reduction='none')
            elif model == 'SOFT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            elif model == 'BOTH':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            else:
                raise NotImplementedError('{} is not supported loss model'.format(model))
        else:
            self.cls_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, idx, logit, target):
        if self.apply: # always train
            hyperW = _SOFTPLUS_UNITY_ * self.act(self.hyperW[idx])
            hyperS = self.hyperS[idx]
            hyperH = torch.argmax(hyperS, dim=1)
            if self.soft:
                if self.sym:
                    cls_loss = 0.5*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)) +
                                             self.cls_loss(self.actLogSoft(hyperS), self.actSoft(logit[0])), dim=1)
                else:
                    cls_loss = 1.0*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)), dim=1)
            else:
                cls_loss = self.cls_loss(logit[0], hyperH)
            cls_loss = hyperW*cls_loss
            cls_loss = cls_loss.mean()
            return cls_loss
        else: # always valid
            cls_loss = self.cls_loss(logit[0], target)
            cls_loss = cls_loss.mean()
            return cls_loss

class Med_LossModel(nn.Module):
    def __init__(self, N, C, apply, model, grad, sym, device):
        """
        N : totla image size
        C : num_classes
        model : los_model (WGHT=update loss-reweighting) (SOFT : update soft-labeling) (BOTH=update loss-reweighting and soft-labeling)
        """
        super(Med_LossModel, self).__init__()
        self.alpha = 0.1
        self.apply = apply
        self.sym = sym
        self.act = nn.Softplus()
        self.actSoft     = nn.Softmax(dim=1)
        self.actLogSoft  = nn.LogSoftmax(dim=1)
        self.num_classes = C
        if self.apply:
            initWeights = torch.zeros(N)
            #soft label
            eps = 0.05 # alpha label-smoothing constant [0.05-0.2]
            diff = math.log(1.0-C+C/eps) # solution for softmax
            # initSoftTargets = diff*(F.one_hot(init_targets, num_classes=C)-0.5)
            if model == 'NONE':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                # self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                # self.soft = False
                self.cls_loss = nn.CrossEntropyLoss()
            elif model == 'WGHT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                # self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                self.soft = False
                self.cls_loss = nn.CrossEntropyLoss()
            elif model == 'SOFT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                # self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            elif model == 'BOTH':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                # self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            else:
                raise NotImplementedError('{} is not supported loss model'.format(model))
        else:
            self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, idx, logit, target):
        if self.apply: # always train
            dice = dice_loss(F.softmax(logit, dim=1).float(),
                                F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True)
            loss = self.cls_loss(logit, target)
            hyperW = _SOFTPLUS_UNITY_ * self.act(self.hyperW[idx])
            # hyperS = self.hyperS[idx]
            # hyperH = torch.argmax(hyperS, dim=1)
            # if self.soft:
            #     if self.sym:
            #         cls_loss = 0.5*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)) +
            #                                  self.cls_loss(self.actLogSoft(hyperS), self.actSoft(logit[0])), dim=1)
            #     else:
            #         cls_loss = 1.0*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)), dim=1)
            # else:
            #     cls_loss = self.cls_loss(logit[0], hyperH)
            cls_loss = (hyperW*(loss+dice)).mean()
            return cls_loss, dice
        else: # always valid
            dice = dice_loss(F.softmax(logit, dim=1).float(),
                                F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True)
            loss = self.cls_loss(logit, target)
            return loss, dice

class AugmentModelNONE(nn.Module):
    def __init__(self):
        super(AugmentModelNONE, self).__init__()

    def forward(self, idx, x):
        return x


class AugmentModel(nn.Module):
    def __init__(self, N, magn, apply, mode, grad, device):
        """
        N : totla image size
        magn : magnitude
        """
        super(AugmentModel, self).__init__()
        # enable/disable manual augmentation
        self.N = N
        self.apply = apply
        self.device = device
        self.mode = mode # mode
        self.K = 7 # number of affine magnitude params
        self.J = 0 # number of color magnitude params
        K = self.K
        J = self.J
        #預設機率與強度超參數的初始值
        magnNorm = torch.ones(1)*magn/10.0 # normalize to 10 like in RandAugment
        probNorm = torch.ones(1)*1/(K-2) # 1/(K-2) probability
        magnLogit = torch.log(magnNorm/(1-magnNorm)) # convert to logit
        probLogit = torch.log(probNorm/(1-probNorm)) # convert to logit
        # affine transforms (mid, range)
        self.angle = [0.0, 30.0] # [-30.0:30.0] rotation angle
        self.trans = [0.0, 0.45] # [-0.45:0.45] X/Y translate
        self.shear = [0.0, 0.30] # [-0.30:0.30] X/Y shear
        # self.scale = [1.0, 0.50] # [ 0.50:1.50] X/Y scale
        # color transforms (mid, range)
        self.bri = [0.0, 0.9] # [-0.9:0.9] brightness
        self.con = [1.0, 0.9] # [0.1:1.9] contrast
        self.sat = [0.1, 1.9] # [-0.30:0.30] saturation
        #self.hue = [1.0, 0.50] # [ 0.70:1.30] hue
        #self.gam = [1.0, 0.50] # [ 0.70:1.30] gamma
        # actP: 機率的激活函數，actM: 強度的激活函數
        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()
        # 建立機率與強度的可學習超參數矩陣 shape = (擴增方法數量*資料數量)
        self.paramP = nn.Parameter(probLogit*torch.ones(K+J,N), requires_grad=grad)
        self.paramM = nn.Parameter(magnLogit*torch.ones(K+J,N), requires_grad=grad)

    def forward(self, idx, x):
        B,C,H,W = x.shape
        device = self.device
        mode = self.mode
        if self.apply:
            K = self.K
            J = self.J
            # learnable hyperparameters
            if self.N == 1:
                paramPos = torch.log(    self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramM = self.actM(self.paramM).repeat(1,B) # (K+J)xB [0:1], default=magn
            else:
                paramPos = torch.log(    self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的使用機率之參數
                paramNeg = torch.log(1.0-self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的不使用機率之參數
                paramM = self.actM(self.paramM[:,idx]) # (K+J)xB [0:1], default=magn
            paramP = torch.cat([paramPos.view(-1,1), paramNeg.view(-1,1)], dim=1) # B*(K+J)x2
            # reparametrize probabilities and magnitudes
            sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True).to(device) # B*(K+J)x2 對K+種擴增方法的使用與不使用做採樣
            sampleP = sampleP[:,0] #取使用那個col，1為使用
            sampleP = sampleP.reshape(K+J,B) #reshape回batch
            # reparametrize magnitudes
            #sampleM = paramM[:K] * torch.rand(K,B).to(device) # KxB, prior: U[0,1]
            sampleM = paramM[:K] * torch.randn(K,B).to(device) # KxB, prior: N(0,1) #
            # affine augmentations
            R: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device) #3*3對角線為1
            # define the rotation angle
            ANG: torch.tensor = sampleP[0] * sampleM[0] * self.angle[1] # B(0/1)*U[0,1]*M/10
            # define the rotation center
            CTR: torch.tensor = torch.cat((W*torch.ones(B).to(device)//2, H*torch.ones(B).to(device)//2)).view(-1,2)
            # define the scale factor
            SCL: torch.tensor = torch.zeros_like(CTR).to(device)
            SCL[:,0] = self.scale[0] + sampleP[5] * sampleM[5] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            SCL[:,1] = self.scale[0] + sampleP[6] * sampleM[6] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            R[:,0:2] = kornia.get_rotation_matrix2d(CTR, ANG, SCL)
            # translation: border not defined yet
            T: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            T[:,0,2] = W * sampleP[1] * sampleM[1] * self.trans[1]
            T[:,1,2] = H * sampleP[2] * sampleM[2] * self.trans[1]
            # shear: check this
            S: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            S[:,0,1] = sampleP[3] * sampleM[3] * self.shear[1]
            S[:,1,0] = sampleP[4] * sampleM[4] * self.shear[1]
            # apply the transformation to original image
            M: torch.tensor = torch.bmm(torch.bmm(S,T),R)
            if mode == 0: #upscale
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = self.bri[0] + sampleP[6] * sampleC[0] * self.bri[1] # mid + B(0/1)*U[0,1]*M/10
            #    CON: torch.tensor = self.con[0] + sampleP[7] * sampleC[1] * self.con[1]
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped

        else: # process val to compensate for Kornia artifacts!
            if mode == 0: #upscale
                M: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device)
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped = x #torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = torch.zeros(B).to(device)
            #    CON: torch.tensor = torch.ones(B).to(device)
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped

class Med_AugmentModel_2(nn.Module):
    def __init__(self, N, magn, apply, mode, grad, device):
        """
        N : totla image size
        magn : magnitude
        """
        super(Med_AugmentModel_2, self).__init__()
        # enable/disable manual augmentation
        self.N = N
        self.apply = apply
        self.device = device
        self.mode = mode # mode
        self.K = 15 # number of ops
        self.KM = 18 # number of m param
        K = self.K
        Km = self.KM
        #預設機率與強度超參數的初始值
        magnNorm = torch.ones(1)*magn/10.0 # normalize to 10 like in RandAugment
        probNorm = torch.ones(1)*1/K # 1/(K-2) probability
        magnLogit = torch.log(magnNorm/(1-magnNorm)) # convert to logit
        probLogit = torch.log(probNorm/(1-probNorm)) # convert to logit
        # affine transforms (mid, range)
        self.angle = [0.0, 30.0] # [-30.0:30.0] rotation angle 0
        self.trans = [0.0, 0.45] # [-0.45:0.45] X/Y translate 1, 2
        self.shear = [0.0, 0.30] # [-0.30:0.30] X/Y shear 3, 4
        self.sharpen = [1.0, 2.5, 2.5]
        # gaussian
        self.blur_kernel = [1, 2]
        self.blur_sigma = [2.6,2.5] # [0.1:5.1]
        self.noise = [0.25,0.25] # [0:10]
        # elastic_transform
        self.elt = [0.0, 5.0]
        # color transforms (mid, range)
        self.bri = [0.0, 1.0] # [-0.9:0.9] brightness
        self.con = [1.0, 2.5, 2.5] # [0.1:1.9] contrast
        self.sat = [1.0, 2.5, 2.5] # [-0.30:0.30] saturation
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                        [0.07, 0.99, 0.11],
                                        [0.27, 0.57, 0.78]], dtype=torch.float32).to(self.device)
        self.hed_from_rgb = torch.inverse(self.rgb_from_hed).to(self.device)
        # actP: 機率的激活函數，actM: 強度的激活函數
        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()
        # 建立機率與強度的可學習超參數矩陣 shape = (擴增方法數量*資料數量)
        self.paramP = nn.Parameter(probLogit*torch.ones(K,N), requires_grad=grad)
        self.paramM = nn.Parameter(magnLogit*torch.ones(Km,N), requires_grad=grad)

    def rgb_to_hed(self, rgb):
        r: torch.Tensor = rgb[..., 0, :, :]
        g: torch.Tensor = rgb[..., 1, :, :]
        b: torch.Tensor = rgb[..., 2, :, :]
        m = self.hed_from_rgb
        h: torch.Tensor = m[0,0] * r + m[0,1] * g + m[0,2] * b
        e: torch.Tensor = m[1,0] * r + m[1,1] * g + m[1,2] * b
        d: torch.Tensor = m[2,0] * r + m[2,1] * g + m[2,2] * b

        out: torch.Tensor = torch.stack([h, e, d], -3)

        return out

    def hed_to_rgb(self, stains):
        h: torch.Tensor = stains[..., 0, :, :]
        e: torch.Tensor = stains[..., 1, :, :]
        d: torch.Tensor = stains[..., 2, :, :]
        m = self.rgb_from_hed
        r: torch.Tensor = m[0,0] * h + m[0,1] * e + m[0,2] * d
        g: torch.Tensor = m[1,0] * h + m[1,1] * e + m[1,2] * d
        b: torch.Tensor = m[2,0] * h + m[2,1] * e + m[2,2] * d
        
        out: torch.Tensor = torch.stack([r, g, b], -3)

        return out

    def forward(self, idx, data):
        x = data['image']
        y = data['mask']
        y = y.unsqueeze(1).float()
        y_warped: torch.tensor = y
        x_warped: torch.tensor = x
        B,C,H,W = x.shape
        if self.apply:
            K = self.K
            Km = self.KM
            # learnable hyperparameters
            if self.N == 1:
                paramPos = torch.log(    self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramM = self.actM(self.paramM).repeat(1,B) # (K+J)xB [0:1], default=magn
            else:
                paramPos = torch.log(    self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的使用機率之參數
                paramNeg = torch.log(1.0-self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的不使用機率之參數
                paramM = self.actM(self.paramM[:,idx]) # (K+J)xB [0:1], default=magn
            paramP = torch.cat([paramPos.view(-1,1), paramNeg.view(-1,1)], dim=1) # B*(K+J)x2
            # reparametrize probabilities and magnitudes
            sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True).to(self.device) # B*(K+J)x2 對K+種擴增方法的使用與不使用做採樣
            sampleP = sampleP[:,0] #取使用那個col，1為使用
            sampleP = sampleP.reshape(K,B) #reshape回batch
            # sampleP = torch.ones(K,B).to(self.device)
            # reparametrize magnitudes
            # sampleM = paramM * torch.rand(Km,B).to(self.device) # KxB, prior: U[0,1]
            # sampleM = sampleM*2-1.
            sampleM = paramM * torch.randn(Km,B).to(self.device) # KxB, prior: N(0,1)
            sampleM = (sampleM-sampleM.min())/(sampleM.max()-sampleM.min())*2-1
            # sampleM = (paramM-paramM.min())/(paramM.max()-paramM.min())*2-1
            # sampleM = torch.clamp(paramM,min=-1.0,max=1.0)
            ################# color augmentations #################
            # equalize
            x_warped_eq = kornia.enhance.equalize(x)
            EQU = x-x_warped_eq
            for i in range(B):
                EQU[i] = EQU[i].clone() * sampleP[14,i]
            x_warped = x_warped - (EQU)
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            # contrast, brightness, saturation
            BRI: torch.tensor = sampleP[0] * sampleM[0] 
            CON: torch.tensor = self.con[0] + sampleP[1] * (self.con[1] + sampleM[1] * self.con[2]) # mid + B(0/1)*U[0,1]*M/10
            SAT: torch.tensor = self.sat[0] + sampleP[2] * (self.sat[1] + sampleM[2] * self.sat[2])
            for i in range(B):
                x_warped[i] = kornia.enhance.adjust_saturation(x_warped[i].clone(), SAT[i])
                x_warped[i] = torch.clamp(x_warped[i].clone(),min=0.0,max=1.0)
                x_warped[i] = kornia.enhance.adjust_brightness(x_warped[i].clone(), BRI[i])
                x_warped[i] = torch.clamp(x_warped[i].clone(),min=0.0,max=1.0)
                x_warped[i] = kornia.enhance.adjust_contrast(x_warped[i].clone(), CON[i])
                x_warped[i] = torch.clamp(x_warped[i].clone(),min=0.0,max=1.0)
            # HSV
            x_warped = kornia.color.rgb_to_hsv(x_warped)
            for i in range(B):
                # Augment the hue channel.
                x_warped[i, 0, :, :] = x_warped[i, 0, :, :].clone() / (2.*math.pi)
                x_warped[i, 0, :, :] = x_warped[i, 0, :, :].clone() + (sampleM[3,i] % 1.0) * sampleP[3,i]
                x_warped[i, 0, :, :] = x_warped[i, 0, :, :].clone() % 1.0
                x_warped[i, 0, :, :] = x_warped[i, 0, :, :].clone() * (2.*math.pi)
                # Augment the Saturation channel.
                if sampleM[4,i] < 0.0:
                    x_warped[i, 1, :, :] = x_warped[i, 1, :, :].clone() * (1.0 + sampleM[4,i] * sampleP[3,i])
                else:
                    x_warped[i, 1, :, :] = x_warped[i, 1, :, :].clone() * (1.0 + (1.0 - x_warped[i, 1, :, :]) * sampleM[4,i] * sampleP[3,i])
                # Augment the Brightness channel.
                if sampleM[5,i] < 0.0:
                    x_warped[i, 2, :, :] = x_warped[i, 2, :, :].clone() * (1.0 + sampleM[5,i] * sampleP[3,i])
                else:
                    x_warped[i, 2, :, :] = x_warped[i, 2, :, :].clone() + (1.0 - x_warped[i, 2, :, :]) * sampleM[5,i] * sampleP[3,i]
            x_warped = kornia.color.hsv_to_rgb(x_warped)
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            # HED
            x_warped = self.rgb_to_hed(x_warped,)
            for i in range(B):
                x_warped[i, 0, :, :] = x_warped[i, 0, :, :].clone() * (1.0 + sampleM[6,i] * sampleP[4,i]) + (torch.rand(1).to(self.device) * sampleM[6,i] * sampleP[4,i])
                x_warped[i, 1, :, :] = x_warped[i, 1, :, :].clone() * (1.0 + sampleM[7,i] * sampleP[4,i]) + (torch.rand(1).to(self.device) * sampleM[7,i] * sampleP[4,i])
                x_warped[i, 2, :, :] = x_warped[i, 2, :, :].clone() * (1.0 + sampleM[8,i] * sampleP[4,i]) + (torch.rand(1).to(self.device) * sampleM[8,i] * sampleP[4,i])
            x_warped = self.hed_to_rgb(x_warped)
            x_warped = torch.clamp(x_warped, min=0.0, max=1.0)
            # gaussian blur
            BLUR_K: torch.tensor = self.blur_kernel[0] + sampleP[5] * self.blur_kernel[1]
            BLUR_S: torch.tensor = self.blur_sigma[0] + sampleM[9] * self.blur_sigma[1]
            for i in range(B):
                x_warped[i] = kornia.filters.gaussian_blur2d(x_warped[i].clone().unsqueeze(0), (int(BLUR_K[i]), int(BLUR_K[i])), (BLUR_S[i], BLUR_S[i]))
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            # sharpen
            SHARP: torch.tensor = self.sharpen[0] + sampleP[7] * (self.sharpen[1] + self.sharpen[2] * sampleM[11])
            x_warped = kornia.enhance.sharpness(x_warped, SHARP)
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            # gaussian noise
            NOISE: torch.tensor = sampleP[6] * (self.noise[0] + sampleM[10] * self.noise[1])
            for i in range(B):
                GN = RandomGaussianNoise(std=NOISE[i], p=1.)
                x_warped[i] = torch.clamp(GN(x_warped[i].clone().unsqueeze(0)), min=0.0, max=1.0)
            ################# affine augmentations #################
            # elastic_transform
            ELT: torch.tensor = self.elt[0] + sampleP[8] * sampleM[12] * self.elt[1]
            for i in range(B):
                elt_ops = kornia.augmentation.AugmentationSequential(
                    kornia.augmentation.RandomElasticTransform(alpha=(ELT[i],ELT[i]),mode='nearest',p=1.),
                    data_keys=['input', 'mask'],
                )
                x_warped[i], y_warped[i] = elt_ops(x_warped[i].clone(),y[i].clone())
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            # affine augmentations
            R: torch.tensor = torch.zeros(B,3,3).to(self.device) + torch.eye(3).to(self.device) #3*3對角線為1
            # define the rotation angle
            ANG: torch.tensor = sampleP[9] * sampleM[13] * self.angle[1] # B(0/1)*U[0,1]*M/10
            # define the rotation center
            CTR: torch.tensor = torch.cat((W*torch.ones(B).to(self.device)//2, H*torch.ones(B).to(self.device)//2)).view(-1,2)
            # define the scale factor
            SCL: torch.tensor = torch.ones((B, 2)).to(self.device)
            R[:,0:2] = kornia.geometry.transform.get_rotation_matrix2d(CTR, ANG, SCL)
            # translation: border not defined yet
            T: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(self.device)
            T[:,0,2] = W * sampleP[10] * sampleM[14] * self.trans[1]
            T[:,1,2] = H * sampleP[11] * sampleM[15] * self.trans[1]
            # shear: check this
            S: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(self.device)
            S[:,0,1] = sampleP[12] * sampleM[16] * self.shear[1]
            S[:,1,0] = sampleP[13] * sampleM[17] * self.shear[1]
            # apply the transformation to original image
            M: torch.tensor = torch.bmm(torch.bmm(S,T),R)
            x_warped = kornia.geometry.transform.warp_perspective(x_warped, M, dsize=(H,W), mode='bilinear', padding_mode ='zeros')
            x_warped = torch.clamp(x_warped,min=0.0,max=1.0)
            y_warped = kornia.geometry.transform.warp_perspective(y_warped, M, dsize=(H,W), mode='nearest', padding_mode ='zeros')
            y_warped = y_warped.squeeze(1)
            aug_ops = []
            for i in range(B):
                batch_ops = []
                for j in range(len(sampleP[:,i])):
                    if sampleP[j,i]:
                        name, m_idx = Aug_list_tensor()[j]
                        batch_ops.append((name,sampleM[m_idx,i].view(-1).cpu().detach()))
                if not batch_ops:
                    batch_ops.append(('idenity',[1.]))
                aug_ops.append(batch_ops)
            return {
                'image': x_warped.float().contiguous(),
                'mask': y_warped.long().contiguous(),
                'aug_ops': aug_ops
            }

        else: 
            y_warped = y_warped.squeeze(1)
            return {
                'image': x_warped.float().contiguous(),
                'mask': y_warped.long().contiguous()
            }

class Med_AugmentModel(nn.Module):
    def __init__(self, N, magn, apply, mode, grad, device):
        """
        N : totla image size
        magn : magnitude
        """
        super(Med_AugmentModel, self).__init__()
        # enable/disable manual augmentation
        self.N = N
        self.apply = apply
        self.device = device
        self.mode = mode # mode
        self.aug_ops_num = 16 # number of affine magnitude params
        K = self.aug_ops_num
        #預設機率與強度超參數的初始值
        magnNorm = torch.ones(1)*magn/10.0 # normalize to 10 like in RandAugment
        probNorm = torch.ones(1)*1/(K-2) # 1/(K-2) probability
        magnLogit = torch.log(magnNorm/(1-magnNorm)) # convert to logit
        probLogit = torch.log(probNorm/(1-probNorm)) # convert to logit
        # actP: 機率的激活函數，actM: 強度的激活函數
        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()
        # 建立機率與強度的可學習超參數矩陣 shape = (擴增方法數量*資料數量)
        self.paramP = nn.Parameter(probLogit*torch.ones(K,N), requires_grad=grad)
        self.paramM = nn.Parameter(magnLogit*torch.ones(K,N), requires_grad=grad)

    def forward(self, idx, x):
        B,C,H,W = x['image'].shape
        idx = idx.cpu().detach().numpy()
        device = self.device
        mode = self.mode
        if self.apply:
            K = self.aug_ops_num
            # learnable hyperparameters
            if self.N == 1:
                paramPos = torch.log(    self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramM = self.actM(self.paramM).repeat(1,B) # (K+J)xB [0:1], default=magn
            else:
                paramPos = torch.log(    self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的使用機率之參數
                paramNeg = torch.log(1.0-self.actP(self.paramP[:,idx])) # [-Inf:0] [擴增方法數, data point number] 取針對該資料的不使用機率之參數
                paramM = self.actM(self.paramM[:,idx]) # (K+J)xB [0:1], default=magn
            paramP = torch.cat([paramPos.view(-1,1), paramNeg.view(-1,1)], dim=1) # B*(K+J)x2
            # reparametrize probabilities and magnitudes
            sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True).to(device) # B*(K+J)x2 對K+種擴增方法的使用與不使用做採樣
            sampleP = sampleP[:,0] #取使用那個col，1為使用
            sampleP = sampleP.reshape(K,B) #reshape回batch
            # reparametrize magnitudes
            sampleM = paramM[:K] * torch.rand(K,B).to(device) # KxB, prior: U[0,1]
            # sampleM = paramM[:K] * torch.randn(K,B).to(device) # KxB, prior: N(0,1)
            img = x['image'].cpu().detach().numpy()
            mask = x['mask'].cpu().detach().numpy()
            aug_ops = []
            for i in range(B):
                image_aug = img[i]
                mask_aug = mask[i]
                transform = A.Compose([])
                used_ops = []
                for p in range(K):
                    if sampleP[p,i] :
                        transform.transforms.insert(0, aug_operator(p,1,sampleM[p,i].item()))
                        name, _, _ = Aug_list()[p]
                        used_ops.append((name,sampleM[p,i].item()))
                image_aug = image_aug.transpose(1,2,0)
                image_aug = image_aug.astype(np.float32)
                mask_aug = mask_aug.astype(np.float32)
                aug_data = transform(image=image_aug, mask=mask_aug)
                image_aug = aug_data['image']
                mask_aug = aug_data['mask']
                image_aug = image_aug.transpose(2, 0, 1)
                img[i] = image_aug
                mask[i] = mask_aug
                if not used_ops:
                    used_ops.append(('idenity',1))
                aug_ops.append(used_ops)
            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'aug_ops': aug_ops
            }

        else: # process val to compensate for Kornia artifacts!
            x_warped = x #torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = torch.zeros(B).to(device)
            #    CON: torch.tensor = torch.ones(B).to(device)
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped

def hyperHesTrain(args, Dnn_model, optimizer, device, valid_loader, train_loader, epoch, start,
        trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer, logger):
    logger = logger
    Dnn_model.eval()
    # encoder.eval()
    # decoder.eval()
    validLosModel.eval()
    validAugModel.eval()
    trainLosModel.train()
    trainAugModel.train()
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    B = len(train_loader) # number of batches
    v_loader_iterator = iter(valid_loader)
    t_loader_iterator = iter(train_loader)
    dDivs = 4*[0.0]
    #
    for param_group in optimizer.param_groups:
        task_lr = param_group['lr']
    hyperParams = list()
    for n,p in trainAugModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    for n,p in trainLosModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    theta = list()
    # for n,p in decoder.named_parameters():
    #     if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
    #         theta.append(p)
    for n,p in Dnn_model.named_parameters():
        if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
            theta.append(p)
    #
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    v0Norms    = AverageMeter()
    v1Norms    = AverageMeter()
    mvpNorms   = AverageMeter()
    end = time.time()
    ######## start train hyper model ########
    for batch_idx in range(0,1*B):
        # sample val batch
        try:
            vData, vTarget, vIndex = next(v_loader_iterator)
        except StopIteration:
            v_loader_iterator = iter(valid_loader)
            vData, vTarget, vIndex = next(v_loader_iterator)
        # sample train batch
        try:
            tData, tTarget, tIndex = next(t_loader_iterator)
        except StopIteration:
            t_loader_iterator = iter(train_loader)
            tData, tTarget, tIndex = next(t_loader_iterator)
        #
        tData  = tData.to(device)
        tTarget= tTarget.to(device)
        tIndex = tIndex.to(device)
        vData  = vData.to(device)
        vTarget= vTarget.to(device)
        vIndex = vIndex.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        hyper_lr = hyper_warmup_learning_rate(args, epoch-start, batch_idx, B, hyperOptimizer)

        # v1 = dL_v / dTheta: Lx1
        optimizer.zero_grad()
        vData = validAugModel(vIndex, vData)
        vOutput = Dnn_model(vData)
        # vEncode = encoder(vData)
        # vOutput = decoder(vEncode)
        vLoss = validLosModel(vIndex, vOutput, vTarget)
        g1 = torch.autograd.grad(vLoss, theta)
        v1 = [e.detach().clone() for e in g1]

        # v0 = dL_t / dTheta: Lx1
        optimizer.zero_grad()
        tData = trainAugModel(tIndex, tData)
        tOutput = Dnn_model(tData)
        # tEncode = encoder(tData)
        # tOutput = decoder(tEncode)
        tLoss = trainLosModel(tIndex, tOutput, tTarget)
        g0 = torch.autograd.grad(tLoss, theta, create_graph=True)
        v0 = [e.detach().clone() for e in g0]

        # v2 = H^-1 * v0: Lx1
        v2 = [-e.detach().clone() for e in v1]
        if args.hyper_iters > 0: # Neumann series
            for j in range(0, args.hyper_iters):
                ns = torch.autograd.grad(g0, theta, grad_outputs=v1, create_graph=True)
                v1 = [v1[l] - args.hyper_alpha*e for l,e in enumerate(ns)]
                v2 = [v2[l] - e.detach().clone() for l,e in enumerate(v1)]

        # MVP compute
        v0Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v0)])) # gLt*gLt
        v1Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v1,v1)])) # gLv*gLv
        v2Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v1)])) # gLv*gLt
        mmd = v0Norm + v1Norm -2.0*v2Norm
        bDivs = list([v0Norm, v1Norm, -2.0*v2Norm, mmd])
        dDivs = [e1+e2 for e1,e2 in zip(dDivs, bDivs)]
        mvpNorm = mmd
        #
        v0Norms.update(v0Norm.item())
        v1Norms.update(v1Norm.item())
        mvpNorms.update(mvpNorm.item())

        # v3 = (dL_t / dLambda) * v2: Px1
        hyperOptimizer.zero_grad()
        torch.autograd.backward(g0, grad_tensors=v2)
        hyperOptimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger
        if batch_idx % 200 == 0: #args.log_interval == 0:
            logger.info('hyperTrain batch {:.0f}% ({}/{}), task_lr={:.6f}, hyper_lr={:.6f}\t'
                #'gLtNorm {v0Norm.val:.4f} ({v0Norm.avg:.4f})\t'
                #'gLvNorm {v1Norm.val:.4f} ({v1Norm.avg:.4f})\t'
                #'mvpNorm {mvpNorm.val:.4f} ({mvpNorm.avg:.4f})\n'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, task_lr, hyper_lr,
                #v0Norm=v0Norms, v1Norm=v1Norms, mvpNorm=mvpNorms,
                batch_time=batch_time, data_time=data_time))
    #
    dDivs = [e/B for e in dDivs]
    #logger.info('Epoch: {}\t Divergence: {:.4f}'.format(epoch, dDivs[-1]))
    return dDivs

def Med_hyperHesTrain(args, Dnn_model, optimizer, device, valid_loader, train_loader, epoch, start,
        trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer, logger, experiment, global_img_step):
    logger = logger
    Dnn_model.eval()
    # encoder.eval()
    # decoder.eval()
    validLosModel.eval()
    validAugModel.eval()
    trainLosModel.train()
    trainAugModel.train()
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    B = len(train_loader) # number of batches
    v_loader_iterator = iter(valid_loader)
    t_loader_iterator = iter(train_loader)
    dDivs = 4*[0.0]
    #
    task_lr = optimizer.param_groups[0]['lr']
    hyperParams = list()
    for n,p in trainAugModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    for n,p in trainLosModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    if not hyperParams:
        raise NotImplementedError('hyperParams is empty')
    theta = list()
    # for n,p in decoder.named_parameters():
    #     if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
    #         theta.append(p)
    for n,p in Dnn_model.named_parameters():
        if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
            theta.append(p)
    #
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    v0Norms    = AverageMeter()
    v1Norms    = AverageMeter()
    mvpNorms   = AverageMeter()
    end = time.time()
    batch_count = 0
    valid_loss = 0
    ######## start train hyper model ########
    for batch_idx in range(0,1*B):
        # sample val batch
        try:
            vData = next(v_loader_iterator)
        except StopIteration:
            v_loader_iterator = iter(valid_loader)
            vData = next(v_loader_iterator)
        # sample train batch
        try:
            tData = next(t_loader_iterator)
        except StopIteration:
            t_loader_iterator = iter(train_loader)
            tData = next(t_loader_iterator)
        #
        batch_count += 1
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        hyper_lr = hyper_warmup_learning_rate(args, epoch-start, batch_idx, B, hyperOptimizer)
        # v1 = dL_v / dTheta: Lx1
        optimizer.zero_grad()
        vIndex = vData['idx']
        vData = validAugModel(vIndex, vData)
        
        vImg = vData['image']
        vMask = vData['mask']

        vImg  = vImg.to(device)
        vMask= vMask.to(device)
        vIndex = vIndex.to(device)

        vOutput = Dnn_model(vImg)
        vLoss, vDice = validLosModel(vIndex, vOutput, vMask)
        dice_loss = vLoss+vDice
        valid_loss += dice_loss.item()
        g1 = torch.autograd.grad(dice_loss, theta)
        v1 = [e.detach().clone() for e in g1]
        # v0 = dL_t / dTheta: Lx1
        optimizer.zero_grad()
        tData['image'] = tData['image'].to(device)
        tData['mask'] = tData['mask'].to(device)
        tIndex = tData['idx']
        tData = trainAugModel(tIndex, tData)

        tImg = tData['image']
        tMask = tData['mask']
        tOps = tData['aug_ops']

        tImg  = tImg.to(device)
        tMask = tMask.to(device)
        tIndex = tIndex.to(device)

        tOutput = Dnn_model(tImg)
        tLoss, tDice = trainLosModel(tIndex, tOutput, tMask)
        g0 = torch.autograd.grad(tLoss, theta, create_graph=True)
        v0 = [e.detach().clone() for e in g0]
        # v2 = H^-1 * v0: Lx1
        v2 = [-e.detach().clone() for e in v1]
        if args.hyper_iters > 0: # Neumann series
            for j in range(0, args.hyper_iters):
                ns = torch.autograd.grad(g0, theta, grad_outputs=v1, create_graph=True)
                v1 = [v1[l] - args.hyper_alpha*e for l,e in enumerate(ns)]
                v2 = [v2[l] - e.detach().clone() for l,e in enumerate(v1)]
        # MVP compute
        v0Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v0)])) # gLt*gLt
        v1Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v1,v1)])) # gLv*gLv
        v2Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v1)])) # gLv*gLt
        mmd = v0Norm + v1Norm -2.0*v2Norm
        bDivs = list([v0Norm, v1Norm, -2.0*v2Norm, mmd])
        dDivs = [e1+e2 for e1,e2 in zip(dDivs, bDivs)]
        mvpNorm = mmd
        #
        v0Norms.update(v0Norm.item())
        v1Norms.update(v1Norm.item())
        mvpNorms.update(mvpNorm.item())
        # v3 = (dL_t / dLambda) * v2: Px1
        hyperOptimizer.zero_grad()
        torch.autograd.backward(g0, grad_tensors=v2)
        hyperOptimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger
        if batch_idx % 2500 == 0: #args.log_interval == 0:
            if batch_idx==0:
                for tag, value in trainAugModel.named_parameters():
                    tag = tag.replace('/', '.')
                    value = value.data.cpu()[:,0]
                    logger.info(f'data point 0 weight: {tag}_{value}')
            histograms = {}
            for tag, value in trainAugModel.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
            experiment.log({
                            'task learning rate': task_lr,
                            'hyper learning rate': hyper_lr,
                            # 'validation Dice': tLoss.item(),
                            'hyper batch idx': batch_idx,
                            'images': wandb.Image(tImg[0].cpu()),
                            'masks': {
                                'true': wandb.Image(tMask[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(tOutput, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'epoch': epoch,
                            **histograms
                        })
            logger.info(f'hyperTrain epoch: {epoch}, batch_idx: {batch_idx}, global_img_step: {global_img_step}, aug_ops:{tOps[0]}')
            logger.info('hyperTrain batch {:.0f}% ({}/{}), task_lr={:.6f}, hyper_lr={:.6f}\t'
                'gLtNorm {v0Norm.val:.4f} ({v0Norm.avg:.4f})\t'
                'gLvNorm {v1Norm.val:.4f} ({v1Norm.avg:.4f})\t'
                'mvpNorm {mvpNorm.val:.4f} ({mvpNorm.avg:.4f})\n'
                .format(100.0*batch_idx/B, batch_idx, B, task_lr, hyper_lr,
                v0Norm=v0Norms, v1Norm=v1Norms, mvpNorm=mvpNorms,
                ))
            global_img_step += 1
    #
    experiment.log({'valid loss': valid_loss/batch_count,
                    'epoch': epoch,
                    'gLtNorm val': v0Norms.val,
                    'gLvNorm val':v1Norms.val,
                    'mvpNorm val':mvpNorms.val,
                    'gLtNorm avg': v0Norms.avg,
                    'gLvNorm avg':v1Norms.avg,
                    'mvpNorm avg':mvpNorms.avg,})
    dDivs = [e/B for e in dDivs]
    #logger.info('Epoch: {}\t Divergence: {:.4f}'.format(epoch, dDivs[-1]))
    return dDivs, global_img_step

def Med_innerTrain(args, Dnn_model, optimizer, scheduler, device, loader, epoch, losModel, augModel, logger, experiment, global_img_step, hyperEnable):
    logger = logger
    Dnn_model.train()
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    pbr = tqdm(loader)
    #
    score = 0
    for batch_idx, data in enumerate(pbr):
        # measure data loading time
        data_time.update(time.time() - end)
        
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # warm-up learning rate
        # model+loss
        index = data['idx'].to(device)
        if hyperEnable:
            data['image'] = data['image'].to(device)
            data['mask'] = data['mask'].to(device)
            aug_data = augModel(index, data)
            img = aug_data['image'].to(device)
            t_mask = aug_data['mask'].to(device)
            aug_ops = aug_data['aug_ops']
        else:
            img = data['image'].to(device)
            t_mask = data['mask'].to(device)
        output = Dnn_model(img)
        loss, dice = losModel(index, output, t_mask)
        score += 1-dice.item()
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        pbr.set_postfix({'loss' : (train_loss/(batch_idx+1)), 'score': score/(batch_idx+1)})
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=data['image'], fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=img, fname=fname)
        if batch_idx % 1000 == 0:
            experiment.log({
                            'train batch idx': batch_idx,
                            'train images': wandb.Image(img[0].cpu()),
                            'train masks': {
                                'true': wandb.Image(t_mask[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(output, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'epoch': epoch
                        })
            if hyperEnable:
                logger.info(f'Train epoch: {epoch}, batch_idx: {batch_idx}, global_img_step: {global_img_step}, aug_ops:{aug_ops[0]}')
            else:
                logger.info(f'Train epoch: {epoch}, batch_idx: {batch_idx}, global_img_step: {global_img_step}')
            global_img_step += 1
        # logger
        # if batch_idx % args.log_interval == 0:
        #     logger.info('innerTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #         100.0*batch_idx/B, batch_idx, B, lr,loss=losses))
    score /= B
    # scheduler.step(score)
    train_loss /= B
    experiment.log({
                        'train Dice score': score,
                        'train loss': train_loss,
                        'epoch': epoch
                    })
    logger.info('Epoch: {}\t Inner Train loss: {:.4f}, acc={:.4f}, lr={:.6f}\t'.format(epoch, train_loss, score, optimizer.param_groups[0]['lr'],loss=losses))
    return train_loss, score, global_img_step

def innerTrain(args, Dnn_model, optimizer, device, loader, epoch, losModel, augModel, logger):
    logger = logger
    Dnn_model.train()
    # encoder.train() # train encoder model
    # decoder.train() # train decoder model
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    for batch_idx, data in enumerate(loader):
        image = data[0].to(device)
        target= data[1].to(device)
        index = data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # model+loss
        aug_image = augModel(index, image)
        output = Dnn_model(aug_image)
        # encode = encoder(aug_image)
        # output = decoder(encode)
        loss = losModel(index, output, target)
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=aug_image, fname=fname)
        # logger
        if batch_idx % args.log_interval == 0:
            logger.info('innerTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, lr,
                loss=losses, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    logger.info('Epoch: {}\t Inner Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def classTrain(args, Dnn_model, optimizer, device, loader, epoch, losModel, augModel, logger, experiment):
    logger = logger
    Dnn_model.train() # train decoder
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    train_score = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    end = time.time()
    pbr = tqdm(loader)
    # score = AverageMeter()
    #
    for batch_idx, data in enumerate(pbr):
        image = data['image'].to(device)
        target= data['mask'].to(device)
        index = data['idx'].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # augment+encoder
        # with torch.no_grad():
        #     aug_image = augModel(index, image)
        #     encode = encoder(aug_image)
        # classifier
        output = Dnn_model(image)
        loss, dice = losModel(index, output, target)
        # score.update(1-dice.item())
        train_score += (1-dice.item())
        # losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbr.set_postfix({'loss' : (train_loss/(batch_idx+1)), 'score': train_score/(batch_idx+1)})
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger
        # if batch_idx % args.log_interval == 0:
        #     logger.info('classTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         'Score {score.val:.4f} ({score.avg:.4f})\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
        #         100.0*batch_idx/B, batch_idx, B, lr,
        #         loss=losses,score=score, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    train_score /= B
    logger.info('Epoch: {}\t Class Train loss: {:.4f}, Train score: {:.4f}'.format(epoch, train_loss, train_score))
    experiment.log({
                            'train Dice score': train_score,
                            'train loss': train_loss,
                            'epoch': epoch
                        })
    return train_loss, train_score

def Med_innerTest(args, Dnn_model, device, loader, epoch, logger, experiment):
    logger = logger
    Dnn_model.eval() # eval encoder
    #
    B = len(loader) # number of batches
    test_loss = 0.0
    score = 0
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_fn = nn.CrossEntropyLoss()
    end = time.time()
    pbr = tqdm(loader)
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(pbr):
            image = data['image'].to(device)
            target= data['mask'].to(device)
            # plot images for debugging
            if args.plot_debug and (epoch == 0) and (batch_idx < 64):
                fname = 'test_batch_{}_{}.png'.format(batch_idx, args.dataset)
                plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            #
            output = Dnn_model(image)
            dice = dice_loss(F.softmax(output, dim=1).float(),
                                F.one_hot(target, 2).permute(0, 3, 1, 2).float(),
                                multiclass=True)
            loss = loss_fn(output, target) + dice
            losses.update(loss)
            test_loss += loss.item()
            score += 1-dice.item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pbr.set_postfix({'loss' : test_loss/(batch_idx+1), 'score': score/(batch_idx+1)})
            # logger
            # if batch_idx % args.log_interval == 0:
            #     logger.info('innerTest batch {:.0f}% ({}/{})\t'
            #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(100.0*batch_idx/B, batch_idx, B,
            #         loss=losses))
    #
    test_loss /= B
    score = score / B
    logger.info('Epoch: {}\t Test loss: {:.4f}, score: {:.4f}'.format(epoch, test_loss, score))
    experiment.log({
                            'test loss': test_loss,
                            'test Dice score': score,
                            'epoch': epoch
                        })

    return test_loss, score

def innerTest(args, encoder, decoder, device, loader, epoch, logger):
    logger = logger
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    #
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    test_loss = 0.0
    correct = 0
    miss_indices = list()
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image = data[0].to(device)
            target= data[1].to(device)
            index = data[2].to(device)
            # plot images for debugging
            if args.plot_debug and (epoch == 0) and (batch_idx < 64):
                fname = 'test_batch_{}_{}.png'.format(batch_idx, args.dataset)
                plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            test_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if batch_idx % args.log_interval == 0:
                logger.info('innerTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    test_loss /= B
    acc = 100.0 * correct / M
    logger.info('Epoch: {}\t Test loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss, correct, M, acc))

    return acc, test_loss, miss_indices


def vizStat(args, encoder, decoder, device, loader, T, F):
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    total_loss = 0.0
    correct = 0
    #
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    #
    fv = torch.zeros(T, F).to(device)
    gt = torch.zeros(T, dtype=torch.long).to(device)
    pr = torch.zeros(T, dtype=torch.long).to(device)
    mi = torch.tensor([], dtype=torch.long).to(device)
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image  = data[0].to(device)
            target = data[1].to(device)
            index  = data[2].to(device)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            total_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            mask = match.le(0)
            fv[index] = encode
            gt[index] = target.view(-1)
            pr[index] = pred.view(-1)
            mi = torch.cat([mi, torch.masked_select(index.view(-1, 1), mask)])
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if 1: #batch_idx % args.log_interval == 0:
                print('vizTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    total_loss /= B
    acc = 100.0 * correct / M
    print('Loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(total_loss, correct, M, acc))

    return fv, gt, pr, mi
