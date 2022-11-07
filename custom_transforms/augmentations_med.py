from pickletools import uint8
import random, sys
sys.path.append(r'/home/rayeh/workspace/project/med/Med_AutoDO')
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import numpy as np
from color.hsbcoloraugmenter import HsbColorAugmenter
from color.hedcoloraugmenter import HedColorAugmenter
import albumentations as A

__all__ = ['Aug_list','aug_operator','Aug_list_tensor']

class Hed_shift(A.ImageOnlyTransform):
    def __init__(self, factor, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.factor = factor

    def get_transform_init_args_names(self):
        return "factor",

    def hed(self, image, factor):
        #print('image',image.shape)
        image=np.transpose(image,[2,0,1])
        image = (image*255).astype(np.uint8)
        augmentor= HedColorAugmenter(haematoxylin_sigma_range=(-factor, factor), haematoxylin_bias_range=(-factor, factor),
                                                eosin_sigma_range=(-factor, factor), eosin_bias_range=(-factor, factor),
                                                dab_sigma_range=(-factor, factor), dab_bias_range=(-factor, factor),
                                                cutoff_range=(0.15, 0.85))
        ##To select a random magnitude value between -factor:factor, if commented the m value will be constant
        augmentor.randomize()
        image = np.transpose(augmentor.transform(image),[1,2,0])
        image = np.clip(image, 0, 255)/255
        return image
    
    def apply(self, image, mask=None, **params):
        return self.hed(image, self.factor)

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        return params
class Hsv_shift(A.ImageOnlyTransform):
    def __init__(self, factor, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.factor = factor

    def get_transform_init_args_names(self):
        return "factor",

    def hsv(self, image, factor):
        #print('image',image.shape)
        image=np.transpose(image,[2,0,1])
        image = (image*255).astype(np.uint8)
        augmentor= HsbColorAugmenter(hue_sigma_range=(-factor, factor), saturation_sigma_range=(-factor, factor), brightness_sigma_range=(0, 0))
        #To select a random magnitude value between -factor:factor, if commented the m value will be constant
        augmentor.randomize()
        image = np.transpose(augmentor.transform(image),[1,2,0])
        image = np.clip(image, 0, 255)/255
        return image
    
    def apply(self, image, mask=None, **params):
        factor = self.factor
        return self.hsv(image, factor)

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        return params
class Aug_ShearX(A.DualTransform):

    def __init__(self,factor, always_apply=False, p=0.5):
        super(Aug_ShearX, self).__init__(always_apply, p)
        self.factor = factor

    def ShearX(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        image = Image.fromarray((img * 255).astype(np.uint8))
        image = image.transform(image.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
        image = np.asarray(image)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image/255

    def apply(self, img, **params):
        return self.ShearX(img, self.factor)

    def apply_to_mask(self, img, **params):
        return self.ShearX(img, self.factor)

    def get_params(self):
        return {"factor": self.factor}

    def get_transform_init_args_names(self):
        return "factor",
class Aug_ShearY(A.DualTransform):

    def __init__(self,factor, always_apply=False, p=0.5):
        super(Aug_ShearY, self).__init__(always_apply, p)
        self.factor = factor

    def ShearY(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        image = Image.fromarray((img * 255).astype(np.uint8))
        image = image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
        image = np.asarray(image)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image/255

    def apply(self, img, **params):
        return self.ShearY(img, self.factor)

    def apply_to_mask(self, img, **params):
        return self.ShearY(img, self.factor)

    def get_params(self):
        return {"factor": self.factor}

    def get_transform_init_args_names(self):
        return "factor",
class Aug_AutoContract(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def get_transform_init_args_names(self):
        return "p",

    def Contrast(self, img):
        image = Image.fromarray((img * 255).astype(np.uint8))
        image = PIL.ImageOps.autocontrast(image)
        image = np.asarray(image)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image/255
    
    def apply(self, image, mask=None, **params):
        return self.Contrast(image)

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        return params
class Aug_Equalize(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def get_transform_init_args_names(self):
        return "p",

    def Equalize(self, img):
        image = Image.fromarray((img * 255).astype(np.uint8))
        image = PIL.ImageOps.equalize(image)
        image = np.asarray(image)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image/255
    
    def apply(self, image, mask=None, **params):
        return self.Equalize(image)

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        return params

#level: 0.0~1.0 間距:0.1

def Aug_list():
    l = [
        # ('GaussNoise', 100, 1000), #0
        # ('GaussianBlur', 0, 13), #1
        # ('ElasticTransform', 0, 1),# 2
        ('Hed', 0.0, 0.3), #3
        ('Hsv', 0.0, 0.3), #4
        # ('Sharpen', 0.0, 1.0), #5
        # ('Brightness', 0.0, 2.0), #Brightness 6 ok
        # ('Contrast', 0.0, 2.0), #Contrast 7 ok
        # ('Saturation', 0.0, 2.0), #saturation 8 ok
        # ('Equalize', 0, 1), #9 NO
        # ('AutoContract', 0, 1), #10 NO
        # ('Rotate', -30, 30),  #Rotate 11 ok
        # ('TranslateX', -0.45, 0.45), #TranslateX 12 ok
        # ('TranslateY', -0.45, 0.45), #TranslateY 13 ok
        # ('ShearX', -0.3, 0.3), #14 ok
        # ('ShearY', -0.3, 0.3) #15 ok
        ] # total 16
    return l

def Aug_list_tensor():
    l = [
        ('brightness', [0]), #0
        ('contrast', [1]), #1
        ('saturation', [2]),# 2
        ('Hsv', [3,4,5]), #3
        ('Hed', [6,7,8]), #4
        ('gaussian blur', [9]), #5
        ('sharpen', [11]), # 6 ok
        ('gaussian noise', [10]), # 7 ok
        ('elastic transform', [12]), # 8 ok
        ('Rotate', [13]),  # 9 ok
        ('TranslateX', [14]), # 10 ok
        ('TranslateY', [15]), # 11 ok
        ('ShearX', [16]), # 12 ok
        ('ShearY', [17]), # 13 ok
        ('Equalize', [0]), # 14 NO
        ] # total 15
    return l

Aug_dict = {name: (v1, v2) for name, v1, v2 in Aug_list()}

def aug_operator(idx, pr, level):
    name, low, high = Aug_list()[idx]
    factor=(level * (high - low) + low)
    if name == "GaussNoise":
        return A.GaussNoise(var_limit=(factor,factor),p=pr)
    elif name == "GaussianBlur":
        factor = int(factor)
        if factor%2==0:
            factor += 1
        return A.GaussianBlur(blur_limit=(factor,factor), sigma_limit=0,p=pr)
    elif name == "ElasticTransform":
        return A.ElasticTransform(p=pr)
    elif name == "Hed":
        return Hed_shift(factor=factor,p=pr)
    elif name == "Hsv":
        return Hsv_shift(factor=factor,p=pr)
    elif name == "Sharpen":
        return A.Sharpen(alpha=(level,level),p=pr)
    elif name == "Brightness":
        return A.ColorJitter(brightness=(factor,factor),contrast=0.0,saturation=0.0,hue=0.0,p=pr)
    elif name == "Contrast":
        return A.ColorJitter(brightness=0.0,contrast=(factor,factor),saturation=0.0,hue=0.0,p=pr)
    elif name == "Saturation":
        return A.ColorJitter(brightness=0.0,contrast=0.0,saturation=(factor,factor),hue=0.0,p=pr)
    elif name == "Equalize":
        return Aug_Equalize(p=pr)
    elif name == "AutoContract":
        return Aug_AutoContract(p=pr)
    elif name == "Rotate":
        return A.Affine(rotate=factor,p=pr)
    elif name == "TranslateX":
        return A.Affine(translate_percent={'x':factor,'y':0.0},p=pr)
    elif name == "TranslateY":
        return A.Affine(translate_percent={'x':0.0,'y':factor},p=pr)
    elif name == "ShearX":
        return Aug_ShearX(factor=factor,p=pr)
    elif name == "ShearY":
        return Aug_ShearY(factor=factor,p=pr)
    else:
        NameError(f'no augment name {name}.')
        