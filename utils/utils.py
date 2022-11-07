import random
import numpy as np
import torch

__all__ = ['init_seeds', 'AverageMeter','Log']

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#!/usr/bin/python
# -*- coding:utf-8 -*-
 
import logging
import time
import os
 
 
class Log(object):
    '''
封装后的logging
    '''
 
    def __init__(self, logger=None, log_name=''):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        self.log_name = log_name
        handler1 = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        handler2 = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        self.logger.addHandler(handler1)
        self.logger.addHandler(handler2) 

        handler1.close()
        handler2.close()
 
    def getlog(self):
        return self.logger