import torch.nn as nn
import pytorch_ssim
from mc_dataset_infinite_patch3D import *
from convlstm3D import *
from stack_convlstm3D import *

def ssim_smoothl1(out, gt):
    return 20*nn.SmoothL1Loss()(out, gt) + 1 - pytorch_ssim.SSIM()(out, gt)

def ssim_mse(out, gt):
    return 20*nn.MSELoss()(out, gt) + 1 - pytorch_ssim.SSIM()(out, gt)

def ssim_loss(out, gt):
    return 1  - pytorch_ssim.SSIM()(out, gt)