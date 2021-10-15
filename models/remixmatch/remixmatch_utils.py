import torch
import torch.nn.functional as F
from train_utils import ce_loss
import numpy as np

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def mixup_one_target(x, y, gpu, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).cuda(gpu)

    mixed_x = lam*x + (1-lam)*x[index, :]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y, lam

def consistency_loss(logits_w, y):
    return F.mse_loss(torch.softmax(logits_w,dim=-1), y, reduction='mean') 
