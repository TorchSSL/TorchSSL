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


def consistency_loss(logits_w, class_acc, it, ds, p_cutoff, use_flex=False):
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    if use_flex:
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()
    else:
        mask = max_probs.ge(p_cutoff).float()
    select = max_probs.ge(p_cutoff).long()

    return (ce_loss(logits_w, max_idx.detach(), use_hard_labels=True,
                    reduction='none') * mask).mean(), select, max_idx.long()
