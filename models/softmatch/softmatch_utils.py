import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value



def consistency_loss(logits_s, logits_w, mask, name='ce', T=0.5, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        _, max_idx = torch.max(pseudo_label, dim=-1)
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
        return masked_loss.mean(), mask

    else:
        assert Exception('Not Implemented consistency_loss')