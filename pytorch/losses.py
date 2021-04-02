import torch
import torch.nn.functional as F
import torch.nn as nn

def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    #device = 'cpu'
    #criterion = nn.NLLLoss()
    #target_dict['target'] = target_dict['target'].long()
    #args = torch.max(target_dict['target'],1)[1]

    #loss = criterion(output_dict['clipwise_output'],args)
    return loss


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll