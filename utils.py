
"""
Created on Wed Sep 18 20:27:19 2019
@author: MOHAMEDR002
"""
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import math
from torch import nn
from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     def __init__(self,  data,labels):
#         """Reads source and target sequences from processing file ."""
#         self.input_tensor=(torch.from_numpy(data)).float()
#
#         self.label=(torch.torch.FloatTensor(labels))
#         self.num_total_seqs = len(self.input_tensor)
#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         input_seq = self.input_tensor[index]
#         input_labels = self.label[index]
#         return input_seq,input_labels
#     def __len__(self):
#         return self.num_total_seqs
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
def loop_iterable(iterable):
    while True:
        yield from iterable

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
def scoring_func(error_arr):
    pos_error_arr = error_arr[error_arr >= 0] 
    neg_error_arr = error_arr[error_arr < 0]
    score = 0 
    for error in neg_error_arr:
            score = math.exp(-(error / 13)) - 1 + score 
    for error in pos_error_arr: 
            score = math.exp(error / 10) - 1 + score
    return score

def roll(x, shift: int, dim: int = -1, fill_pad: int = None):

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift))], dim=dim)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
