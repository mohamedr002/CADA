import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np 
device = torch.device('cuda:1')
import random
from torch.autograd import Variable
import pandas as pd
from models.my_models import *
from models.models_config import get_model_config, initlize
from trainer.pre_train_test_split import pre_train
from data.mydataset import create_dataset, create_dataset_full
import wandb
from torch.utils.tensorboard import SummaryWriter
#Different Domain Adaptation  approaches
from trainer.cross_domain_models.ADDA_NCE_v2 import cross_domain_train


select_method='ATL_NCE'
# hyper parameters
hyper_param={ 'FD001_FD002': {'epochs':75,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD001_FD003': {'epochs':75,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD001_FD004': {'epochs':75,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD002_FD001': {'epochs':20,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.001},
              'FD002_FD003': {'epochs':20,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.001},
              'FD002_FD004': {'epochs':20,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.001},
              'FD003_FD001': {'epochs':100,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD003_FD002': {'epochs':100,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD003_FD004': {'epochs':100,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD004_FD001': {'epochs':50,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD004_FD002': {'epochs':50,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.2},
              'FD004_FD003': {'epochs':50,'batch_size':256,'lr':5e-5,'nce_lr':1e-2, 'alpha_nce':0.001}}
# load dataset
data_path= "/home/emad/Mohamed2/Mohamed/data/cmapps_train_test_cross_domain.pt"
my_dataset = torch.load(data_path)

# configuration setup 
config = get_model_config('LSTM') 
config.update({'num_runs':1, 'save':False, 'tensorboard':False,'tsne':False,'tensorboard_epoch':False,'k_disc':100, 'k_clf':1,'iterations':1}) 

if config['tensorboard']:
  wandb.init(project="Domain Adaptation for with Contrastive Coding",name=f"{select_method}",dir= "/home/emad/Mohamed2/ATL_NCE/visualize/", sync_tensorboard=True) 
  wandb.config  = hyper_param  



if __name__ == '__main__':
  df=pd.DataFrame();res = [];full_res = []
  print('=' * 89)
  print (f'Domain Adaptation using: {select_method}')
  print('=' * 89)
  for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
      for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
          if src_id != tgt_id:
              total_loss = []
              total_score = []
              for run_id in range(config['num_runs']):
                  src_only_loss, src_only_score, test_loss, test_score = cross_domain_train(hyper_param,device,config,LSTM_RUL,my_dataset,src_id,tgt_id,run_id)
                  total_loss.append(test_loss)
                  total_score.append(test_score)
              loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
              score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
              full_res.append((f'run_id:{run_id}',f'{src_id}-->{tgt_id}', f'{src_only_loss:2.2f}' ,f'{loss_mean:2.2f}',f'{loss_std:2.2f}',f'{src_only_score:2.2f}',f'{score_mean:2.2f}',f'{score_std:2.2f}'))
              
  df= df.append(pd.Series((f'{select_method}')), ignore_index=True)
  df= df.append(pd.Series(("run_id", 'scenario','src_only_loss', 'mean_loss','std_loss', 'src_only_score', f'mean_score',f'std_score')), ignore_index=True)
  df = df.append(pd.DataFrame(full_res), ignore_index=True)
  print('=' * 89)
  print (f'Results using: {select_method}')
  print('=' * 89)
  print(df.to_string())
  df.to_csv(f'/home/emad/Mohamed2/results/Final_results_{select_method}.csv')