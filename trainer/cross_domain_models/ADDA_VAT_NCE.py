import sys

sys.path.append("..")
from utils import *
from data.mydataset import create_dataset_full
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import copy
import numpy as np
device = torch.device('cuda:1')
import time
from trainer.cross_domain_models.VAT import  VATLoss
#from tensorboardX  import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import wandb

def dicriminator():
    discriminator = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    return discriminator
    
class NCL_model(nn.Module): 
    def __init__(self, input_dim=64, seq_length=30, out_dim=14):
      super(NCL_model,self).__init__()
      self.input_dim=input_dim
      self.seq_length= seq_length 
      self.out_dim = out_dim
      self.softmax  = nn.Softmax()
      self.lsoftmax = nn.LogSoftmax()
      self.base_model= nn.Linear(self.input_dim,self.out_dim) 
      self.ncl_model =  nn.ModuleList([self.base_model for i in range(self.seq_length)])
    def forward (self, f_t, input_x):
        nce = 0 
        batch_size= input_x.size(0)
        # input_f--> output features of feature extractor #dim (batch_size, feature_dim)
        # input_x--> original input the feature extractor # dim (batch_size,seq_length,input_dim)
        # preds is output after applying module_list on the features f: from batch_size, feature_dim)-->(batch_size,seq_length,out_dim)
        preds = torch.empty((batch_size, self.seq_length,self.out_dim)).float()
        for i in range  (self.seq_length):
            preds[:, i, :] = self.ncl_model[i](f_t)
            outs= preds[:, i, :].permute(1,0)
            total = torch.mm(input_x[:, i, :], outs.to(device)) # e.g. size 8*8
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch_size*self.seq_length
        return nce


def cross_domain_train(da_params, config, model, my_dataset, src_id, tgt_id, run_id):
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id],batch_size=da_params['batch_size'])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id],batch_size=da_params['batch_size'])
    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'/home/emad/Mohamed2/Mohamed/trained_models/single_domain/pretrained_{config["model_name"]}_{src_id}_new.pt')
    # pretrained source model
    source_model = model(14, 32, 5, 0.5, True, device).to(device)
    wandb.watch(source_model)
    print('=' * 89)
    print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.encoder

    # initialize target model
    # target_model = copy.deepcopy(source_model)
    target_model = model(14, 32, 5, 0.5, True, device).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    target_encoder = target_model.encoder
    target_encoder.train()
    # discriminator network
    discriminator = dicriminator()
    comput_ncl= NCL_model().to(device)
    
    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=da_params['lr'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=da_params['lr'], betas=(0.5, 0.9))
    ncl_optim = torch.optim.AdamW(comput_ncl.parameters(), lr=0.01, betas=(0.5, 0.5))
    if da_params['tensor_board']:
        comment = (f'/home/emad/Mohamed2/visualize/Scenario={src_id} to {tgt_id}')
        tb = SummaryWriter(comment)
    for epoch in range(1, da_params['epochs'][src_id] + 1):
        batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
        total_loss = 0
        total_accuracy = 0
        alpha =#0.2
        beta = 0.2 
        target_losses, vat_losses, ncl = 0, 0, 0
        start_time = time.time()
        for _ in range(da_params['iterations']):  # , leave=False):
            # Train discriminator
            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(da_params['k_disc']):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                # source_features, (h, c) = source_encoder(source_x)
                # source_features = source_features[:, -1:].squeeze()
                _, source_features = source_model(source_x)
                # source_features = source_encoder(source_x).view(source_x.shape[0], -1)
                # target_features, (h, c) = target_encoder(target_x)
                # target_features = target_features[:, -1:].squeeze()
                _, target_features = target_model(target_x)


                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])
                preds = discriminator(discriminator_x).squeeze()
                loss = dis_critierion(preds, discriminator_y)
                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()
                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
            # Train predictor
            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(da_params['k_clf']):
                target_optim.zero_grad()
                ncl_optim.zero_grad()
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                # target_features, (h, c) = target_encoder(target_x)
                # target_features = target_features[:, -1:].squeeze()
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                # LDS should be calculated before the forward pass
                lds = vat_loss(target_model, target_x)
                
                _, target_features = target_model(target_x)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)
                preds = discriminator(target_features).squeeze()
                target_loss = dis_critierion(preds, discriminator_y)
                ncl_loss= comput_ncl(target_features,target_x)
                loss = target_loss + alpha * lds + beta*ncl_loss
                loss.backward()
                target_optim.step()
                ncl_optim.step()
                target_losses += target_loss.item()
                vat_losses +=  lds.item()
                ncl += ncl_loss.item()
        mean_loss = total_loss / (da_params['iterations'] * da_params['k_disc'])
        mean_accuracy = total_accuracy / (da_params['iterations'] * da_params['k_disc'])
        mean_ncl = ncl / (da_params['iterations'] * da_params['k_clf'])
        mean_tgt_loss = target_losses / (da_params['iterations'] * da_params['k_clf'])
        mean_vat_loss = vat_losses / (da_params['iterations'] * da_params['k_clf'])
        # tensorboard logging
#        if da_params['tensor_board']:
#            tb.add_scalar('Discriminator_loss', mean_loss, epoch)
#            tb.add_scalar('Discriminator_accuracy', mean_accuracy, epoch)
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy{mean_accuracy}')
        print(f'target_loss:{mean_tgt_loss} \t VATLoss{mean_vat_loss} \t NCL_loss{mean_ncl}')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, config)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, config)
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            if da_params['tensor_board_epoch']:
              tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
              tb.add_scalar('Loss/DA', test_loss, epoch)
              tb.add_scalar('Score/Src_Only', src_only_score, epoch)
              tb.add_scalar('Score/DA', test_score, epoch)
            
    src_only_loss, src_only_score, _, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, config)
    test_loss, test_score, _, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, config)
    pred_labels = sorted(pred_labels, reverse=True)
    true_labels = sorted(true_labels, reverse=True)
    pred_labels_DA = sorted(pred_labels_DA, reverse=True)
    true_labels_DA = sorted(true_labels_DA, reverse=True)

    fig1 = plt.figure()
    plt.plot(pred_labels, label='pred labels')
    plt.plot(true_labels, label='true labels')
    plt.legend()
    fig2 = plt.figure()
    plt.plot(pred_labels_DA, label='pred labels')
    plt.plot(true_labels_DA, label='true labels')
    plt.legend()
    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    # print the true and predicted labels
    if da_params['tensor_board']:
        tb.add_figure('Src_Only', fig1)
        tb.add_figure('DA', fig2)
        tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
        tb.add_scalar('Loss/DA', test_loss, epoch)
        tb.add_scalar('Score/Src_Only', src_only_score, epoch)
        tb.add_scalar('Score/DA', test_score, epoch)
        if da_params['tsne']:
            _, _, src_features, _, _, _ = evaluate(source_model, src_train_dl, criterion, config)
            _, _, tgt_features, _, _, _ = evaluate(source_model, tgt_train_dl, criterion, config)
            _, _, tgt_trained_features, _, _, _ = evaluate(target_model, tgt_train_dl, criterion, config)
            tb.add_embedding(src_features)
            tb.add_embedding(tgt_features)
            tb.add_embedding(tgt_trained_features)
    if da_params['save']:
        torch.save(target_model.state_dict(), f'/home/emad/Mohamed/trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_new.pt')
    return src_only_loss, src_only_score, test_loss, test_score
