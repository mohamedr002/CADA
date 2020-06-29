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
import time
from torch.utils.tensorboard import SummaryWriter
from trainer.cross_domain_models.NCE_model import NCE_model,dicriminator
import wandb


def cross_domain_train(params,device, config, model, my_dataset, src_id, tgt_id, run_id):
    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id],batch_size=hyper['batch_size'])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id],batch_size=hyper['batch_size'])
    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'/home/emad/Mohamed2/Mohamed/trained_models/single_domain/pretrained_{config["model_name"]}_{src_id}_new.pt')
    # pretrained source model
    source_model = model(14, 32, 5, 0.5, True, device).to(device)
    if config['tensorboard']:
      wandb.watch(source_model, log='all')
    print('=' * 89)
    print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.encoder

    # initialize target model
    target_model = model(14, 32, 5, 0.5, True, device).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    target_encoder = target_model.encoder
    target_encoder.train()
    # discriminator network
    discriminator = dicriminator().to(device)
    comput_nce= NCE_model(device).to(device)
    
    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['lr'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hyper['lr'], betas=(0.5, 0.9))
    nce_optim = torch.optim.AdamW(comput_nce.parameters(), lr=hyper['nce_lr'], betas=(0.5, 0.9))
    if config['tensorboard']:
        comment = (f'/home/emad/Mohamed2/visualize/Scenario={src_id} to {tgt_id}')
        tb = SummaryWriter(comment)
    for epoch in range(1, hyper['epochs'] + 1):
        batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
        total_loss = 0
        total_accuracy = 0
        alpha = hyper['alpha_nce'] 
        target_losses, nce = 0, 0
        start_time = time.time()
        for _ in range(config['iterations']):  # , leave=False):
            # Train discriminator
            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(config['k_disc']):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)
                _, source_features = source_model(source_x)
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
            # Train Feature Extractor
            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(config['k_clf']):
                target_optim.zero_grad()
                nce_optim.zero_grad()
                # Get a batch
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)              
                _, target_features = target_model(target_x)
                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)
                preds = discriminator(target_features).squeeze()
                target_loss = dis_critierion(preds, discriminator_y)
                # Negaative Contrastive Estimtion Loss
                nce_loss= comput_nce(target_features,target_x)
                #total loss
                loss = target_loss + alpha*nce_loss
                loss.backward()
                target_optim.step()
                nce_optim.step()
                target_losses += target_loss.item()
                nce += nce_loss.item()
        mean_loss = total_loss / (config['iterations'] * config['k_disc'])
        mean_accuracy = total_accuracy / (config['iterations'] * config['k_disc'])
        mean_nce = nce / (config['iterations'] * config['k_clf'])
        mean_tgt_loss = target_losses / (config['iterations'] * config['k_clf'])

        # tensorboard logging
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy{mean_accuracy}')
        print(f'target_loss:{mean_tgt_loss}  \t NCE_loss{mean_nce}')
        if epoch % 10 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, config,device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, config,device)
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            if config['tensorboard_epoch']:
              tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
              tb.add_scalar('Loss/DA', test_loss, epoch)
              tb.add_scalar('Score/Src_Only', src_only_score, epoch)
              tb.add_scalar('Score/DA', test_score, epoch)
            
    src_only_loss, src_only_score, _, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, config,device)
    test_loss, test_score, _, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, config,device)
    # Plot true vs pred labels
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
    if config['tensorboard']:
        tb.add_figure('Src_Only', fig1)
        tb.add_figure('DA', fig2)
        tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
        tb.add_scalar('Loss/DA', test_loss, epoch)
        tb.add_scalar('Score/Src_Only', src_only_score, epoch)
        tb.add_scalar('Score/DA', test_score, epoch)
        if config['tsne']:
            _, _, src_features, _, _, _ = evaluate(source_model, src_train_dl, criterion, config,device)
            _, _, tgt_features, _, _, _ = evaluate(source_model, tgt_train_dl, criterion, config,device)
            _, _, tgt_trained_features, _, _, _ = evaluate(target_model, tgt_train_dl, criterion, config,device)
            tb.add_embedding(src_features)
            tb.add_embedding(tgt_features)
            tb.add_embedding(tgt_trained_features)
    if config['save']:
        torch.save(target_model.state_dict(), f'/home/emad/Mohamed/trained_models/cross_domains/{src_id}_to_{tgt_id}_{run_id}_new.pt')
    return src_only_loss, src_only_score, test_loss, test_score
