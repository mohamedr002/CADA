import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda')


def get_model_config(model_key,input_dim=14,hid_dim=32,n_layers=3,drop=0.2, bid=True):
    model_configs = {
        "CNN": {
            "model_name": "CNN",
            "input_dim": input_dim,
            "CLIP": 1,
            "permute": True,
            "out_dim": 32,
            "fc_drop": 0.5},
        "LSTM": {
            "model_name": 'LSTM',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "out_dim": input_dim,
            "n_layers": 3,
            "permute": False,
            "bid": True,
            "drop": 0.5,
            "CLIP": 1,
            "save":False,
            "fc_drop": 0.5
        },
        "Transformer": {
            "model_name": 'Transformer',
            "input_dim": 14,
            "hid_dim": 200,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "out_dim": 1,
            "n_layers": 2,
            "nhead" : 1,  # the number of heads in the multiheadattention models
            "dropout" : 0.2 , # the dropout value
            "permute": False,
            "bid": True,
            "drop": 0.5,
            "CLIP": 1,
            "save": False,
            "fc_drop": 0.5
        },
        "VRNN": {
            "model_name": 'VRNN',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "z_dim": 32,
            "permute":False,
            "n_layers": n_layers,
            "drop": 0.2,
            "CLIP": 1,
            "fc_drop": 0.2},
        "RESNET": {
            "model_name": 'RESNET',
            "input_dim": input_dim,
            "num_classes": 1,
            "permute": True
            },
        "TCN":{"model_name": "TCN",
         "input_channels": input_dim,
         "n_classes": 1,
         "permute":True,
         "num_channels": [8] * 4,
         "kernel_size": 16,
         "drop": 0.2
               },
        "seq2seq":{
            "model_name": 'seq2seq',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "out_dim": input_dim,
            "n_layers": 3,
            "permute": False,
            "bid": False,
            "drop": 0.5,
            "CLIP": 1,
            "fc_drop": 0.5}


    }
    return model_configs[model_key]
def initlize(config):
    if config["model_name"]=='LSTM':
        model = lstm(config['input_dim'], config['hid_dim'], config['n_layers'], config['drop'], config['bid'],
                 device).to(device)
    if config["model_name"]=='RESNET':
        model = resnet18(in_dim=config['input_dim'], num_classes=config['num_classes']).to(device)
    elif config["model_name"] == 'TCN':
        model = TCN(config['input_channels'], config['n_classes'], config['num_channels'], config['kernel_size'], config['drop']).to(device)
    elif config["model_name"] == 'CNN':
        model = CNN(config['input_dim'], config['out_dim'], config['fc_drop'], device).to(device)
    elif config["model_name"] == 'VRNN':
        model = VRNN_model(config['input_dim'], config['hid_dim'], config['z_dim'], config['n_layers'], config['drop'],device).to(device)
    elif config["model_name"] == 'seq2seq':
        model = seq2seq(config['input_dim'], config['hid_dim'], config['n_layers'], config['drop'], config['bid'],device).to(device)
    return model