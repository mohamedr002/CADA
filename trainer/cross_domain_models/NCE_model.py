import torch
from torch import nn

    
class NCE_model(nn.Module): 
    def __init__(self,device, input_dim=64, seq_length=30, out_dim=14):
      super(NCE_model,self).__init__()
      self.input_dim=input_dim
      self.seq_length= seq_length 
      self.out_dim = out_dim
      self.softmax  = nn.Softmax()
      self.lsoftmax = nn.LogSoftmax()
      self.device = device
      self.base_model= nn.Linear(self.input_dim,self.out_dim) 
      self.nce_model =  nn.ModuleList([self.base_model for i in range(self.seq_length)])
    def forward (self, f_t, input_x):
        nce = 0 
        batch_size= input_x.size(0)
        # input_f--> output features of feature extractor #dim (batch_size, feature_dim)
        # input_x--> original input the feature extractor # dim (batch_size,seq_length,input_dim)
        # preds is output after applying module_list on the features f: from batch_size, feature_dim)-->(batch_size,seq_length,out_dim)
        preds = torch.empty((batch_size, self.seq_length,self.out_dim)).float()
        for i in range  (self.seq_length):
            preds[:, i, :] = self.nce_model[i](f_t)
            outs= preds[:, i, :].permute(1,0)
            total = torch.mm(input_x[:, i, :], outs.to(self.device)) # e.g. size 8*8
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch_size*self.seq_length
        return nce
        
def dicriminator():
    discriminator = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return discriminator