
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
device = torch.device('cuda:1')
import random
from torch.autograd import Variable
# from utils import *
""" CNN Model """


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            Flatten(),
            nn.Linear(8, self.hidden_dim)) 
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))    
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.regressor(features)
        return predictions, features
cnn_model=CNN_RUL(14,30,0.5)
cnn_model.encoder

""" LSTM Model """
class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
#         encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions, features
# model=LSTM_RUL(14, 32, 5, 0.5, True, device)
""" Resnet Model """
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out', out.size(), 'res', residual.size(), self.downsample)
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, in_dim, num_classes, arch):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1])  # , stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2])  # , stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3])  # , stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(256, num_classes)  # 512 * block.expansion
        self.arch = arch
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        predictions = self.fc(x)
        return predictions, x
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], arch='resnet18', **kwargs)
    return model
def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], arch='resnet34', **kwargs)
    return model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], arch='resnet50', **kwargs)
    return model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnet101', **kwargs)

    return model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], arch='resnet152', **kwargs)
    return model
# model = resnet18(in_dim=14, num_classes=1).to(device)


# # TCN

# In[5]:

""" TCN Model """
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TCN_RUL(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_RUL, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.hidden_dim= num_channels[-1]
        self.dropout=dropout
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))   
    def forward(self, x):
            outs = self.network(x)
            features=outs[:, :, -1]
            predictions =self.regressor(features)
            return predictions,features
# Model Definition
# model=TCN_RUL(14, [8] * 4, 16, 0.2)


# # VRNN

# In[6]:
""" VRNN Model """
class VRNN_RUL(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers,dropout, bias=False):
        super(VRNN, self).__init__()
        self.dropout=dropout
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        # self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())
        # recurrence
#         self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias, batch_first=True)
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, batch_first=True)

        # regressor network 
        self.regressor= nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.h_dim, self.h_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.h_dim//2, 1))    
            
    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).cuda()
        for t in range(x.size(1)):
            phi_x_t = self.phi_x(x[:, t])
            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # recurrence
#             set_trace()
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
        features= phi_z_t
        predictions= self.regressor(features)
        return predictions, phi_z_t

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass
    
    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)
# model definition 
# model = VRNN(14, 32, 32, 3, 0.5,device).to(device)


# # Attention Model 

# In[23]:

""" Transformer_LSTM Model """
class TransformerModel_LSTM(nn.Module):
    def __init__(self, out_dim, input_dim, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_LSTM, self).__init__()
        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
      
        #LSTM paramaters
        self.dropout=0
        self.bid=True
        self.hidden_dim= 32 
        self.n_layers = 1
        self.decoder = nn.LSTM(input_dim, self.hidden_dim,self.n_layers, dropout=self.dropout, batch_first=True, bidirectional=self.bid)
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1)) 
        
#         self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
#     def init_weights(self):
#         initrange = 0.1
#         # self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        attention_outputs = self.transformer_encoder(src, self.src_mask)
        decoder_outputs,(h,c) = self.decoder(attention_outputs)
        features = decoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions,features
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

""" Transformer_CNN Model """
class TransformerModel_CNN(nn.Module):
    def __init__(self, out_dim, input_dim, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_CNN, self).__init__()
        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
      
        #CNN paramaters
        self.hidden_dim = nhid
        self.input_dim = input_dim
        self.dropout=dropout
        self.decoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Flatten(),
            nn.Linear(8, self.hidden_dim)) 
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1)) 
        
#         self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        attention_outputs = self.transformer_encoder(src, self.src_mask)
        attention_outputs=attention_outputs.permute(0,2,1)
        features = self.decoder(attention_outputs)
        predictions = self.regressor(features)
        return predictions,features
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_dims,bid):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dims+hidden_dims*self.bid, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
            nn.LogSoftmax() )
    def forward(self, input):
        out = self.layer(input)
        return out

