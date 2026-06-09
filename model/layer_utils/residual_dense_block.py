import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class RDB(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(RDB,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =growth_rate,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x

class BasicBlock_Lightweight(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock_Lightweight,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim, out_channels = output_dim,kernel_size=3,padding=2,stride=1,dilation=2, groups=32)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class RDB_Lightweight(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(RDB_Lightweight,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =growth_rate,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock_Lightweight(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x