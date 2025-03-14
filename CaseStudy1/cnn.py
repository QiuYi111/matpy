import torch
from torch.nn.modules import activation
import torchvision
from datasets import load_from_disk
from torch import nn
class cnn(nn.Module):
    def __init__(self,kernal_size,dims,hidden_layers,nn_num,dataset_dir):
        super.__init__()
        self.datasets=load_from_disk(dataset_dir)
        self.hidden_layers=hidden_layers
        self.nn_num=nn_num
        self.kernal_size=kernal_size
        print("cnn initialized!")
    def build_nn(self):
        self._cnn=nn.Sequential(
            nn.Conv2d()
        )
        layers=[]
        for i in range(self.hidden_layers):
            layer=nn.Linear(self.dims[i],self.dims[i+1])
            activation=nn.ELU()
            layers.append(layer)
            layers.append(activation)
        layers.append(nn.Linear(self.dims[-1],self.label_num),nn.Softmax)

        self._layers=nn.Sequential(*layers)

    def forward(self):




