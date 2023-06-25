import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

"""AutoEncoder"""
class Model(nn.Module):
    def __init__(self, zd):
        super(Model, self).__init__()
        # Encoder
        self.__encoder=Encoder(zd)
        # Decoder
        self.__decoder=Decoder(zd)

    def forward(self, x):
        # データを潜在変数空間へ圧縮
        __z=self.__encoder(x)
        # 潜在変数空間からデータを復元
        __xhat=self.__decoder(__z)
        return __xhat
