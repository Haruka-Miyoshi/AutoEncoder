import torch.nn as nn

"""Encoder"""
class Encoder(nn.Module):
    """コンストラクタ"""
    def __init__(self, zd):
        super(Encoder, self).__init__()
        # Encoder
        self.__encoder=nn.Sequential(
            # 784次元->256次元
            nn.Linear(28 * 28, 256),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True),
            # 256次元->128次元
            nn.Linear(256, 128),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True),
            # 128次元->zd:潜在変数空間の次元数
            nn.Linear(128, zd)
        )
    
    """順伝播"""
    def forward(self,x):
        # 潜在変数空間へ圧縮
        __z=self.__encoder(x)
        return __z