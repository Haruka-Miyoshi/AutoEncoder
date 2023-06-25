from autoencoder import AutoEncoder
import torchvision
import torchvision.transforms as transforms

def main():
    # 検証データ
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download = True)

    # データ
    data=test_dataset
    # AutoEncoder
    autoencoder=AutoEncoder(zd=64, mode=True, model_path='./model/parameter.pth')
    # 検証データで損失計算
    autoencoder.test_loss(data, mode=True)

# おそらくだが，損失は評価指標にはならない -> 潜在変数空間を見て評価した方が良い．
if __name__=='__main__':
    main()