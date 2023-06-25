from autoencoder import AutoEncoder
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 実行文
def main():
    # 訓練データ
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download = True)

    # データ
    data=train_dataset
    # AutoEncoder
    autoencoder=AutoEncoder(zd=64)
    # 学習
    autoencoder.update(data, mode=True)

    for n in range(10):
        x,y=train_dataset[n]

        x_hat=autoencoder.generated(x)

        x_hat=x_hat.to('cpu').detach()

        plt.imshow(x_hat.view(-1,28), cmap='gray')

        plt.savefig(f'./figs/{y}_{n}.png')

if __name__=='__main__':
    main()