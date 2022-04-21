import torch
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torchvision.transforms as T
from mnist_learner import Learner

import utils

from pathlib import Path

if __name__ == '__main__':
    train_mnist_images = MNIST(
        Path('assets/datasets'), train=True, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(utils.convert_image_data_to_row)
            ]
        ),
        target_transform=T.Lambda(lambda x: F.one_hot(torch.tensor(x), num_classes=10).squeeze())
    )
    
    valid_mnist_images = MNIST(
        Path('assets/datasets'), train=False, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(utils.convert_image_data_to_row)
            ]
        ),
        target_transform=T.Lambda(lambda x: F.one_hot(torch.tensor(x), num_classes=10).squeeze())
    )
    
    train_dataset = torch.utils.data.DataLoader(train_mnist_images, batch_size=256, shuffle=True)
    valid_dataset = torch.utils.data.DataLoader(valid_mnist_images, batch_size=256, shuffle=True)

    model = torch.nn.Sequential(
                torch.nn.Linear(28*28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10),
                torch.nn.Softmax()
    )
    
    learn = Learner(train_dataset, valid_dataset, model, opt_func=torch.optim.SGD, loss_func=utils.categorical_cross_entropy_loss, metrics=utils.multiclass_accuracy)
    
    learn.fit(10, lr=1)
