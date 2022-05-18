import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T
from mnist_learner import Learner

import utils

from pathlib import Path

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    
    
    
    train_mnist_images = MNIST(
        Path('assets/datasets'), train=True, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(utils.convert_image_data_to_row)
            ]
        ),
        target_transform=utils.dataset_load_label_mask
    )
    
    valid_mnist_images = MNIST(
        Path('assets/datasets'), train=False, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(utils.convert_image_data_to_row)
            ]
        ),
        target_transform=utils.dataset_load_label_mask
    )

    mask_vals = [3, 7]

    train_mnist_images.data, train_mnist_images.targets = train_mnist_images.data[utils.create_image_dataset_mask(train_mnist_images, mask_vals)], train_mnist_images.targets[utils.create_image_dataset_mask(train_mnist_images, mask_vals)]

    valid_mnist_images.data, valid_mnist_images.targets = valid_mnist_images.data[utils.create_image_dataset_mask(valid_mnist_images, mask_vals)], valid_mnist_images.targets[utils.create_image_dataset_mask(valid_mnist_images, mask_vals)]
    
    train_dataset = torch.utils.data.DataLoader(train_mnist_images, batch_size=256, shuffle=True)
    valid_dataset = torch.utils.data.DataLoader(valid_mnist_images, batch_size=256, shuffle=True)
    
    model = torch.nn.Sequential(
                torch.nn.Linear(28*28, 1),
                torch.nn.Sigmoid()
    )
    
    # learn = Learner(train_dataset, valid_dataset, model, opt_func=torch.optim.SGD, loss_func=utils.binary_cross_entropy_loss, metrics=utils.batch_accuracy)
    learn = Learner(train_dataset, valid_dataset, model, opt_func=utils.BasicOptimizer, loss_func=utils.binary_cross_entropy_loss, metrics=utils.batch_accuracy)
    
    learn.fit(10, lr=1)
    
    