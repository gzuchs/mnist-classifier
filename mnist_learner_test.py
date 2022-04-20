import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T
from mnist_learner import Learner

from pathlib import Path

# Train Loop Functions

class BasicOptimizer:
    """Basic Optimizer that can update parameters using gradients."""
    
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
        
    def step(self):
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad.data
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    
    # loss = (((1 - predictions) ** targets) * (predictions ** (1 - targets))).mean()
    loss = torch.where(targets == 1, 1 - predictions, predictions).mean()
    return loss
    

def batch_accuracy(predictions, targets):
    predictions = predictions.sigmoid()
    
    accuracy = ((predictions > 0.5) == targets).float().mean()
    
    return accuracy


# Data Loading Functions

def dataset_load_label_mask(label):
    if label == 3:
        return torch.tensor([1])
    elif label == 7:
        return torch.tensor([0])
    else: 
        return torch.tensor([-1])


def create_image_dataset_mask(dataset, mask_vals):
    mask = dataset.targets != dataset.targets
    
    for mask_val in mask_vals:
        mask |= dataset.targets == mask_val
        
    return mask


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    
    
    
    train_mnist_images = MNIST(
        Path('Datasets'), train=True, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(lambda x: x.flatten(start_dim=-2).squeeze())
            ]
        ),
        target_transform=dataset_load_label_mask
    )
    
    valid_mnist_images = MNIST(
        Path('Datasets'), train=False, 
        transform=T.Compose(
            [
                T.ToTensor(), T.Lambda(lambda x: x.view(-1, 28*28).squeeze())
            ]
        ),
        target_transform=dataset_load_label_mask
    )

    mask_vals = [3, 7]

    train_mnist_images.data, train_mnist_images.targets = train_mnist_images.data[create_image_dataset_mask(train_mnist_images, mask_vals)], train_mnist_images.targets[create_image_dataset_mask(train_mnist_images, mask_vals)]

    valid_mnist_images.data, valid_mnist_images.targets = valid_mnist_images.data[create_image_dataset_mask(valid_mnist_images, mask_vals)], valid_mnist_images.targets[create_image_dataset_mask(valid_mnist_images, mask_vals)]
    
    train_dataset = torch.utils.data.DataLoader(train_mnist_images, batch_size=256, shuffle=True)
    valid_dataset = torch.utils.data.DataLoader(valid_mnist_images, batch_size=256, shuffle=True)
    
    # learn = Learner(train_dataset, valid_dataset, torch.nn.Linear(28*28, 1), opt_func=BasicOptimizer, loss_func=mnist_loss, metrics=batch_accuracy)
    learn = Learner(train_dataset, valid_dataset, torch.nn.Linear(28*28, 1), opt_func=torch.optim.SGD, loss_func=mnist_loss, metrics=batch_accuracy)
    
    learn.fit(10, lr=1)
    
    