import torch 

# Optimizer Classes and Functions

class BasicOptimizer:
    """Basic Optimizer that can update parameters using gradients."""
    
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        """Applies gradients to parameters."""
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad.data
    
    def zero_grad(self):
        """Zeroes out gradients after applying step."""
        for p in self.parameters:
            p.grad = None


# Loss and Functions
def binary_cross_entropy_loss(predictions, targets):
    """Calculates cross entropy loss for two classes."""
    loss = (((1 - predictions) ** targets) * (predictions ** (1 - targets))).mean()
    return loss

def categorical_cross_entropy_loss(predictions, targets):
    """Calculates cross entropy loss for multiple classes."""
    loss = -(targets * predictions.log()).sum(axis=1).mean()
    return loss

# Accuracy Functions
def batch_accuracy(predictions, targets):
    """Calculates prediction accuracy."""
    accuracy = ((predictions > 0.5) == targets).float().mean()
    return accuracy

def multiclass_accuracy(predictions, targets):
    """Calculates multiclass prediction accuracy."""
    accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).float().mean()
    return accuracy

# Data Load Functions
def convert_image_data_to_row(img):
    """Converts 2-D tensor of image to 1-D long tensor."""
    img_dimensions = img.shape[-2], img.shape[-1]
    
    return img.view(-1, img_dimensions[0] * img_dimensions[1]).squeeze()

def create_image_dataset_mask(dataset, mask_vals):
    """Creates mask of labels in a dataset to keep."""
    mask = dataset.targets != dataset.targets
    
    for mask_val in mask_vals:
        mask |= dataset.targets == mask_val
    
    return mask
    
def dataset_load_label_mask(label):
    if label == 3:
        return torch.tensor([1])
    elif label == 7:
        return torch.tensor([0])
    else: 
        return torch.tensor([-1])