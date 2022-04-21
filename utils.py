
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

# Accuracy Functions

# Data Load Functions
def convert_image_data_to_row(img):
    """Converts 2-D tensor of image to 1-D long tensor."""
    img_dimensions = img.shape[-2], img.shape[-1]
    
    return img.view(-1, img_dimensions[0] * img_dimensions[1]).squeeze()