import torch

class Learner:
    """Class to fit a two-layer neural network.
    """
    
    HEADER = "|  train loss  |  valid loss  | valid metric |"
    HEADER_COL_LENGTH = len("  train loss  ")
    
    PADDING = "-" * len(HEADER)
    
    def __init__(self, training_dataset, validation_dataset, model, opt_func=None, loss_func=None, metrics=None, opt_requires_model=False):
        """Creates object.
        
        dataloaders = DataLoader containing training set DataLoader [0] and validation set DataLoader [1]
        model = PyTorch module that returns outcome values
        
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        
        self.model = model
        
        if opt_func is None:
            opt_func = torch.optim.SGD
            
        self.optimizer_class = opt_func
        
        self.opt_requires_model = opt_requires_model
        
        if loss_func is None:
            loss_func = torch.nn.NLLLoss()
            
        self.loss_function = loss_func
        
        if metrics is None:
            metrics = lambda xb, yb: ((xb.sigmoid() > 0.5) == yb).float().mean()
            
        self.metric_function = metrics
        
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
        self.epoch_metrics = []


    def _calculate_gradient(self, x_batch, y_batch):
        predictions = self.model(x_batch)
        loss = self.loss_function(predictions, y_batch)
        loss.backward()
        
        return loss


    def _calculate_validation_loss_metric(self, x_batch, y_batch):
        predictions = self.model(x_batch)
        loss = self.loss_function(predictions, y_batch)
        
        metric = self.metric_function(predictions, y_batch)
        
        return loss.item(), metric.item()


    def _train_epoch(self):
        train_losses = 0
        
        for x_batch, y_batch in self.training_dataset:
            loss = self._calculate_gradient(x_batch, y_batch)
            train_losses += loss.item()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_train_loss = round(train_losses / len(self.training_dataset), 4)
        self.epoch_training_loss.append(epoch_train_loss)


    def _validate_epoch(self):
        num_batches = len(self.validation_dataset)
        
        val_losses = 0
        val_metrics = 0
        
        for x_batch, y_batch in self.validation_dataset:
            loss, metric = self._calculate_validation_loss_metric(x_batch, y_batch)
            
            val_losses += loss
            val_metrics += metric
        
        epoch_val_loss = round(val_losses / num_batches, 4)
        epoch_metric = round(val_metrics / num_batches, 4)
        
        self.epoch_validation_loss.append(epoch_val_loss)
        self.epoch_metrics.append(epoch_metric)

    
    def fit(self, epochs, lr=0.1):
        """Fits network parameters to best fit inputs and outputs over epochs loops using lr as learning rate.
        """
        optimizer_input = self.model if self.opt_requires_model else self.model.parameters()
        self.optimizer = self.optimizer_class(optimizer_input, lr)
        
        print(self.PADDING)
        print(self.HEADER)
        print(self.PADDING)
        
        for i in range(epochs):
            self._train_epoch()
            self._validate_epoch()
            
            print("|" + "|".join(
                [
                    f"{value:0.4f}".center(self.HEADER_COL_LENGTH) 
                    for value in [
                        self.epoch_training_loss[-1], self.epoch_validation_loss[-1], self.epoch_metrics[-1]
                    ]
                ]
            ) + "|")
        
        print(self.PADDING)