import torch
# from typing import Tuple
# from .saver import SaveBestModel


class TrainerCenterNet:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 device: str,
                 scheduler: torch.optim.lr_scheduler.StepLR = None):
        """Initialize the trainer.
        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            loss (torch.nn.Module): Loss function to use.
            device (str): Device to use (cuda or cpu).
            scheduler (torch.optim.lr_scheduler.StepLR): Scheduler to use, to reduce the learning rate. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.device = device
        
        
    def _train(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train the model for one epoch.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        
        Returns:
            float: Loss of the epoch.
        """
        
        self.model.train()
        
        for batch in dataloader:
            outputs = self.model(batch['image'])
            
            loss = self.loss(outputs, batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        return loss.item()

    def save(self, path: str) -> None:
        """Save the model.
        Args:
            path (str): Path to save the model.
        """
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}.")
   
    def train(self,
              train_loader: torch.utils.data.DataLoader,
              epochs: int = 10,
              keep_best: bool = False,
              path_best_model: str = "best_model.pth"
              ) -> list:
        """Train the model for a given number of epochs.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
            epochs (int, optional): Number of epochs. Defaults to 10.
            keep_best (bool, optional): Keep the best model. Defaults to False.
            path_best_model (str, optional): Path to save the best model. Defaults to "best_model.pth".
            
        Returns:
            list: Train loss
        """
        train_losses = [0] * epochs
        
        for e in range(epochs):
            train_loss = self._train(train_loader)
            
            train_losses[e] = train_loss
            
            print(f"Epoch {e + 1}/{epochs}, Train loss: {train_loss:.4f}")

        return train_losses
            