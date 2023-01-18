import torch
from .saver import SaveBestModel


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
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.device = device
        
    def _batch_to_device(self, batch):
        batch['image'] = batch['image'].to(self.device)
        
        batch['inds_tl'] = batch['inds_tl'].to(self.device)
        batch['inds_br'] = batch['inds_br'].to(self.device)
        batch['inds_ct'] = batch['inds_ct'].to(self.device)
        batch['hmap_tl'] = batch['hmap_tl'].to(self.device)
        batch['hmap_br'] = batch['hmap_br'].to(self.device)
        batch['hmap_ct'] = batch['hmap_ct'].to(self.device)
        batch['regs_tl'] = batch['regs_tl'].to(self.device)
        batch['regs_br'] = batch['regs_br'].to(self.device)
        batch['regs_ct'] = batch['regs_ct'].to(self.device)
        batch['ind_masks'] = batch['ind_masks'].to(self.device)
        
        
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Compute the loss for a given dataloader.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        
        Returns:
            float: Loss of the dataloader.
        """
        
        num_batches = len(dataloader)
        self.model.eval()
        
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                self._batch_to_device(batch)
                outputs = self.model(batch['image'])
                
                test_loss += self.loss(outputs, batch).item()
                
        test_loss /= num_batches

        return test_loss
        
    def _train(self, dataloader: torch.utils.data.DataLoader):
        """Train the model for one epoch.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        """
        
        self.model.train()
        
        for batch in dataloader:
            self._batch_to_device(batch)
            outputs = self.model(batch['image'])
            
            loss = self.loss(outputs, batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
    def save(self, path: str) -> None:
        """Save the model.
        Args:
            path (str): Path to save the model.
        """
        
        torch.save(self.model.to('cpu').state_dict(), path)
        print(f"Model saved to {path}.")
        self.model.to(self.device)
   
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
        
        save_model = SaveBestModel(path_best_model, float('inf'), lambda b, v: v < b)
        
        for e in range(epochs):
            self._train(train_loader)
           
            train_loss = self.evaluate(train_loader)
            train_losses[e] = train_loss
            
            saved = ""
            if keep_best:
                saved = "---> Saved" if save_model(self.model, train_loss) else ""
            
            print(f"Epoch {e + 1}/{epochs}, Train loss: {train_loss:.4f} {saved}")
                
        if keep_best:
            self.model.load_state_dict(torch.load(path_best_model))
            self.model = self.model.to(self.device)

        return train_losses
 