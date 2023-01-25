import torch
from .saver import SaveBestModel
import numpy as np
from ..center_net.metrics import AP
from ..center_net.keypoints import decode, filter_detections, rescale_detection


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
        
        
    def evaluate(self, dataloader: torch.utils.data.DataLoader,
                 score: float = 0.1, K: int = 70, num_dets: int = 1000, n: int = 5) -> np.array:
        """Compute the loss for a given dataloader.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
            score (float): Minimum score to consider a detection. Defaults to 0.1.
            K (int, optional): Number of centers. Defaults to 70.
            num_dets (int, optional): Number of detections. Defaults to 1000.
            n (int, optional): Odd number 3 or 5. Determines the scale of the central region. Defaults to 5.
             
        Returns:
            np.array: Losses for each loss function. [loss, focal_loss, push_loss, pull_loss, reg_loss]
        """
        
        num_batches = len(dataloader)
        metric_ap = AP(min_score=score)
        
        self.model.eval()
        
        test_losses = np.array([0, 0, 0, 0, 0], dtype=np.float64)

        ap = 0
        fd = 0
        size = 0
        
        with torch.no_grad():
            for batch in dataloader:
                self._batch_to_device(batch)
                outputs = self.model(batch['image'])
                
                losses = self.loss(outputs, batch, True)
                losses = np.array([l.item() for l in losses])
                
                test_losses += losses 

                detections, centers = decode(*outputs[0], K, 3, 0.5, num_dets=num_dets)
                detections = detections.to('cpu')
                centers = centers.to('cpu')
                
                y = batch['bbox'].numpy()
                
                for i in range(len(batch['image'])):
                    det = filter_detections(detections[i], centers[i], n=n)

                    if len(det) == 0:
                        continue
                    
                    det = rescale_detection(det)
                    
                    ap_l, fd_l = metric_ap.calculate(y[i], det)
                    ap += ap_l
                    fd += fd_l
                
                size += len(batch['image'])
                
        return test_losses / num_batches, ap / size, fd / size
        
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
              test_loader: torch.utils.data.DataLoader,
              epochs: int = 10,
              keep_best: bool = False,
              path_best_model: str = "best_model.pth"
              ) -> list:
        """Train the model for a given number of epochs.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
            train_loader (torch.utils.data.DataLoader): Dataloader for the test set.
            epochs (int, optional): Number of epochs. Defaults to 10.
            keep_best (bool, optional): Keep the best model. Defaults to False.
            path_best_model (str, optional): Path to save the best model. Defaults to "best_model.pth".
            
        Returns:
            list: Train loss
        """
        train_losses = [0] * epochs
        test_losses = [0] * epochs
        
        save_model = SaveBestModel(path_best_model, float('inf'), lambda b, v: v < b)
        
        for e in range(epochs):
            self._train(train_loader)

            tr_losses, tr_ap, tr_fd = self.evaluate(train_loader)
            te_losses, te_ap, te_fd = self.evaluate(test_loader)
            
            train_losses[e] = tr_losses[0]
            test_losses[e] = te_losses[0]
           
            saved = ""
            if keep_best:
                saved = "---> Saved" if save_model(self.model, te_fd) else ""
                
            print(f"Epoch {e + 1}/{epochs}, Train loss: {tr_losses[0]:.4f}, "
                  f"Train AP: {tr_ap:.4f}, Train FD: {tr_fd:.4f} | "
                  f"Test loss: {te_losses[0]:.4f}, "
                  f"Test AP: {te_ap:.4f}, Test FD: {te_fd:.4f} {saved}\n")
            
        if keep_best:
            self.model.load_state_dict(torch.load(path_best_model))
            self.model = self.model.to(self.device)

        return train_losses
 