import torch
from typing import Tuple
import tqdm
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 device: str) -> None:
        """Initialize the trainer.

        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            loss (torch.nn.Module): Loss function to use.
            device (str): Device to use (cuda or cpu).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def _train(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        """
        
        self.model.train()
        
        for _, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss(pred, y)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _test(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Compute the loss and the accurary for a given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to use.
        """
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= num_batches
        correct /= size
        
        return test_loss, correct
    
    def save(self, path: str) -> None:
        """Save the model.

        Args:
            path (str): Path to save the model.
        """
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}.")
   
    def training_process(self,
                         train_loader: torch.utils.data.DataLoader,
                         test_loader: torch.utils.data.DataLoader,
                         epochs: int = 10) -> Tuple[list, list, list, list]:
        """Train the model for a given number of epochs.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
            test_loader (torch.utils.data.DataLoader): Dataloader for the test set.
            epochs (int, optional): Number of epochs. Defaults to 10.
            
        Returns:
            Tuple[list, list, list, list]: Train loss, train accuracy, test loss, test accuracy.
        """
        
        train_losses = [0] * epochs
        train_accuracies = [0] * epochs
        test_losses = [0] * epochs
        test_accuracies = [0] * epochs
         
        with tqdm.tqdm(range(epochs), desc="Epoch") as t:
            for e in t:
                self._train(train_loader)
                train_loss, train_accuracy = self._test(train_loader)
                test_loss, test_accuracy = self._test(test_loader)
                
                train_losses[e] = train_loss
                train_accuracies[e] = train_accuracy
                test_losses[e] = test_loss
                test_accuracies[e] = test_accuracy
                
                t.set_postfix(train_loss=train_loss,
                              train_accuracy=train_accuracy,
                              test_loss=test_loss,
                              test_accuracy=test_accuracy)

        return train_losses, train_accuracies, test_losses, test_accuracies

class Ploting:
    @staticmethod
    def plot_loss(train_losses: list, test_losses: list) -> None:
        """Plot the loss.

        Args:
            train_losses (list): Train loss.
            test_losses (list): Test loss.
        """
        
        plt.plot(train_losses, label="Train loss")
        plt.plot(test_losses, label="Test loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(train_accuracies: list, test_accuracies: list) -> None:
        """Plot the accuracy.

        Args:
            train_accuracies (list): Train accuracy.
            test_accuracies (list): Test accuracy.
        """

        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_loss_accuracy(train_losses: list, train_accuracies: list,
                           test_losses: list, test_accuracies: list) -> None:
        """Plot the loss and the accuracy.
        
        Args:
            train_losses (list): Train loss.
            train_accuracies (list): Train accuracy.
            test_losses (list): Test loss.
            test_accuracies (list): Test accuracy.
        """

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(train_losses, color=color, label="Train loss")
        ax1.plot(test_losses, color=color, linestyle='dashed', label="Test loss")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc="upper left")

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(train_accuracies, color=color, label="Train accuracy")
        ax2.plot(test_accuracies, color=color, linestyle='dashed', label="Test accuracy")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc="upper right")
        
        fig.tight_layout()
        plt.show()