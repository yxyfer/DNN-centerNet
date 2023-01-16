import torch

class SaveBestModel:
    def __init__(self, path: str, best_accuracy: float = 0) -> None:
        """Initialize the best model saver.
        Args:
            path (str): Path to save the model.
            best_accuracy (float, optional): Best accuracy. Defaults to 0.
        """
        
        self.path = path
        self.best_accuracy = best_accuracy
        
    def __call__(self, model: torch.nn.Module, accuracy: float) -> bool:
        """Save the model if the loss is lower than the best loss.
        Args:
            model (torch.nn.Module): Model to save.
            accuracy (float): Accuracy of the model.
            
        Returns:
            bool: True if the model is saved, False otherwise.
        """

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(model.state_dict(), self.path)
            
            return True
        
        return False