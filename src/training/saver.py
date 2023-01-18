import torch

class SaveBestModel:
    def __init__(self, path: str, best_value: float, compare) -> None:
        """Initialize the best model saver.
        Args:
            path (str): Path to save the model.
            best_value (float, optional): Current best value to beat.
            compare (function): Function to compare the current value with the best value.
            This function will be called as compare(best_value, value). It should return a bool.
        """
        
        self.path = path
        self.best_value = best_value
        self.compare = compare
        
    def __call__(self, model: torch.nn.Module, value: float) -> bool:
        """Save the model if the loss is lower than the best loss.
        Args:
            model (torch.nn.Module): Model to save.
            value (float): Current value to compare with the best value.
            
        Returns:
            bool: True if the model is saved, False otherwise.
        """
        
        if self.compare(self.best_value, value):
            self.best_value = value
            torch.save(model.state_dict(), self.path)
            
            return True
        
        return False
