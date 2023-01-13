from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple

def _load_data_MNIST(train: bool = True):
    return datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor()
    )
    
def get_MNIST(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Get the MNIST dataset.
    
    Args:
        batch_size (int, optional): Batch size. Defaults to 64.
        
    Returns:
        tuple[DataLoader, DataLoader]: Training and test dataloader.
    """
    
    training_data = _load_data_MNIST(train=True)
    test_data = _load_data_MNIST(train=False)
    
    return (DataLoader(training_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size, shuffle=True))