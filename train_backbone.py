import argparse
from src.backbone import BackboneNetwork, get_MNIST
from src.training import Trainer

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=bool, default=True,
                        help="Whether to save the model or not")
    parser.add_argument("--name", type=str, default="backbone_model.pth",
                        help="Name of the model to save")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataloader, test_dataloader = get_MNIST(batch_size=64)
    
    model = BackboneNetwork().to(device)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, optimizer, loss, device)
    vals = trainer.train(train_dataloader, test_dataloader, epochs=20)

    if args.save:
        trainer.save(f"models/{args.name}")
    