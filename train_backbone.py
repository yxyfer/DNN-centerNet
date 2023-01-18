import argparse
from src.backbone import BackboneNetwork, get_MNIST
from src.training import TrainerBackbone

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="backbone_model.pth",
                        help="Name of the model to save. Default backbone_model.pth")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataloader, test_dataloader = get_MNIST(batch_size=64)
    
    model = BackboneNetwork().to(device)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = TrainerBackbone(model, optimizer, loss, device)

    print("Training Backbone model...")    
    vals = trainer.train(train_dataloader, test_dataloader, epochs=20)

    trainer.save(f"models/{args.name}")
    