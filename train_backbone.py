import argparse
from src.backbone import BackboneNetwork, get_MNIST
from src.training import TrainerBackbone

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="backbone_model.pth",
                        help="Name of the model to save. Default backbone_model.pth")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train the model. Default: 20")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size. Default: 64")
    parser.add_argument("--keep_best", type=bool, default=True,
                        help="Keep the best model. Default: True")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataloader, test_dataloader = get_MNIST(batch_size=args.batch_size)
    
    model = BackboneNetwork().to(device)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = TrainerBackbone(model, optimizer, loss, device)

    print("Training Backbone model...")    
    vals = trainer.train(train_dataloader, test_dataloader, epochs=args.epochs,
                         keep_best=args.keep_best)

    trainer.save(f"models/{args.name}")
    