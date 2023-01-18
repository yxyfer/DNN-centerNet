import argparse
from src.dataset import MnistDetection
from src.center_net import CenterNetLoss
from src.center_net.network import CenterNet
from src.training import TrainerCenterNet


import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="center_net_model.pth",
                        help="Name of the model to save. Default: center_net_model.pth")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train the model. Default: 20")
    parser.add_argument("--dataset", type=str, default="data/mnist_detection",
                        help="Path to the dataset. Default: data/mnist_detection")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size. Default: 8")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset = MnistDetection(args.dataset, train=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CenterNet()
    optimizer = torch.optim.Adam(model.parameters())
    loss = CenterNetLoss()

    trainer = TrainerCenterNet(model, optimizer, loss, device)

    print("Training CenterNet model...")
    vals = trainer.train(train_loader, epochs=args.epochs)
    
    trainer.save(f"models/{args.name}")