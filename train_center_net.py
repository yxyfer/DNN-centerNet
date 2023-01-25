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
    parser.add_argument("--keep_best", type=bool, default=True,
                        help="Keep the best model. Default: True")
    parser.add_argument("--max_objects", type=int, default=30,
                        help="Maximum number of objects in an image. Default: 30")
    parser.add_argument("--max_images_train", type=int, default=700,
                        help="Maximum number of images to use for training. Default: 700")
    parser.add_argument("--max_images_test", type=int, default=200,
                        help="Maximum number of images to use for testing. Default: 200")
    parser.add_argument("--pretrained_backbone", type=str, default="models/backbone_model.pth",
                        help="Path to the pretrained backbone model, if 'none' do not load anything. Default: models/backbone_model.pth")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_train = MnistDetection(args.dataset, train=True, max_objects=args.max_objects,
                                   max_images=args.max_images_train)
    dataset_test = MnistDetection(args.dataset, train=False, max_objects=args.max_objects,
                                  max_images=args.max_images_test)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=True) 
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                              shuffle=True) 
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CenterNet(args.pretrained_backbone)
    optimizer = torch.optim.Adam(model.parameters())
    loss = CenterNetLoss()

    trainer = TrainerCenterNet(model, optimizer, loss, device)

    print("Training CenterNet model...")
    vals = trainer.train(train_loader, test_loader, epochs=args.epochs, keep_best=args.keep_best)
    
    trainer.save(f"models/{args.name}")
