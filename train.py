"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
from utils import data_setup, engine, model, utils

from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.1)

args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate  # lr is divided by 10 when the error plateaus

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms -> we will need to gather the transforms used in the inital paper
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model.ResNet18(input_shape=3,
                       output_shape=len(class_names)).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            momentum=0.9,
                            weight_decay=0.001)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="18_layer_resnet_model.pth")
