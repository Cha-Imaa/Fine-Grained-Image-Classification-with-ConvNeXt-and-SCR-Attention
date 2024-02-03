import os
import logging
import pickle
import random
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

import wandb
wandb.init(project="CV703_task1_aug_color_drop_seb")

RESULTS="/lustre/scratch/users/rusiru.achchige/Results"
PATH_CUB_DATASET = "./data/CUB/CUB_200_2011"
PATH_AIRCRAFT_DATASET = "./data/fgvc-aircraft-2013b"
PATH_CHECKPOINT_FOLDER = f'{RESULTS}/Checkpoints/Task1_aug_color_drop_seb'
PATH_LOGGINGS_FOLDER = f'{RESULTS}/loggings/Task1_aug_color_drop_seb'
PATH_PICKLE_FOLDER = f'{RESULTS}/pickels/Task1_aug_color_drop_seb'

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PATH_CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(PATH_LOGGINGS_FOLDER, exist_ok=True)
os.makedirs(PATH_PICKLE_FOLDER, exist_ok=True)

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
    
# Define transformations
train_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize to a standard size
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.RandomRotation(15),       # Rotation
    transforms.ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25], saturation=[0.75, 1.25], hue=[-0.05, 0.05]), # Color jitter
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

test_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=train_data_transforms, split="train")
test_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=test_data_transforms, split="test")


# load in into the torch dataloader to get variable batch size, shuffle 
train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=128, drop_last=True, shuffle=True)
test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=128, drop_last=False, shuffle=False)

num_classes = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SCRAttention(nn.Module):
   def __init__(self, in_channels):
       super(SCRAttention, self).__init__()
       self.se_layer = nn.Conv2d(in_channels, 1, kernel_size=1)
       self.sigmoid = nn.Sigmoid()

   def forward(self, x):
       scale = torch.mean(x, dim=(2, 3), keepdim=True)
       scale = self.se_layer(scale)
       scale = self.sigmoid(scale)
       return x * scale
   
class CustomBlock(nn.Module):
   def __init__(self, original_blocks, attention_block):
       super(CustomBlock, self).__init__()
       # The first three layers of the original block
       self.first_blocks = nn.Sequential(*list(original_blocks)[:1]) 
       # SCRA block
       self.attention_block = attention_block
       # The remaining layers of the original block
       self.last_blocks = nn.Sequential(*list(original_blocks)[1:])

   def forward(self, x):
       x = self.first_blocks(x)
       x = self.attention_block(x)
       x = self.last_blocks(x)
       return x
   
model = timm.create_model('convnextv2_large', pretrained=True) 
model.head.fc = nn.Sequential(
    nn.Dropout(0.1), 
    nn.Linear(model.head.fc.in_features, num_classes)
    )
original_blocks = model.stages  
# The number of input channels to the SCRA block is 
# the same as the number of output channels from the last layer of the original block
in_channels = 192
model.stages = CustomBlock(original_blocks, SCRAttention(in_channels))

model = model.to(device)

# Loss Function, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train(model, concat_loader_train, criterion, optimizer, scheduler):
    model.train()  
    total_loss = 0
    correct = 0
    total = 0
    print_every = 225
    batch_count = 0

    # Loop over each batch from the training set
    for batch in concat_loader_train:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        batch_count += 1

        if batch_count % print_every == 0:
            current_accuracy = 100.0 * correct / total
            print(f'Epoch {epoch+1}, Batch {batch_count}, Current Loss: {loss.item()}, Current Accuracy: {current_accuracy}%')

    scheduler.step()

    return total_loss / len(concat_loader_train), correct / total

# Function to evaluate the model
def evaluate(model, test_loader, criterion):
    model.eval()  
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return total_loss / len(test_loader), correct / total

# Training loop
num_epochs = 30
history_baseline = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_val_accuracy = 0

# Set up logging
log_filename = os.path.join(PATH_LOGGINGS_FOLDER, 'Task1_augmentation_logfile.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader_cub, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate(model, test_loader_cub, criterion)
    history_baseline['train_loss'].append(train_loss)
    history_baseline['train_accuracy'].append(train_accuracy)
    history_baseline['val_loss'].append(val_loss)
    history_baseline['val_accuracy'].append(val_accuracy)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy {train_accuracy}, Test Loss: {val_loss}, Test Accuracy: {val_accuracy}')
    logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy {train_accuracy}, Test Loss: {val_loss}, Test Accuracy: {val_accuracy}')
    wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Test Loss": val_loss, "Test Accuracy": val_accuracy})

    # Save the model if it has the best validation accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        
        checkpoint_path = os.path.join(PATH_CHECKPOINT_FOLDER, 'Task1_augmentation_best_weights.pth')

        # Save model, optimizer, and scheduler state dicts
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'accuracy': val_accuracy,
        }, checkpoint_path)
        print(f"Checkpoint saved at Epoch {epoch+1}")

logging.shutdown()
wandb.finish()

filename =  os.pah.join(PATH_PICKLE_FOLDER,'Task1_augmentation_history.pickle')
with open(filename, 'wb') as file:
    pickle.dump(history_baseline, file)

# Plot the training and validation accuracy curves on the same figure   
input_size = (3, 224, 224)
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Assuming 'model2' is your PyTorch modelTask1_
# Create a sample input tensor
input_tensor = torch.randn(1, *input_size).to('cuda:0')

# Use fvcore to calculate FLOPs
flops = FlopCountAnalysis(model, input_tensor)
total_flops = flops.total()

# Convert FLOPs to GFLOPs
total_gflops = total_flops / 1e9

# Get the number of parameters
params = sum(p.numel() for p in model.parameters())
params_in_k = params / 1e3

# Optionally, print per-layer statistics
print(flops.by_module())
print(parameter_count_table(model))

# Print total FLOPs and number of parameters
print(f"{RED}Computational complexity (gLOPs): {total_gflops}{RESET}")
print(f"{CYAN}Number of parameters (K): {params_in_k}{RESET}")