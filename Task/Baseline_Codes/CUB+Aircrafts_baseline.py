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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

import wandb
wandb.init(project="CV703_task2_baseline")

RESULTS="/lustre/scratch/users/rusiru.achchige/Results"
PATH_CUB_DATASET = "./data/CUB/CUB_200_2011"
PATH_AIRCRAFT_DATASET = "./data/fgvc-aircraft-2013b"
PATH_CHECKPOINT_FOLDER = f'{RESULTS}/Checkpoints/Task2_baseline'
PATH_LOGGINGS_FOLDER = f'{RESULTS}/loggings/Task2_baseline'
PATH_PICKLE_FOLDER = f'{RESULTS}/pickels/Task2_baseline'

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
    
# write data transform here as per the requirement
test_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
])

# Define the data transformation/augmentation pipeline
train_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=train_data_transform, split="train")
test_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=test_data_transform, split="test")

# load in into the torch dataloader to get variable batch size, shuffle 
train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=128, drop_last=True, shuffle=True)
test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=128, drop_last=False, shuffle=False)

class FGVCAircraft(VisionDataset):
    """
    FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None):
        super(FGVCAircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]
        
        # Modify class index as we are going to concat to CUB dataset
        num_cub_classes = len(train_dataset_cub.class_to_idx)
        targets = [t + num_cub_classes for t in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images
    

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_dataset_aircraft = FGVCAircraft(root=f"{PATH_AIRCRAFT_DATASET}", transform=train_data_transform, train=True)
test_dataset_aircraft = FGVCAircraft(root=f"{PATH_AIRCRAFT_DATASET}", transform=test_data_transform, train=False)

# load in into the torch dataloader to get variable batch size, shuffle 
train_loader_aircraft = torch.utils.data.DataLoader(train_dataset_aircraft, batch_size=128, drop_last=True, shuffle=True)
test_loader_aircraft = torch.utils.data.DataLoader(test_dataset_aircraft, batch_size=128, drop_last=False, shuffle=False)

concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

concat_loader_train = torch.utils.data.DataLoader(
             concat_dataset_train,
             batch_size=128, shuffle=True,
             num_workers=4, pin_memory=True
            )
concat_loader_test = torch.utils.data.DataLoader(
             concat_dataset_test,
             batch_size=128, shuffle=False,
             num_workers=4, pin_memory=True
            )

model = timm.create_model('convnextv2_large', pretrained=True)
num_classes = 300
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Loss Function, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Function to train the model
def train(model, concat_loader_train, criterion, optimizer, scheduler):
    model.train()  
    total_loss = 0
    correct = 0
    total = 0
    print_every = 225
    batch_count = 0

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
log_filename = os.path.join(PATH_LOGGINGS_FOLDER, 'baseline_logfile.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, concat_loader_train, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate(model, concat_loader_test, criterion)
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
        checkpoint_path = os.path.join(PATH_CHECKPOINT_FOLDER, 'baseline_best_weights.pth')
        
        # Save the model checkpoint
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

filename =  os.pah.join(PATH_PICKLE_FOLDER,'baseline_history.pickle')
with open(filename, 'wb') as file:
    pickle.dump(history_baseline, file)
   
input_size = (3, 224, 224)
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

model.eval()  # Set the model to evaluation mode

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