import torch
import os
import timm
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
warnings.filterwarnings('ignore')

PATH_FOODX_DATASET = "./data/FoodX/food_dataset"
CHECKPOINT='/lustre/scratch/users/rusiru.achchige/Results/Checkpoints/Task3_baseline/Task3_baseline_best_weights.pth'

split = 'train'
train_df = pd.read_csv(f'{PATH_FOODX_DATASET}/annot/{split}_info.csv', names= ['image_name','label'])
train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{PATH_FOODX_DATASET}/{split}_set/', x))

split = 'val'
val_df = pd.read_csv(f'{PATH_FOODX_DATASET}/annot/{split}_info.csv', names= ['image_name','label'])
val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{PATH_FOODX_DATASET}/{split}_set/', x))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# write data transform here as per the requirement
test_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean, std=std)
])

# Define the data transformation/augmentation pipeline
train_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, set):
        self.dataframe = dataframe
        self.set = set

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        if self.set  == 'train':
            return (train_data_transform(Image.open(row["path"])), row['label'])
        elif self.set  == 'val':
            return (test_data_transform(Image.open(row["path"])), row['label'])
        

train_dataset = FOODDataset(train_df, 'train')
val_dataset = FOODDataset(val_df, 'val')

# load in into the torch dataloader to get variable batch size, shuffle 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, drop_last=False, shuffle=True)

num_classes = train_df['label'].nunique()
model = timm.create_model('convnextv2_large', pretrained=False)
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda')

# Evaluate the model
model.eval()  # set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # turn off gradients
    for batch in tqdm(val_loader):
        data, target = batch

        data = data.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))
        outputs = model(data)
        
        # Get the predicted class labels
        _, predicted = torch.max(outputs.data, 1)

        # Update total and correct counts
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Print the number of correct classifications
print(f'Number of correct classifications: {correct}')
print(f'Accuracy: {100 * correct / total}%')
