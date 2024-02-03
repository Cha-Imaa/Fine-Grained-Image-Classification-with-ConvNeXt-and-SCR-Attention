import os
import torch
import timm
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

PATH_CUB_DATASET = "./data/CUB/CUB_200_2011"
PATH_AIRCRAFT_DATASET = "./data/fgvc-aircraft-2013b"
CHECKPOINT='/lustre/scratch/users/rusiru.achchige/Results/Checkpoints/Task2_baseline/Task2_baseline_best_weights.pth'

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

train_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize to a standard size
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize
])

train_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=train_data_transforms, split="train")
test_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=test_data_transform, split="test")
test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=64, drop_last=False, shuffle=False)

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

train_dataset_aircraft = FGVCAircraft(root=f"{PATH_AIRCRAFT_DATASET}", transform=train_data_transforms, train=True)
test_dataset_aircraft = FGVCAircraft(root=f"{PATH_AIRCRAFT_DATASET}", transform=test_data_transform, train=False)

test_loader_aircraft = torch.utils.data.DataLoader(test_dataset_aircraft, batch_size=128, drop_last=False, shuffle=False)

concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

concat_loader_test = torch.utils.data.DataLoader(
             concat_dataset_test,
             batch_size=128, shuffle=False,
             num_workers=4, pin_memory=True
            )

num_classes = 300
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
    for batch in tqdm(concat_loader_test):
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
