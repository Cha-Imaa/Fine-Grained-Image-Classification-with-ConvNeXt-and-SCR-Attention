import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm


PATH_CUB_DATASET = "./data/CUB/CUB_200_2011"
CHECKPOINT='/lustre/scratch/users/rusiru.achchige/Results/Checkpoints/Task1_baseline/Task1_baseline_best_weights.pth'

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
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean, std=std)
])

test_dataset_cub = CUBDataset(image_root_path=f"{PATH_CUB_DATASET}", transform=test_data_transform, split="test")
test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=64, drop_last=False, shuffle=False)

num_classes = 200
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
    for batch in tqdm(test_loader_cub):
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
