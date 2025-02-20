import os
import glob
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Define the CombinedEyeDataset to handle both SLIDE images and .tif images
class CombinedEyeDataset(Dataset):
    def __init__(self, csv_file=None, images_dir=None, tif_dir=None, partition='train', transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with SLIDE annotations.
            images_dir (str): Directory with SLIDE images.
            tif_dir (str): Directory containing .tif files for bacterial keratitis.
            partition (str): One of 'train', 'test', 'val', or 'all' for SLIDE images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data_items = []
        
        # Process SLIDE dataset if provided
        if csv_file is not None and images_dir is not None:
            slide_df = pd.read_csv(csv_file)
            if partition != 'all':
                slide_df = slide_df[slide_df['partition_group'] == partition].reset_index(drop=True)
            for idx, row in slide_df.iterrows():
                self.data_items.append({
                    'path': os.path.join(images_dir, row['Filename']),
                    'label': row['epiphora_stage']
                })
                
        # Process .tif files if directory provided
        if tif_dir is not None:
            tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
            if partition != 'all':
                tif_files.sort()
                total = len(tif_files)
                if partition == 'train':
                    tif_files = tif_files[:int(0.8 * total)]
                elif partition == 'val':
                    tif_files = tif_files[int(0.8 * total):int(0.9 * total)]
                else:  # test
                    tif_files = tif_files[int(0.9 * total):]
            for tif_file in tif_files:
                self.data_items.append({
                    'path': tif_file,
                    'label': 'bacterial_keratitis'
                })
        
        # Create label mapping including both SLIDE categories and bacterial keratitis
        unique_labels = set([item['label'] for item in self.data_items])
        self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        image = Image.open(item['path']).convert('RGB')
        label = self.label_map[item['label']]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label_mapping(self):
        return self.label_map

# Define image transformations for training and validation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Specify your directories here
slide_images_dir = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\train\train"  # Update with your path
tif_images_dir = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\Blue_Light"        # Update with your path
csv_path = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\SLID_E_information.csv"
# Create dataset instances
train_dataset = CombinedEyeDataset(csv_file=csv_path,
                                   images_dir=slide_images_dir,
                                   tif_dir=tif_images_dir,
                                   partition='train',
                                   transform=train_transform)

val_dataset = CombinedEyeDataset(csv_file=csv_path,
                                 images_dir=slide_images_dir,
                                 tif_dir=tif_images_dir,
                                 partition='val',
                                 transform=val_transform)

test_dataset = CombinedEyeDataset(csv_file=csv_path,
                                  images_dir=slide_images_dir,
                                  tif_dir=tif_images_dir,
                                  partition='test',
                                  transform=val_transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Print label mapping
print('Label Mapping:', train_dataset.get_label_mapping())

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a model using a pretrained ResNet18 and adjusting the final linear layer
num_classes = len(train_dataset.get_label_mapping())
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print('Epoch: {}/{} \tTrain Loss: {:.4f} \tVal Loss: {:.4f} \tVal Acc: {:.4f}'.format(
            epoch+1, num_epochs, epoch_loss, val_loss, val_acc))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saving best model...')
    print('Training complete.')

# Train the model
num_epochs = 10
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Define a function to evaluate the model on the test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images