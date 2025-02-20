import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pandas as pd

# Import necessary libraries
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import glob

class EyeDiseaseDataset(Dataset):
    def __init__(self, csv_file=None, images_dir=None, tif_dir=None, partition='train', transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with SLIDE annotations
            images_dir (str): Directory with SLIDE images
            tif_dir (str): Directory containing .tif files for bacterial keratitis
            partition (str): One of 'train', 'test', or 'val'
            transform (callable, optional): Optional transform to be applied
        """
        self.transform = transform
        self.data_items = []
        
        # Process SLIDE dataset if provided
        if csv_file is not None and images_dir is not None:
            slide_df = pd.read_csv(csv_file)
            if partition != 'all':
                slide_df = slide_df[slide_df['partition_group'] == partition].reset_index(drop=True)
            
            # Add SLIDE data
            for idx, row in slide_df.iterrows():
                self.data_items.append({
                    'path': os.path.join(images_dir, row['Filename']),
                    'label': row['epiphora_stage']
                })

        # Process .tif files if directory provided
        if tif_dir is not None:
            tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
            # If partition is specified, split the tif files accordingly
            if partition != 'all':
                # Deterministic split based on file names
                tif_files.sort()
                total = len(tif_files)
                if partition == 'train':
                    tif_files = tif_files[:int(0.8 * total)]
                elif partition == 'val':
                    tif_files = tif_files[int(0.8 * total):int(0.9 * total)]
                else:  # test
                    tif_files = tif_files[int(0.9 * total):]
            
            # Add bacterial keratitis data
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_items[idx]
        image = Image.open(item['path']).convert('RGB')
        label = self.label_map[item['label']]
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label_mapping(self):
        return self.label_map

# Example transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Print the structure of the class and its methods
print("Dataset class created successfully")
print("\
Available methods:")
print("- __len__: Returns the total number of samples")
print("- __getitem__: Returns a single sample (image, label) pair")
print("- get_label_mapping: Returns the mapping between class names and indices")

# Example of how to create the datasets
try:
    # Create dataset instance
    dataset = EyeDiseaseDataset(
        csv_file='SLID_E_information.csv',
        partition='train',
        transform=transform
    )
    
    print("\
Label mapping:")
    print(dataset.get_label_mapping())
    print("\
Total number of samples:", len(dataset))
    
except Exception as e:
    print("\
Error creating dataset:", str(e))



# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Paths
tif_dir = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\Blue_Light"
csv_file = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\SLID_E_information.csv"
images_dir = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\train"

# Create dataset
full_dataset = EyeDiseaseDataset(tif_dir=tif_dir, csv_file=csv_file, images_dir=images_dir, transform=transform)

# Split dataset into train and validation sets
train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Model definition
class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=3):  # Three classes: normal (0), epiphora (1), disease (2)
        super(EyeDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze base model weights

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify final layer

    def forward(self, x):
        return self.model(x)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeDiseaseModel(num_classes=3).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        train_accuracy = 100 * correct_preds / total_preds
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        val_accuracy = 100 * correct_preds / total_preds
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print("Training Complete")


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Load best model
model.load_state_dict(torch.load("best_model.pth"))


# Prediction function
def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# Example prediction
image_path = r'C:\Users\Ludovic\Pytorch\ExpoSciences\OIP.jpg'
predicted_class = predict(image_path)
print(f"Predicted Class: {predicted_class}")
