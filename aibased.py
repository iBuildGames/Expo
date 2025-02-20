import torch 

import torch.nn as nn 

import torch.optim as optim 

import torch.nn.functional as F 

from torch.utils.data import DataLoader, Dataset 

from torchvision import transforms, datasets, models 

from sklearn.model_selection import train_test_split 

from PIL import Image 

import os 

import numpy as np 

import matplotlib.pyplot as plt 

 
class EyeDiseaseDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        # If image_paths is a directory, list files in that directory
        if isinstance(image_paths, str):  # Check if image_paths is a directory path (string)
            self.image_paths = [os.path.join(image_paths, f) for f in os.listdir(image_paths) if f.endswith('.tiff')]  # Adjust extension if needed
        elif isinstance(image_paths, list):  # If image_paths is already a list of paths
            self.image_paths = image_paths
        else:
            raise ValueError("image_paths should be either a directory path or a list of file paths.")
        
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image (you may use any image loading library, such as PIL or OpenCV)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


 

transform = transforms.Compose([ 

    transforms.Resize((224, 224)),   

    transforms.ToTensor(),           

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation 

]) 

 

image_dir = r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\White_Light" 

image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.tif')] 

labels = [0 if 'white_light' in fname else 1 for fname in os.listdir(image_dir)]  # Exemple de labels (0: Cataracte, 1: Glaucome) 

 

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42) 

 

train_dataset = EyeDiseaseDataset(train_paths, train_labels, transform=transform) 

val_dataset = EyeDiseaseDataset(val_paths, val_labels, transform=transform) 

 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 



class EyeDiseaseModel(nn.Module): 

    def __init__(self, num_classes=2): 

        super(EyeDiseaseModel, self).__init__() 

         

        self.model = models.resnet18(pretrained=True) 

        for param in self.model.parameters(): 

            param.requires_grad = False   

         

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)   

         

    def forward(self, x): 

        return self.model(x) 

 

model = EyeDiseaseModel(num_classes=2).cuda() 

 

criterion = nn.CrossEntropyLoss()   

optimizer = optim.Adam(model.parameters(), lr=1e-4) 

 

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10): 

    best_accuracy = 0.0 

    for epoch in range(num_epochs): 

        model.train() 

        running_loss = 0.0 

        correct_preds = 0 

        total_preds = 0 

 

        for inputs, labels in train_loader: 

            inputs, labels = inputs.cuda(), labels.cuda() 

 

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

         

 

        model.eval() 

        correct_preds = 0 

        total_preds = 0 

        with torch.no_grad(): 

            for inputs, labels in val_loader: 

                inputs, labels = inputs.cuda(), labels.cuda() 

 

                outputs = model(inputs) 

                _, predicted = torch.max(outputs, 1) 

                correct_preds += (predicted == labels).sum().item() 

                total_preds += labels.size(0) 

 

        val_accuracy = 100 * correct_preds / total_preds 

        print(f"Validation Accuracy: {val_accuracy:.2f}%") 

 

        if val_accuracy > best_accuracy: 

            best_accuracy = val_accuracy 

            torch.save(model.state_dict(), "best_model.pth") 

 

    print("Training Complete") 

 

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10) 

 

model.load_state_dict(torch.load("best_model.pth")) 

 

def predict(image_path): 

    model.eval() 

    image = Image.open(image_path).convert('RGB') 

    image = transform(image).unsqueeze(0).cuda() 

 

    with torch.no_grad(): 

        output = model(image) 

        _, predicted = torch.max(output, 1) 

    return predicted.item() 
     

 

image_path = r'C:\Users\Ludovic\Pytorch\ExpoSciences\OIP.jpg' 

predicted_class = predict(image_path) 

print(f"Predicted Class: {predicted_class}") 