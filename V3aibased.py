# Import necessary libraries
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import glob

class CombinedEyeDataset(Dataset):
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
    dataset = CombinedEyeDataset(
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
    
# Create train/val/test datasets
train_dataset = CombinedEyeDataset(
    csv_file='SLID_E_information.csv',
    images_dir='path_to_slide_images',
    tif_dir='path_to_tif_files',
    partition='train',
    transform=transform
)

val_dataset = CombinedEyeDataset(
    csv_file='SLID_E_information.csv',
    images_dir='path_to_slide_images',
    tif_dir='path_to_tif_files',
    partition='val',
    transform=transform
)

test_dataset = CombinedEyeDataset(
    csv_file='SLID_E_information.csv',
    images_dir='path_to_slide_images',
    tif_dir='path_to_tif_files',
    partition='test',
    transform=transform
)