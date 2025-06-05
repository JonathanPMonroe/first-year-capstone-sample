import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#I transform the data to tensor for the CNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),         
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),                  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_mapping = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["frame_path"]
        label_str = self.dataframe.iloc[idx]["label"]
        numeric_label = self.label_mapping.get(label_str.upper(), -1)
        if numeric_label == -1:
            raise ValueError(f"Unknown label {label_str} found.")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(numeric_label, dtype=torch.long)
