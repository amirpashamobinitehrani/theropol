import os
import numpy as np
import cv2
import glob
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms


class MouthData(Dataset):

    def __init__(self,
                 root_dir,
                 subset = "training",
                 transform=None):
        
        self.root_dir = root_dir
        file_list = glob.glob(self.root_dir + "/*")
        self.subset = subset
        self.transform = transform
        self.data = []

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for image_path in glob.glob(class_path  + "/*.png"):
                self.data.append([image_path, class_name])
        
        self.class_map = {"0": 0,
                          "1": 1,
                          "2": 2,
                          "3": 3,
                          "4": 4,
                          "5": 5}
 
    def __getitem__(self, idx):
        
        #load image and labels
        image_path, class_name = self.data[idx]
        image = cv2.imread(image_path)

        #convert image arrays to tensors, normalize and reshape
        image_tensor = torch.from_numpy(image.astype(np.float32)) / 255.
        image_tensor = image_tensor.permute(2, 0, 1)
        
        #get class ids and convert to one hot vectors
        class_id = self.class_map[class_name]        
        class_id = F.one_hot(torch.tensor([class_id]), num_classes = 6)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        
        return image_tensor, class_id

  
    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    
    root = "/Users/amirpashamobinitehrani/Desktop/data"
    dataset = MouthData(root)
    dataloader = DataLoader(dataset,
                            batch_size = 1,
                            shuffle = True,
                            num_workers = 4)
    
    for image, label in dataloader:
        
        print(f"Batch of images has shape: {image.shape}" )
        print(f"Batch of labels has shape: {label.shape}")

        #print a single data and label tensor
        print(image)
        print(label)
        
        break

    