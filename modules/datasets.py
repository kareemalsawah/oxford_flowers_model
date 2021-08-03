from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def txt_to_df(path):
    with open(path,'r') as f:
        lines = f.read()
        lines = lines.split("\n")
        df_data = []
        for line in lines:
            if len(line)>0: # To ignore empty lines
                line_split = line.split(" ")
                line_split[1] = int(line_split[1])  # It was checked that there are only ints and no null values
                df_data.append(line_split)
        df = pd.DataFrame(data=np.array(df_data), columns=["img_path", "target"])
        f.close()
    
    return df

def load_imgs(img_paths):
    imgs = []
    for path in img_paths:
        img = Image.open("./oxford-102-flowers/"+path)
        imgs.append(np.array(img))
    return np.array(imgs)

def load_dataframes():
    train_df = txt_to_df('/content/oxford-102-flowers/train.txt')
    valid_df = txt_to_df('/content/oxford-102-flowers/valid.txt')
    test_df = txt_to_df('/content/oxford-102-flowers/test.txt')

    data = {
        "train":train_df,
        "valid":valid_df,
        "test":test_df
    }

    return data

class FlowersDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        
        img_paths = self.data_df['img_path'].values
        self.imgs = load_imgs(img_paths)
        self.labels = self.data_df['target'].values.astype(int)
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx]
    
def get_dataloaders(batch_size, train_transform=None, test_transform=None, add_test=False):
    data = load_dataframes()

    train_dataset = FlowersDataset(data['train'],train_transform)
    valid_dataset = FlowersDataset(data['valid'],test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    
    if add_test:
        test_dataset = FlowersDataset(data['test'],test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return {'train':train_loader,
                'valid':valid_loader,
                'test':test_loader}
    else:
        return {'train':train_loader,
                'valid':valid_loader}
