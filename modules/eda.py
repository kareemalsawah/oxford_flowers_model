import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from .datasets import *

def class_dists(data):
    # Dataset Size
    for idx,key in enumerate(data.keys()):
        print("Number of {} rows: {}".format(key,data[key].shape[0]))

    # Plot class distributions
    fig,ax = plt.subplots(3,figsize=(10,6))

    for idx,key in enumerate(data.keys()):
        df = data[key]
        unique_classes = np.unique(df['target'].values).astype(int)
        unique_classes = np.sort(unique_classes)
        num_classes = unique_classes.shape[0]
        print("Number of {} classes: {}".format(key,num_classes))
        ax[idx].hist(df['target'],bins=num_classes)
        ax[idx].set_xticklabels([unique_classes[0]]+[""]*(num_classes-2)+[unique_classes[-1]])
        ax[idx].set_title("{} targets".format(key))

    plt.show()

def check_leaks(data):
    train_df = data['train']
    valid_df = data['valid']
    test_df = data['test']

    # Check for leaks using paths
    train_imgs = set(train_df['img_path'].values.tolist())
    valid_imgs = set(valid_df['img_path'].values.tolist())
    test_imgs = set(test_df['img_path'].values.tolist())

    train_leak_val = train_imgs.intersection(valid_imgs)
    val_leak_test = valid_imgs.intersection(test_imgs)
    train_leak_test = train_imgs.intersection(test_imgs)
    assert len(train_leak_val) == 0
    assert len(val_leak_test) == 0
    assert len(train_leak_test) == 0

    # Check for leaks using the actual images

    print("No data leaks found")

def img_size_hists(imgs):
    shapes = []
    for img in imgs:
        shapes.append(list(img.shape[:2]))
    shapes = np.array(shapes)

    fig,ax = plt.subplots(3,figsize=(15,10))
    ax[0].hist(shapes[:,0],bins=np.unique(shapes[:,0]).shape[0]//5)
    ax[0].set_title("Height of images")

    ax[1].hist(shapes[:,1],bins=np.unique(shapes[:,1]).shape[0]//5)
    ax[1].set_title("Width of images")

    ax[2].hist2d(shapes[:,0],shapes[:,1],bins=np.unique(shapes[:,0]).shape[0]//5)
    ax[2].set_title("2D Histogram of width and height")
    fig.tight_layout()
    plt.show()

def plot_imgs_from_classes(imgs, labels, n_plot:int=5):
    for i in range(102):
        img_idx = np.where(labels==i)[0][:n_plot]
        compatible_imgs = imgs[img_idx]
        n_to_plot = min(n_plot,compatible_imgs.shape[0])
        fig,ax = plt.subplots(1,n_to_plot, figsize=(15,10))
        plt.suptitle("Class {}".format(i))
        for i in range(n_to_plot):
            ax[i].imshow(compatible_imgs[i])
        fig.tight_layout()
        if n_to_plot > 0:
            fig.subplots_adjust(top=1.2)
            plt.show()
        plt.close()
        print()