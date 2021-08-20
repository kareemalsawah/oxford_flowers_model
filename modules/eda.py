import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from .datasets import *

def class_dists(data):
    '''
    Plots the distribution of the different classes

    Parameters
    ----------
    data: dict
        Dictionary with dataframes for each of train, valid, and test datasets
    '''
    # Dataset Size
    for idx,key in enumerate(data.keys()):
        print("Number of {} rows: {}".format(key,data[key].shape[0]))

    # Plot class distributions
    _, ax = plt.subplots(3,figsize=(10,6))

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
    '''
    Checks for leaks in the data

    Parameters
    ----------
    data: dict
        Dictionary with dataframes for each of train, valid, and test datasets
    '''
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

    print("No data leaks found")

def img_size_hists(imgs):
    '''
    Plot a histogram of image shapes (width and height)
    
    Parameters
    ----------
    imgs: np.array, shape = (num_images, height, weight, 3)
    '''
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

def plot_imgs_diff_sizes(imgs, n_plot:int=4):
    '''
    Plot a sample of images for each of the different sizes of images

    Parameters
    ----------
    imgs: np.array, shape = (num_images, height, weight, 3)

    n_plot: int
        The number of images to plot for each size of images
    '''
    shapes = []
    for img in imgs:
        shapes.append([img.shape[0],img.shape[1]])
    shapes = np.array(shapes)

    # The different ranges of heights and widths
    heights = [(500,510),(510,800),(800,1000)]
    widths = [(500,510),(510,640),(640,680),(680,730),(730,770),(770,1000)]
    

    # Plot different heights
    for height in heights:
        width = (500,510)
        cond_1 = np.logical_and(height[0]<=shapes[:,0],shapes[:,0]<=height[1])
        cond_2 = np.logical_and(width[0]<=shapes[:,1],shapes[:,1]<=width[1])
        compatible_imgs = imgs[np.logical_and(cond_1,cond_2)]
        n_to_plot = min(n_plot,compatible_imgs.shape[0])
        if n_to_plot == 0:
            print("No imgs for height in {}, width in {}".format(height, width))
            print(np.sum(cond_1))
            print(np.sum(cond_2))
        fig,ax = plt.subplots(1,n_to_plot, figsize=(15,10))
        plt.suptitle("Imgs with height in {}, width in {}".format(height,width))
        for i in range(n_to_plot):
            ax[i].imshow(compatible_imgs[i])
        fig.tight_layout()
        if n_to_plot > 0:
            fig.subplots_adjust(top=1.2)
            plt.show()

    # Plot different widths
    for width in widths:
        height = (500,510)
        cond_1 = np.logical_and(height[0]<=shapes[:,0],shapes[:,0]<=height[1])
        cond_2 = np.logical_and(width[0]<=shapes[:,1],shapes[:,1]<=width[1])
        compatible_imgs = imgs[np.logical_and(cond_1,cond_2)]
        n_to_plot = min(n_plot,compatible_imgs.shape[0])
        if n_to_plot == 0:
            print("No imgs for height in {}, width in {}".format(height, width))
            print(np.sum(cond_1))
            print(np.sum(cond_2))
        fig,ax = plt.subplots(1,n_to_plot, figsize=(15,10))
        plt.suptitle("Imgs with height in {}, width in {}".format(height,width))
        for i in range(n_to_plot):
            ax[i].imshow(compatible_imgs[i])
        fig.tight_layout()
        if n_to_plot > 0:
            fig.subplots_adjust(top=1.55)
            plt.show()

def plot_imgs_from_classes(imgs, labels, n_plot:int=5):
    '''
    Plot a sample of images from each class

    Parameters
    ----------
    imgs: np.array, shape = (num_images, height, weight, 3)

    labels: np.array, shape = (num_images)
        The class label for each image
    n_plot: int
        The number of images to plot for each class
    '''
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