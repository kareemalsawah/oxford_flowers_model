import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from .datasets import *

import matplotlib.pyplot as plt

def train_epoch(model,optimizer,criterion,train_loader):
    '''
    Train a given model for 1 epoch

    Parameters
    ----------
    model: torch.nn.Module
        The module to train with "forward" implemented
    optimizer: torch.nn.optim
        Optimizer to use during training
    criterion: 
    train_loader: DataLoader

    Returns
    -------
    train_losses: list of floats
        The loss during each iteration of training
    '''
    device = next(model.parameters()).device
    model.train()
    train_losses = []
    train_acc = []
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.type(torch.LongTensor).to(device)

        pred = model.forward(imgs)

        loss = criterion(pred,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item()/imgs.shape[0])
        pred_probs = torch.exp(pred).detach().cpu().numpy()
        pred_classes = np.argmax(pred_probs,axis=1)
        np_targets = targets.cpu().numpy()
        num_correct = np.sum(pred_classes==np_targets)
        train_acc.append(num_correct/imgs.shape[0])
    
    return train_losses, train_acc

def test(model,criterion,test_loader):
    '''

    '''
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    num_correct = 0
    total_num = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.type(torch.LongTensor).to(device)

            pred = model.forward(imgs)

            loss = criterion(pred,targets)

            test_loss += loss.item()
            total_num += imgs.shape[0]

            pred_probs = torch.exp(pred).cpu().numpy()
            pred_classes = np.argmax(pred_probs,axis=1)
            np_targets = targets.cpu().numpy()
            num_correct += np.sum(pred_classes==np_targets)
    
    test_loss /= total_num
    accuracy = num_correct / total_num
    
    return test_loss, accuracy

def train(model,loaders,n_epochs, lr, weight_decay=0.005):
    '''

    '''
    # Criterion
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    val_losses = []
    val_accs = []
    train_losses = []
    train_accs = []

    
    # Training
    for e in tqdm(range(1,n_epochs+1),position=0,leave=True):
        val_loss, val_acc = test(model,criterion,loaders['valid'])
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        train_loss, train_acc = train_epoch(model,optimizer,criterion,loaders['train'])
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
    val_loss, val_acc = test(model,criterion,loaders['valid'])
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    iters_per_epoch = len(train_losses)/(len(val_losses)-1)

    plt.title("Training and Validation Loss")
    plt.plot(np.arange(0,len(train_losses)),train_losses,label="Train Loss")
    plt.plot(np.arange(0,len(val_losses))*iters_per_epoch,val_losses,label="Valid Loss")
    plt.show()

    plt.title("Training and Validation Top-1 Accuracy")
    plt.plot(np.arange(0,len(train_losses)),train_accs,label="Train Acc")
    plt.plot(np.arange(0,len(val_losses))*iters_per_epoch,val_accs,label="Valid Acc")
    plt.show()


def evaluate(model,test_loader, num_wrong:int):
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    num_correct = 0
    total_num = 0
    preds = []
    corrects = []

    criterion = nn.NLLLoss()
    
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader,position=0,leave=True):
            imgs, targets = imgs.to(device), targets.type(torch.LongTensor).to(device)

            pred = model.forward(imgs)

            loss = criterion(pred,targets)

            test_loss += loss.item()
            total_num += imgs.shape[0]

            pred_probs = torch.exp(pred).cpu().numpy()
            pred_classes = np.argmax(pred_probs,axis=1)
            np_targets = targets.cpu().numpy()
            num_correct += np.sum(pred_classes==np_targets)
            preds.extend(pred_classes.tolist())
            corrects.extend(np_targets.tolist())

    preds = np.array(preds)
    corrects = np.array(corrects)

    precs = precision_score(corrects,preds,average=None)
    f1s = f1_score(corrects,preds,average=None)
    classes = np.arange(0,102)

    # Draw precisions
    plt.figure(figsize=(10,5))
    plt.title("Precisions")
    plt.bar(classes,precs*100)
    plt.xlabel("Precision (%)")
    plt.ylabel("Class ID")
    plt.show()

    # Draw F1s
    plt.figure(figsize=(10,5))
    plt.title("F1s")
    plt.bar(classes,f1s*100)
    plt.xlabel("F1 (%)")
    plt.ylabel("Class ID")
    plt.show()

    print("Best classes in precision {}".format(np.argsort(precs)[-3:]))
    print("Worst classes in precision {}".format(np.argsort(precs)[:3]))

    print("Best classes in F1 {}".format(np.argsort(f1s)[-3:]))
    print("Worst classes in F1 {}".format(np.argsort(f1s)[:3]))

    test_loss /= total_num
    accuracy = num_correct / total_num
    prec = precision_score(corrects,preds,average='macro')
    f1 = f1_score(corrects,preds,average='macro')
    print("Overall Accuracy: {}%".format(accuracy*100))
    print("Overall Precision: {}%".format(prec*100))
    print("Overall F1: {}%".format(f1*100))

    worst_class = np.argsort(f1s)[0]

    # num_wrong misclassified images
    indices = np.logical_and(corrects==worst_class,np.logical_not(corrects==preds))
    indices = np.arange(preds.shape[0])[indices]
    indices = indices[:num_wrong]
    imgs = get_imgs(indices, test=True)

    return imgs, corrects[indices]


def grad_cam_viz(model, imgs, labels, alpha:float=0.8):
    '''
    Use GRADCAM to visualize the activations of the last layer of the given model

    Parameters
    ----------
    model: nn.Module
        Model to use
    imgs: torch.tensor
        Tensor of images of shape (num_images, C, H, W)
    labels: torch.tensor
        Correct class labels for the given images

    Warnings
    --------
    - model should have the function get_heatmap implemented
    - It should return heatmaps (resized to size of input images) for all given images and classes
    '''
    device = next(model.parameters()).device
    imgs, labels = imgs.to(device), labels.to(device)

    pred_probs = torch.exp(model.forward(imgs)).detach().cpu().numpy()
    pred_classes = torch.from_numpy(np.argmax(pred_probs,axis=1))

    for idx, img in enumerate(imgs):
        img = img.reshape(1,3,224,224)
        class_pred = pred_classes[idx]
        class_true = labels[idx]
        pred_heatmap = model.get_heatmap(img, class_pred)[0]
        corr_heatmap = model.get_heatmap(img, class_true)[0]
        img_show = img[0].detach().cpu().numpy().transpose(1,2,0)

        fig,ax = plt.subplots(1,2,figsize=(15,10))
        ax[0].set_title("Correct Class: {}".format(class_true))
        ax[0].imshow(img_show)
        ax[0].imshow(pred_heatmap, alpha=alpha)
        
        ax[1].set_title("Incorrect Class: {}".format(class_pred))
        ax[1].imshow(img_show)
        ax[1].imshow(corr_heatmap, alpha=alpha)

        fig.tight_layout()
        plt.show()
        
