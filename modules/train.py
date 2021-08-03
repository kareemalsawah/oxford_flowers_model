import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
        imgs, targets = imgs.to(device), targets.to(device)

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
            imgs, targets = imgs.to(device), targets.to(device)

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

def train(model,optimizer,loaders,n_epochs,lr):
    '''

    '''
    # Criterion
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


def evaluate(model,criterion,test_loader,calc_conf_matrix=False):
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    num_correct = 0
    total_num = 0
    preds = []
    corrects = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)

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
    
    # Draw Confusion Matrix

    # Other metrics here

    test_loss /= total_num
    accuracy = num_correct / total_num
    
    return test_loss, accuracy