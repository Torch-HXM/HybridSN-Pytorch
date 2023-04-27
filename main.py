from dataLoader import GetData, IndianPines
from HybridSN import HybridSN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

import numpy as np
import time, sys, os

## GLOBAL VARIABLES
window_size = 15
C = 30

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epoch_all = 30
    batch_size = 20

    getData = GetData(window_size, C)
    X, Y, test_img, test_label, class_num = getData.getData()

    indianPines = IndianPines(X, Y)
    train_data_loader = DataLoader(indianPines, batch_size=batch_size, shuffle=True)
    
    model = HybridSN(window_size)
    summary(model, (1, C, window_size, window_size))
    
    test_img = test_img.to(device)
    test_label = test_label.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters() ,lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    loss_recoder = []
    test_loss_recoder = []
    test_acc_recoder = []

    for epoch in range(epoch_all):
        with torch.enable_grad():
            start_time = time.time()
            loss = None
            test_acc = 0
            for data in train_data_loader:
                # forward pass
                e_img, e_label, index = data["img"].to(device), data["label"].to(device), data["index"]
                pred = model(e_img)
                print(pred.shape)
                print(e_label.shape)
                loss = loss_func(pred, e_label)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # val loss
        with torch.no_grad():
            test_pred = model(test_img)
            test_loss = loss_func(test_pred, test_label)
            # val acc
            test_pred = torch.softmax(test_pred, dim=1)
            test_pred = torch.argmax(test_pred, dim=1)
            test_acc = sum(test_pred == test_label)/test_label.shape[0]
            # print
            finish_time = time.time()
            sys.stdout.write("Epoch %d/%d, Loss %f, val-loss %f, val-acc %f, Time %f."%(epoch+1, epoch_all, loss.data.item(), test_loss.data.item(), test_acc.data, finish_time-start_time))
            # data recoder
            loss_recoder.append(loss.data.item())
            test_loss_recoder.append(test_loss.data.item())
            test_acc_recoder.append(test_acc.cpu())
            # save the best
            if test_acc == max(test_acc_recoder):
                path = 'weight/'
                if not os.path.exists(path): os.makedirs(path)
                torch.save(model.state_dict(), path+'my_model.pth')
                print("Epoch %d, model saved."%(epoch))
            else:
                print("Epoch %d, model rejected."%(epoch))
    
    # save loss and acc
    data_path = "./result/data/"
    if not os.path.exists(data_path): os.makedirs(data_path)
    np.save(data_path+'loss.npy', np.asarray(loss_recoder))
    np.save(data_path+'test_loss.npy', np.asarray(test_loss_recoder))
    np.save(data_path+'test_acc.npy', np.asarray(test_acc_recoder))
    
    print("Finish.")

