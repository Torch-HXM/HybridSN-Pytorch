from dataLoader import GetData
from HybridSN import HybridSN

import torch
from torchsummary import summary

import numpy as np
import time, sys, os
import matplotlib.pyplot as plt

## GLOBAL VARIABLES
window_size = 15
C = 30

if __name__ == "__main__":
    # get test data
    getData = GetData(window_size, C)
    X, Y, test_img, test_label, class_num = getData.getData()
    # load model
    model = HybridSN(window_size)
    state_dict = torch.load("weight/my_model.pth")
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    
    # Specific class
    specific_class_acc = np.zeros(16)
    print("specific class accuracy".center(15, "="))
    print("class","acc".rjust(10))
    for i in range(16):
        test_img_s = test_img[test_label==i]
        test_label_s = test_label[test_label==i]
        
        test_pred_s = model(test_img_s)
        test_pred_s = torch.softmax(test_pred_s, dim=1)
        test_pred_s = torch.argmax(test_pred_s, dim=1)
        test_acc = sum(test_pred_s == test_label_s)/test_label_s.shape[0]
        print(format(i+1).ljust(2), format(test_acc.data, '.4f').rjust(15))
        specific_class_acc[i] = test_acc
    
    # OA
    test_pred = model(test_img)
    test_pred = torch.softmax(test_pred, dim=1)
    test_pred = torch.argmax(test_pred, dim=1)
    test_acc = sum(test_pred == test_label)/test_label.shape[0]
    print("OA", format(test_acc.data, '.4f').rjust(15))
    # AA
    print("AA", format(sum(specific_class_acc)/len(specific_class_acc), '.4f').rjust(15))
    # class_num
    print("data devide".center(15, "="))
    print("class    train num      test num")
    for i in range(16):

        print(format(i+1).ljust(2), format(int(class_num["train"][i])).rjust(10), format(int(class_num["test"][i])).rjust(15))
    # plot  
    loss = np.load("result/data/loss.npy")
    test_acc = np.load("result/data/test_acc.npy")
    test_loss = np.load("result/data/test_loss.npy")

    x = np.linspace(1, 30, 30)

    pic_path = "result/pic/"
    if not os.path.exists(pic_path): os.makedirs(pic_path)
    plt.plot(x, loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(pic_path+"loss")
    plt.show()

    plt.plot(x, test_acc)
    plt.xlabel("epoch")
    plt.ylabel("test_acc")
    plt.savefig(pic_path+"test_acc")
    plt.show()

    plt.plot(x, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("test_loss")
    plt.savefig(pic_path+"test_loss")
    plt.show()