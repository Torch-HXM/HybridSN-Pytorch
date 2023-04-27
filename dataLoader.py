import torch
import numpy as np
from typing import Tuple

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io as sio

from torch.utils.data import Dataset

np.random.seed(1)

class IndianPines(Dataset):
    def __init__(self, img, label) -> None:
        super(IndianPines, self).__init__()
        self.img = img
        self.label = label

    def __getitem__(self, index):
        return {"img" : self.img[index],
                "label" : self.label[index],
                "index" : index}
    
    def __len__(self):
        return self.img.shape[0]

class GetData():
    def __init__(self, windowSize=15, C=30, testRatio=0.7) -> None:
        self.windowSize = windowSize
        self.C = C
        self.testRatio = testRatio

    def loadData(self)->Tuple[np.ndarray, np.ndarray]:
        img = sio.loadmat("datasets/Indian_pines_corrected.mat")["indian_pines_corrected"]
        label = sio.loadmat("datasets/Indian_pines_gt.mat")["indian_pines_gt"]
        
        return img, label

    def splitTrainTestSet(self, X, y, testRatio, randomState=345):
        # print("y:", y.shape)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
        #                                                     stratify=y)
        
        # X: (10249, 15, 15, 30)  y: (10249,)
        X_train = np.zeros((1, self.C, self.windowSize, self.windowSize))
        X_test  = np.zeros((1, self.C, self.windowSize, self.windowSize))
        y_train = np.zeros((1, ))
        y_test  = np.zeros((1, ))
        
        class_num = {"train":np.zeros((16,)), "test":np.zeros((16,))}
        # get testRatio of data
        for i in range(16):
            x_i = X[y==i]
            y_i = y[y==i]
            
            i_len = x_i.shape[0]
            i_test_len = int(i_len*testRatio)
            class_num["train"][i]= i_len - i_test_len
            class_num["test"][i] = i_test_len
            
            all_random_index = np.random.choice(i_len, i_len, replace=False)
            test_random_index = np.random.choice(i_len, i_test_len, replace=False)
            train_random_index = np.setdiff1d(all_random_index, test_random_index)
            
            x_i_train, y_i_train = x_i[train_random_index], y_i[train_random_index]
            x_i_test, y_i_test  = x_i[test_random_index], y_i[test_random_index]
            
            X_train = np.vstack((X_train, x_i_train))
            X_test  = np.vstack((X_test, x_i_test))
            y_train = np.hstack((y_train, y_i_train))
            y_test  = np.hstack((y_test, y_i_test))
        
        # print("X:", X.shape)
        # print("y:", y.shape)
        # print("X_train:", X_train[1:].shape)
        # print("X_test:", X_test[1:].shape)
        # print("y_train:", y_train[1:].shape)
        # print("y_test:", y_test[1:].shape)
        # print(class_num)
        
        return X_train[1:], X_test[1:], y_train[1:], y_test[1:], class_num

    def applyPCA(self, X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        return newX, pca

    def padWithZeros(self, X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createImageCubes(self, X, y, windowSize=5, removeZeroLabels = True):
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self.padWithZeros(X, margin=margin)
        # split patches
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1
        return patchesData, patchesLabels

    def getData(self):

        X, y = self.loadData()
        X,pca = self.applyPCA(X,numComponents=self.C)
        X, y = self.createImageCubes(X, y, windowSize=self.windowSize)
        # X: (10249, 15, 15, 30)  y: (10249,)
        X = X.swapaxes(2, 3)
        X = X.swapaxes(1, 2)
        Xtrain, Xtest, ytrain, ytest, class_num = self.splitTrainTestSet(X, y, self.testRatio)
        Xtrain = Xtrain.reshape(-1, 1, self.C, self.windowSize, self.windowSize)
        Xtest = Xtest.reshape(-1, 1, self.C, self.windowSize, self.windowSize)

        Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.long)
        Xtest  = torch.tensor(Xtest , dtype=torch.float32)
        ytest  = torch.tensor(ytest , dtype=torch.long)

        return Xtrain, ytrain, Xtest, ytest, class_num

if __name__ == "__main__":
    getData = GetData(windowSize=15, C=30, testRatio=0.7)
    X, Y, x, y, class_num = getData.getData()
    print(X.shape)
    print(Y.shape)
    print(x.shape)
    print(y.shape)
    print(class_num)