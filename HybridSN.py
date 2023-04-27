import torch
import torch.nn as nn
from torchsummary import summary

class HybridSN(nn.Module):
    def __init__(self, window_size) -> None:
        super(HybridSN, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.ReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(nn.Conv3d(8, 16, (5, 3, 3)), nn.ReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(nn.Conv3d(16, 32, (3, 3, 3)), nn.ReLU(inplace=True))
        # [1, 32, 18, 19, 19] # [1, 576, 19, 19]
        self.conv_layer4 = nn.Sequential(nn.Conv2d(576, 64, (3, 3)), nn.ReLU(inplace=True))
        self.flatten = nn.Flatten()
        self.dense_layer1 = nn.Sequential(nn.Linear(64*(window_size-8)**2, 256), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.dense_layer2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.dense_layer3 = nn.Sequential(nn.Linear(128, 16))
        
    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = torch.reshape(x, (-1, x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv_layer4(x)

        x = self.flatten(x)
        
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)
        return x
    
if __name__ == "__main__":
    # test_data = torch.randint(0, 100, (1, 1, 30, 15, 15), dtype=torch.float32)
    model = HybridSN(15)
    # model_out = model(test_data)
    # print(model_out.shape)
    summary(model, (1, 30, 15, 15))
