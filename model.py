import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, pyramids):
        super(SPP, self).__init__()
        self.pools = [nn.AdaptiveMaxPool2d(pyramid) for pyramid in pyramids]

    def forward(self, x):
        outputs = []
        for pool in self.pools:
            outputs.append(torch.flatten(pool(x), 1))
        output = torch.cat(outputs, dim=1)
        return output
    

class ZFNet_SPP(nn.Module):
    def __init__(self, pyramids=[4, 3, 2, 1], num_classes=1000):
        super(ZFNet_SPP, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1), nn.ReLU(inplace=True))
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1), nn.ReLU(inplace=True))
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        
        self.spp = SPP(pyramids)
        channels = sum([num ** 2 for num in pyramids])
        
        self.fc1 = nn.Sequential(nn.Linear(256 * channels, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.spp(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x