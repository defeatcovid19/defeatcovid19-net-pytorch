import torch.nn as nn
from torchvision.models import resnet34

class Resnet34(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(Resnet34, self).__init__()
        resnet = resnet34(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        bottleneck_features = resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # mean = MEAN
        # std = STD
        x = x / 255.
        # x = torch.cat([
        #     (x[:, [0]] - mean[0]) / std[0],
        #     (x[:, [1]] - mean[1]) / std[1],
        #     (x[:, [2]] - mean[2]) / std[2],
        #     (x[:, [3]] - mean[3]) / std[3],
        # ], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x