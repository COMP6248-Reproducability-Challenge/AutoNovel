import torch.nn as nn
import torch.nn.functional as F
import torch
class myResNet(nn.Module):
    def __init__(self, model, num_labeled_classes = 5, num_unlabeled_classes = 5, has_unlabeled = False):
        super(myResNet, self).__init__()
        self.has_unlabeled = has_unlabeled
        # self.conv1 = model.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = model.bn1
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4  = model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu2 = nn.ReLU(inplace = True)
        self.head1 = nn.Linear(512, num_labeled_classes)
        self.head2 = nn.Linear(512, num_unlabeled_classes)


    def forward(self, out):
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.relu2(out)
        out1 = self.head1(out)
        if self.has_unlabeled:
            out2 = self.head2(out)
            return out1, out2, out
        return out1