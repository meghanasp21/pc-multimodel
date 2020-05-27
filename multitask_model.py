import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class multi_model(nn.Module):
    def __init__(self,input_size,num_classes):
        super(multi_model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size,out_channels=8,kernel_size=3,stride=1,padding=1), #same convolution
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())

        self.bn1 = nn.BatchNorm1d(8, eps=2e-1)
        self.max = nn.MaxPool1d(8)

        self.y1o_1 = nn.Linear(256,128)
        self.y1o_2 = nn.Linear(128, num_classes)

        self.y2o_1 = nn.Linear(256,128)
        self.y2o_2 = nn.Linear(128, num_classes)

        # nn.init.xavier_normal_(self.y1o.weight)
        # self.y2o = nn.Linear(64,num_classes)
        # nn.init.xavier_normal_(self.y2o.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.max(out)

        out = out.reshape(out.shape[0], -1)

        head1 =self.y1o_1(out)
        head1 = self.y1o_2(head1)

        head2 = self.y2o_1(out)
        head2 = self.y2o_2(head2)

        return F.softmax(head1,dim=1), F.softmax(head2,dim=1)