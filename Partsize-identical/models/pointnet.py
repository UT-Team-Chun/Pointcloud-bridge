import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# input.shape of PointNet: [batchsize=b, 3(coordinates)+d, number of points=n], maybe [100, 3, 10000]

# "STN" is Spatial Transformer Networks
# The first T-net is called "Input Transform", [b, 3+d, n] --> [b, 3, 3]
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) # input.shape: [batchsize=b, 3+d, number of points=n], output.shape: [b, 64, n]
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # output.shape: [b, 1024, n]
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0] # batch size = b
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # output.shape: [b, 1024, n]
        x = torch.max(x, 2, keepdim=True)[0] # output.shape: [b, 1024, 1]
        x = x.view(-1, 1024) # output.shape: [b, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # output.shape: [b, 9]

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1) # torch.Size([b, 9])
        # 1. "np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)"
        # 2. "torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))"
        # 3. "Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)))"
        # 4. + ".view(1, 9)", torch.Size([9]) to torch.Size([1, 9])
        # 5. + ".repeat(batchsize, 1)", torch.Size([1, 9]) to torch.Size([b, 9])

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # both torch.Size([b, 9])
        x = x.view(-1, 3, 3) # torch.Size([b, 3, 3])
        return x


# The second T-net is called "Feature Transform", [b, k, n] --> [b, k, k]
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] # output.shape: [b, 1024, 1]
        x = x.view(-1, 1024) # output.shape: [b, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # output.shape: [b, k * k]

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # both torch.Size([b, k * k])
        x = x.view(-1, self.k, self.k) # torch.Size([b, k, k])
        return x


# from start to "Global Feature"
# set "global_feat=True" for classification, "global_feat=False" for segmentation
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size() # "B": batch size; "D": dimension (the first 3 are coordinates); "N": number of points
        trans = self.stn(x) # first T-net, output: torch.Size([b, 3, 3]), every dimension is transformed
        x = x.transpose(2, 1) # input.shape: [b, 3+d, n], output.shape: [b, n, 3 + d]
        if D > 3:
            x, feature = x.split([3, D - 3], dim=2) # segment x into coordinates and other features, x: [b, n, 3], feature: [b, n, d]
        x = torch.bmm(x, trans) # "torch.bmm": matrix multiplication, x: [b, n, 3], trans: [b, 3, 3], so output: [b, n, 3]
        if D > 3:
            x = torch.cat([x, feature], dim=2) # only use T-net to the coordinates, output.shape: [b, n, 3 + d]
        x = x.transpose(2, 1) # input.shape: [b, n, 3 + d], output.shape: [b, 3 + d, n].       So the size remains the same after T-net!!!
        x = F.relu(self.bn1(self.conv1(x))) # output.shape: [b, 64, n], the first shared MLP

        if self.feature_transform:
            trans_feat = self.fstn(x) # second T-net, torch.Size([b, 64, 64])
            x = x.transpose(2, 1) # input.shape: [b, 64, n], output.shape: [b, n, 64]
            x = torch.bmm(x, trans_feat) # x: [b, n, 64], trans_feat: [b, 64, 64], so output: [b, n, 64]
            x = x.transpose(2, 1) # output.shape: [b, 64, n].       So the size remains the same after T-net!!!
        else:
            trans_feat = None

        pointfeat = x # x.shape: [b, 64, n], it is used for Segmentation Network
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # out.shape: [b, 1024, n], the second shared MLP (group)
        x = torch.max(x, 2, keepdim=True)[0] # out.shape: [b, 1024, 1]
        x = x.view(-1, 1024) # output.shape: [b, 1024]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # first: [b, 1024, 1], second: [b, 1024, n]
            return torch.cat([x, pointfeat], 1), trans, trans_feat # first: [b, 1088, n]


# this matrix (A) is approximately equal to an orthogonal matrix
# A * A.T = I
def feature_transform_regularizer(trans):
    d = trans.size()[1] # dimension
    I = torch.eye(d)[None, :, :] # [1, d, d]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss