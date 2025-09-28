import torch
import torch.nn as nn


class up_sampling(nn.Module):
    def __init__(self, batch_size):
        super(up_sampling, self).__init__()
        self.batch_size = batch_size
        self.matrix_visual = nn.Parameter(torch.zeros(self.batch_size, 50, 350), requires_grad=True)
        self.matrix_acoustic = nn.Parameter(torch.zeros(self.batch_size, 50, 370), requires_grad=True)
        self.up = nn.Upsample(scale_factor=10, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=5, mode='nearest')
        self.linear = nn.Linear(350, 350)
        self.linear2 = nn.Linear(370, 370)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(50)
    def forward(self, visual, acoustic):
        visual_ = self.up(visual)
        acoustic_ = self.up2(acoustic)
        visual_ = visual_+self.matrix_visual
        acoustic_ = acoustic_+self.matrix_acoustic
        visual_ = self.linear(visual_)
        acoustic_ = self.linear2(acoustic_)
        visual_ = self.norm(visual_)
        acoustic_ = self.norm(acoustic_)
        return visual_, acoustic_

class up_sampling2(nn.Module):
    def __init__(self, batch_size):
        super(up_sampling2, self).__init__()
        self.batch_size = batch_size
        self.matrix_visual = nn.Parameter(torch.zeros(self.batch_size, 50, 700), requires_grad=True)
        self.matrix_acoustic = nn.Parameter(torch.zeros(self.batch_size, 50, 740), requires_grad=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(700, 768)
        self.linear2 = nn.Linear(740, 768)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(50)
    def forward(self, visual, acoustic):
        visual_ = self.up(visual)
        acoustic_ = self.up(acoustic)
        visual_ = visual_+self.matrix_visual
        acoustic_ = acoustic_+self.matrix_acoustic
        visual_ = self.linear(visual_)
        acoustic_ = self.linear2(acoustic_)
        visual_ = self.norm(visual_)
        acoustic_ = self.norm(acoustic_)
        return visual_, acoustic_

class up_sampling_128(nn.Module):
    def __init__(self, batch_size):
        super(up_sampling_128, self).__init__()
        self.batch_size = batch_size
        self.matrix_visual = nn.Parameter(torch.zeros(self.batch_size, 50, 128), requires_grad=True)
        self.matrix_acoustic = nn.Parameter(torch.zeros(self.batch_size, 50, 128), requires_grad=True)
        self.up = nn.Upsample(scale_factor=3.68, mode='linear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=1.74, mode='linear', align_corners=True)
        self.linear = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(50)
    def forward(self, visual, acoustic):
        visual_ = self.up(visual)
        acoustic_ = self.up2(acoustic)
        visual_ = visual_+self.matrix_visual
        acoustic_ = acoustic_+self.matrix_acoustic
        visual_ = self.linear(visual_)
        acoustic_ = self.linear2(acoustic_)
        visual_ = self.norm(visual_)
        acoustic_ = self.norm(acoustic_)
        return visual_, acoustic_


class group_sampling(nn.Module):
    def __init__(self, batch_size):
        super(group_sampling, self).__init__()
        self.batch_size = batch_size
        self.up1 = up_sampling(self.batch_size)
        self.up2 = up_sampling2(self.batch_size)
        self.up_128 = up_sampling_128(self.batch_size)
        self.norm = nn.BatchNorm1d(50)
    def forward(self, visual, acoustic):
        visual_list = []
        acoustic_list = []
        for _ in range(6):
            v, a = self.up_128(visual, acoustic)
            visual_list.append(v)
            acoustic_list.append(a)
        fine_visual = torch.cat(visual_list, dim=2)
        fine_acoustic = torch.cat(acoustic_list, dim=2)
        visual_, acoustic_ = self.up1(visual, acoustic)
        visual_, acoustic_ = self.up2(visual_, acoustic_)
        return fine_visual+visual_, fine_acoustic+acoustic_

# a1 = torch.zeros(128,50,35)
# a2 = torch.zeros(128,50,74)
# b1, b2 = group_sampling(a1 , a2 ,128)
# print(b1.shape)


