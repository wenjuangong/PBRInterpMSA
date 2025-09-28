import torch
import torch.nn as nn
import torch.nn.functional as F
from global_configs import *
import math
from up_sampling import group_sampling
from dc1d.nn import DeformConv1d
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

class CE(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2):
        super(CE, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.hv = SelfAttention(TEXT_DIM)
        self.ha = SelfAttention(TEXT_DIM)
        self.va = SelfAttention(TEXT_DIM)
        self.multihead_routing2 = multihead_routing2(routing_head, batch_size)
        self.routing = multihead_routing2(routing_head, 4659 % batch_size)
        self.SharedEncoder = nn.LSTM(input_size=768, hidden_size=768, batch_first=True)
        self.visualSpecificEncoder = nn.LSTM(input_size=768, hidden_size=768, batch_first=True)
        self.acousticSpecificEncoder = nn.LSTM(input_size=768, hidden_size=768, batch_first=True)
        self.textSpecificEncoder = nn.LSTM(input_size=768, hidden_size=768, batch_first=True)
        self.upsample = group_sampling(batch_size)
        self.upsample2 = group_sampling(4659 % batch_size)
        self.KLloss = KLDivLoss()


    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        b = visual.shape[0]
        visual_final = self.visual_embedding(visual)
        if b == batch_size:
            visual_, acoustic_ = self.upsample(visual, acoustic)
        else:
            visual_, acoustic_ = self.upsample2(visual, acoustic)
        f1, _, f2, _, f3, _ = self.SharedEncoder(visual_) + self.SharedEncoder(acoustic_) + self.SharedEncoder(text_embedding)
        shared_features = f1 + f2 + f3
        specific_features_audio, (_, _) = self.visualSpecificEncoder(visual_)
        specific_features_text, (_, _) = self.textSpecificEncoder(text_embedding)
        specific_features_visual, (_, _) = self.acousticSpecificEncoder(acoustic_)
        KL1 = self.KLloss(shared_features, specific_features_visual)
        KL2 = self.KLloss(shared_features, specific_features_text)
        KL3 = self.KLloss(shared_features, specific_features_audio)
        for route_iter in range(num_routing - 9):
            visual_final = self.hv(shared_features, specific_features_visual)
            acoustic_final = self.ha(shared_features, specific_features_audio)   # (64,50,768)
            text_embedding_final = self.va(shared_features, specific_features_text)
            if b == batch_size:
                shift = self.multihead_routing2(visual_final, acoustic_final, text_embedding_final)
            else:
                shift = self.routing(visual_final, acoustic_final, text_embedding_final)
            # #1111
            # visual_ = visual_ + residual_1 * shift
            # acoustic_ = acoustic_ + residual_1 * shift
            embedding_shift = text_embedding + shift
        return embedding_shift, (KL1+KL2+KL3)/3

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

class DigitCaps128(nn.Module):
    def __init__(self):
        super(DigitCaps128, self).__init__()
        self.W = nn.Parameter(0.01 * torch.randn(64, 50, 768),
                              requires_grad=True)
    def forward(self, visual, acoustic):
        visual_hat = visual
        acoustic_hat = acoustic
        batch_size = visual.shape[0]
        temp_v = visual_hat.detach()
        temp_a = acoustic_hat.detach()
        b = torch.zeros(batch_size, 1536, 50).to(DEVICE)
        b1 = torch.zeros(batch_size, 768, 50).to(DEVICE)
        b2 = torch.zeros(batch_size, 768, 50).to(DEVICE)
        for route_iter in range(num_routing - 1):
            c = b.softmax(dim=1)
            c1 = c[:, :768, :]
            c2 = c[:, 768:, :]
            s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a)  #768*768
            a =squash(s)
            b1 = b1 + torch.matmul(a, temp_v.transpose(1, 2))
            b2 = b2 + torch.matmul(a, temp_a.transpose(1, 2))
            b = torch.cat((b1, b2), dim=1)
        c = b.softmax(dim=1)
        c1 = c[:, :768, :]
        c2 = c[:, 768:, :]
        s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a)  # 768*768
        a = squash(s)
        return torch.matmul(self.W, a)

class DigitCaps46(nn.Module):
    def __init__(self):
        super(DigitCaps46, self).__init__()
        self.W = nn.Parameter(0.01 * torch.randn(46, 50, 768),
                              requires_grad=True)
    def forward(self, visual, acoustic):
        visual_hat = visual
        acoustic_hat = acoustic
        batch_size = visual.shape[0]
        temp_v = visual_hat.detach()
        temp_a = acoustic_hat.detach()
        b = torch.zeros(batch_size, 1536, 50).to(DEVICE)
        b1 = torch.zeros(batch_size, 768, 50).to(DEVICE)
        b2 = torch.zeros(batch_size, 768, 50).to(DEVICE)
        for route_iter in range(num_routing - 1):
            c = b.softmax(dim = 1)
            c1 = c[:, :768, :]
            c2 = c[:, 768:, :]
            s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a)  #768*768
            a =squash(s)
            b1 = b1 + torch.matmul(a, temp_v.transpose(1, 2))
            b2 = b2 + torch.matmul(a, temp_a.transpose(1, 2))
            b = torch.cat((b1, b2),dim=1)
        c = b.softmax(dim=1)
        c1 = c[:, :768, :]
        c2 = c[:, 768:, :]
        s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a)  # 768*768
        a = squash(s)
        return torch.matmul(self.W, a)
class DigitCapsv2(nn.Module):
    def __init__(self, batch_size):
        super(DigitCapsv2, self).__init__()
        self.size = int(768/routing_head)
        self.W = nn.Parameter(0.01 * torch.randn(batch_size, 50, self.size),
                              requires_grad=True)
    def forward(self, visual, acoustic, va):
        visual_hat = visual
        acoustic_hat = acoustic
        va_hat = va
        temp_v = visual_hat.detach()
        temp_a = acoustic_hat.detach()
        temp_va = va_hat.detach()
        bat = visual.shape[0]
        b = torch.zeros(bat, 3*self.size, 50).to(DEVICE)
        b1 = torch.zeros(bat, self.size, 50).to(DEVICE)
        b2 = torch.zeros(bat, self.size, 50).to(DEVICE)
        b3 = torch.zeros(bat, self.size, 50).to(DEVICE)
        for route_iter in range(num_routing - 1):
            c = b.softmax(dim=1)
            c1 = c[:, :self.size, :]
            c2 = c[:, self.size:2*self.size, :]
            c3 = c[:, 2*self.size:, :]
            s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va) #768*768
            a =squash(s)
            b1 = b1 + torch.matmul(a, temp_v.transpose(1, 2))+residual_2*temp_v.transpose(1, 2)
            b2 = b2 + torch.matmul(a, temp_a.transpose(1, 2))+residual_2*temp_a.transpose(1, 2)
            b3 = b3 + torch.matmul(a, temp_va.transpose(1, 2))+residual_2*temp_va.transpose(1, 2)
            b = torch.cat((b1, b2, b3), dim=1)
        c = b.softmax(dim=1)
        c1 = c[:, :self.size, :]
        c2 = c[:, self.size:2*self.size, :]
        c3 = c[:, 2*self.size:, :]
        s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va)  # 768*768
        a = squash(s)
        return torch.matmul(self.W, a)

#拼接或者拆分：拼接
class multihead_routing(nn.Module):
    def __init__(self, routing_head, batch_size):
        super(multihead_routing, self).__init__()
        self.num_head = routing_head
        self.batch_size = batch_size
        self.routing = DigitCapsv2(self.batch_size)
        self.scale = nn.Linear(768, 768)
    def forward(self, visual, acoustic, va):
        tensors = []
        for i in range(self.num_head):
            temp_ = self.routing(visual, acoustic, va)
            tensors.append(temp_)
        multi_fea = torch.cat(tensors, dim=-1)
        final_ = self.scale(multi_fea)
        return final_
#拆分
class multihead_routing2(nn.Module):
    def __init__(self, routing_head, batch_size):
        super(multihead_routing2, self).__init__()
        self.num_head = routing_head
        self.batch_size = batch_size
        self.routing = DigitCapsv2(self.batch_size)
        self.scale = nn.Linear(768, 768)
        self.size = int(768/routing_head)

    def forward(self, visual, acoustic, va):
        visual_ = torch.split(visual, self.size, dim=2)
        acoustic_ = torch.split(acoustic, self.size, dim=2)
        va_ = torch.split(va, self.size, dim=2)
        tensors = []
        for i in range(self.num_head):
            temp_ = self.routing(visual_[i], acoustic_[i], va_[i])
            tensors.append(temp_)
        multi_fea = torch.cat(tensors, dim=-1)
        final_ = self.scale(multi_fea)
        return final_
class DigitCapsv2_10(nn.Module):
    def __init__(self):
        super(DigitCapsv2_10, self).__init__()
        self.W = nn.Parameter(0.01 * torch.randn(10, 50, 768),
                              requires_grad=True)
    def forward(self, visual, acoustic, va):
        visual_hat = visual
        acoustic_hat = acoustic
        va_hat = va
        temp_v = visual_hat.detach()
        temp_a = acoustic_hat.detach()
        temp_va = va_hat.detach()
        bat = visual.shape[0]
        b = torch.zeros(bat, 2304, 50).to(DEVICE)
        b1 = torch.zeros(bat, 768, 50).to(DEVICE)
        b2 = torch.zeros(bat, 768, 50).to(DEVICE)
        b3 = torch.zeros(bat, 768, 50).to(DEVICE)
        for route_iter in range(num_routing - 1):
            c = b.softmax(dim = 1)
            c1 = c[:, :768, :]
            c2 = c[:, 768:1536, :]
            c3 = c[:, 1536:, :]
            s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va) #768*768
            a =squash(s)
            b1 = b1 + torch.matmul(a, temp_v.transpose(1, 2))
            b2 = b2 + torch.matmul(a, temp_a.transpose(1, 2))
            b3 = b3 + torch.matmul(a, temp_va.transpose(1, 2))
            b = torch.cat((b1, b2, b3), dim=1)
        c = b.softmax(dim=1)
        c1 = c[:, :768, :]
        c2 = c[:, 768:1536, :]
        c3 = c[:, 1536:, :]
        s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va)  # 768*768
        a = squash(s)
        return torch.matmul(self.W, a)

class DigitCapsv3(nn.Module):
    def __init__(self):
        super(DigitCapsv3, self).__init__()
        self.W = nn.Parameter(0.01 * torch.randn(128, 50, 768),
                              requires_grad=True)
    def forward(self, visual, acoustic, va):
        visual_hat = visual
        acoustic_hat = acoustic
        va_hat = va
        temp_v = visual_hat.detach()
        temp_a = acoustic_hat.detach()
        temp_va = va_hat.detach()
        bat = visual.shape[0]
        b = torch.zeros(bat, 2304, 50).to(DEVICE)
        b1 = torch.zeros(bat, 768, 50).to(DEVICE)
        b2 = torch.zeros(bat, 768, 50).to(DEVICE)
        b3 = torch.zeros(bat, 768, 50).to(DEVICE)
        for route_iter in range(num_routing - 1):
            c = b.softmax(dim = 1)
            c1 = c[:, :768, :]
            c2 = c[:, 768:1536, :]
            c3 = c[:, 1536:, :]
            s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va) #768*768
            a =squash(s)
            b1 = b1 + torch.matmul(a, temp_v.transpose(1, 2))+temp_v.transpose(1, 2)
            b2 = b2 + torch.matmul(a, temp_a.transpose(1, 2))+temp_a.transpose(1, 2)
            b3 = b3 + torch.matmul(a, temp_va.transpose(1, 2))+temp_va.transpose(1, 2)
            b = torch.cat((b1, b2, b3), dim=1)
        c = b.softmax(dim=1)
        c1 = c[:, :768, :]
        c2 = c[:, 768:1536, :]
        c3 = c[:, 1536:, :]
        s = torch.matmul(c1, temp_v) + torch.matmul(c2, temp_a) + torch.matmul(c3, temp_va)  # 768*768
        a = squash(s)
        return torch.matmul(self.W, a)
class Attention(nn.Module):
    def __init__(self, text_dim):
        super(Attention, self).__init__()
        self.text_dim = text_dim
        self.dim = text_dim 
        self.Wq = nn.Linear(text_dim, text_dim)
        self.Wk = nn.Linear(self.dim, text_dim)
        self.Wv = nn.Linear(self.dim, text_dim)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        tmp = torch.matmul(Q, K.transpose(-1, -2) * math.sqrt(self.text_dim))[0]
        weight_matrix = F.softmax(torch.matmul(Q, K.transpose(-1, -2) * math.sqrt(self.text_dim)), dim=-1)

        return torch.matmul(weight_matrix, V)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
