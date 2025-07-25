import torch
import torch.nn as nn
import random

class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, m_dim):
        super(CLUB, self).__init__()
        # 计算均值μ
        self.pyx_mu = nn.Sequential(nn.Linear(x_dim, m_dim//2),nn.ReLU(),nn.Linear(m_dim//2, y_dim))
        # 计算对数方差lv
        self.pyx_lv = nn.Sequential(nn.Linear(x_dim, m_dim//2),nn.ReLU(),nn.Linear(m_dim//2, y_dim),nn.Tanh())

    def get_mu_lv(self, x): 
        mu = self.pyx_mu(x)
        lv = self.pyx_lv(x)
        return mu,lv

    # 对数似然值
    def loglike(self, x, y):
        mu, lv = self.get_mu_lv(x)
        return (-(mu - y)**2 / 2. / lv.exp()).sum(dim=1).mean(dim=0)

    # 互信息降低loss
    def forward(self, x, y): # 互信息降低loss
        mu, lv = self.get_mu_lv(x)
        sam_num = x.shape[0]
        r_idx = torch.randperm(sam_num).long() # 打乱索引作为负样本
        pos_v = - (mu - y)**2 / lv.exp() # 正
        neg_v = - (mu - y[r_idx])**2 / lv.exp() # 负
        # 上界，正样本-负样本似然值 越小说明互信息越少
        u_bound = (pos_v.sum(dim = -1) - neg_v.sum(dim = -1)).mean() 
        return u_bound / 2.0 
    
    # 分布拟合loss
    def train_loss(self, x, y):
        return - self.loglike(x, y) 



# 模态内专家去互信息
class ExID(nn.Module):
    def __init__(self, n_exp, s_e_dim, i_e_dim, t_e_dim):
        super(ExID, self).__init__()
        self.s_club = CLUB(s_e_dim, s_e_dim, s_e_dim)
        self.i_club = CLUB(i_e_dim, i_e_dim, i_e_dim)
        self.t_club = CLUB(t_e_dim, t_e_dim, t_e_dim)
        self.n_exp = n_exp

    def forward(self, embs): # 专家去互信息
        # loss_club
        s_embs, i_embs, t_embs = embs
        _, n_exp, _ = s_embs.size() # [B, n_exp, dim]
        exp_idx1, exp_idx2 = random.sample(range(n_exp), k=2) # 随机两个专家
        s_emb1, s_emb2 = s_embs[:, exp_idx1, :], s_embs[:, exp_idx2, :]
        i_emb1, i_emb2 = i_embs[:, exp_idx1, :], i_embs[:, exp_idx2, :]
        t_emb1, t_emb2 = t_embs[:, exp_idx1, :], t_embs[:, exp_idx2, :]
        loss = (self.s_club(s_emb1, s_emb2) + self.i_club(i_emb1, i_emb2) + self.t_club(t_emb1, t_emb2) ) / 3.0 
        return loss

    # 提高分布拟合
    def loss_exid(self, embs):
        s_embs, i_embs, t_embs = embs
        _, n_exp, _ = s_embs.size() # [B, n_exp, dim]
        exp_idx1, exp_idx2 = random.sample(range(n_exp), k=2) # 随机两个专家
        s_emb1, s_emb2 = s_embs[:, exp_idx1, :], s_embs[:, exp_idx2, :]
        i_emb1, i_emb2 = i_embs[:, exp_idx1, :], i_embs[:, exp_idx2, :]
        t_emb1, t_emb2 = t_embs[:, exp_idx1, :], t_embs[:, exp_idx2, :]
        loss = (self.s_club.train_loss(s_emb1, s_emb2) + self.i_club.train_loss(i_emb1, i_emb2) + self.t_club.train_loss(t_emb1, t_emb2) ) / 3.0
        return loss