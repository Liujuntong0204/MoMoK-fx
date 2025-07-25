import torch
import torch.nn as nn
import torch.nn.functional as F

# 专家模型，从输入嵌入学习一个输出嵌入
class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True) # 偏置项
        self.linear = nn.Linear(input_size, output_size, bias=False) # 线性变化 input_size -> output_size
        self.linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, embs):
        return self.linear(embs - self.bias) 


# 关系引导模态知识专家，针对每个模态
class ReMoKE(nn.Module):
    def __init__(self, n_exps, layer_size):
        super(ReMoKE, self).__init__()
        self.n_exps = n_exps
        self.experts = nn.ModuleList([Expert(layer_size[0], layer_size[1]) for i in range(n_exps)])
        self.exps_T = nn.Parameter(torch.zeros(layer_size[0], n_exps), requires_grad=True) # 专家权重
        self.exps_noise_T = nn.Parameter(torch.zeros(layer_size[0], n_exps), requires_grad=True) # 噪声

    def get_exps_weight(self, embs, rel_gate=None, is_train=None):
        embs_exps_sco = embs @ self.exps_T # [embs_n, n_exps] 嵌入的专家的得分
        if is_train: # 噪声
            noise_epsilon = 1e-2
            noises = embs @ self.exps_noise_T 
            noise_e = ((F.softplus(noises) + noise_epsilon)) 
            noise_sco = embs_exps_sco + (torch.randn_like(embs_exps_sco).to(embs.device) * noise_e) 
            scores = noise_sco
        else:
            scores = embs_exps_sco
        if rel_gate is not None:
            exps_sco = F.softmax(scores / torch.sigmoid(rel_gate), dim=-1) 
        else:
            exps_sco = F.softmax(scores, dim=-1)
        return exps_sco # 嵌入 每个专家权重 [embs_n, n_exps]
    
    def forward(self, embs, rel_gate=None):
        exps_sco = self.get_exps_weight(embs, rel_gate, self.training) # [embs_n, n_exps]
        expert_outputs_embs = [self.experts[i](embs).unsqueeze(-2) for i in range(self.n_exps)] # 输出嵌入[embs_n, 1, dim]
        expert_outputs_embs = torch.cat(expert_outputs_embs, dim=-2) # [embs_n, n_exps, output_dim] 
        w_outputs_embs = exps_sco.unsqueeze(-1) * expert_outputs_embs 
        multi_outputs_embs = w_outputs_embs.sum(dim=-2) # 权重相加，输出嵌入 [embs_n, output_dim]
        return multi_outputs_embs, expert_outputs_embs 


# 注意力融合，将三个模态嵌入得到融合模态嵌入
class ModalFusionJ(nn.Module):
    def __init__(self, multi_head_num, s_dim, img_dim, txt_dim, out_dim):
        super(ModalFusionJ, self).__init__()
        self.multi_head_num = multi_head_num
        self.s_dim = s_dim
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.out_dim = out_dim
        
        # 三元组模态 s_dim 
        modal1_model = []
        for i in range(self.multi_head_num):
            dropout = nn.Dropout(p=0.2)
            linear = nn.Linear(self.s_dim , self.out_dim)
            modal1_model.append(nn.Sequential(dropout,linear,nn.ReLU()))
        self.modal1 = nn.ModuleList(modal1_model)

        # 图像模态 img_dim 
        modal2_model = []
        for i in range(self.multi_head_num):
            dropout = nn.Dropout(p=0.2)
            linear = nn.Linear(self.img_dim , self.out_dim)
            modal2_model.append(nn.Sequential(dropout,linear,nn.ReLU()))
        self.modal2 = nn.ModuleList(modal2_model)

        # 文本模态 txt_dim 
        modal3_model = []
        for i in range(self.multi_head_num):
            dropout = nn.Dropout(p=0.2)
            linear = nn.Linear(self.txt_dim , self.out_dim)
            modal3_model.append(nn.Sequential(dropout,linear,nn.ReLU()))
        self.modal3 = nn.ModuleList(modal3_model)

        # 注意力 融合权重
        self.attention = nn.Linear(self.out_dim, 1, bias=False)
        self.attention.requires_grad_(True)
     
    def forward(self, emb1, emb2, emb3): 
        embs_num = emb1.size(0)
        embs = []
        for i in range(self.multi_head_num):
            m1_embs = self.modal1[i](emb1)
            m2_embs = self.modal2[i](emb2)
            m3_embs = self.modal3[i](emb3)
            x_stack = torch.stack((m1_embs, m2_embs, m3_embs), dim=1)
            atte_scos = self.attention(x_stack).squeeze(-1)
            atte_weights = torch.softmax(atte_scos, dim=-1)
            head_embs = torch.sum(atte_weights.unsqueeze(-1) * x_stack, dim=1)
            embs.append(head_embs) 
        embs = torch.stack(embs, dim=1)
        embs = embs.sum(1).view(embs_num, self.out_dim) 
        return embs # 融合后的嵌入[embs_num, output_dims]


# 预测模型 h,r -> t
class PredTail(nn.Module): 
    def __init__(self, e_dim, r_dim):
        super(PredTail, self).__init__()
        self.r_T = nn.Parameter(torch.rand(r_dim, e_dim, e_dim)) # 关系张量 [r_dim, e_dim, e_dim]
        nn.init.xavier_uniform_(self.r_T.data)
        self.bn_input = nn.BatchNorm1d(e_dim)
        self.bn_output = nn.BatchNorm1d(e_dim)
        self.dropout_input = nn.Dropout(0.3)
        self.dropout_mid = nn.Dropout(0.4)
        self.dropout_output = nn.Dropout(0.5)

    def forward(self, e_embs, r_embs): 
        e = self.bn_input(e_embs)
        e = self.dropout_input(e)
        e = e.view(-1, 1, e.size(1)) # [num, 1, e_dim]
        
        # 关系矩阵
        # [r_num,r_dim] * [r_dim, e_dim * e_dim]  -> [r_num, e_dim * e_dim] 
        r = torch.mm(r_embs, self.r_T.view(r_embs.size(1), -1)) 
        r = r.view(-1, e.size(2), e.size(2)) # [r_num, e_dim, e_dim]
        r = self.dropout_mid(r)
       
        e_out = torch.bmm(e, r) 
        e_out = e_out.view(-1, e_out.size(2)) 
        e_out = self.bn_output(e_out)
        e_out = self.dropout_output(e_out)
        return e_out # 尾实体嵌入

# 总模型
class MuJoD(nn.Module):
    def __init__(self, device, entity2id, relation2id, e_dim, r_dim, img_features, text_features, K):
        super(MuJoD, self).__init__()
        self.device = device

        # 三元组模态 
        self.s_entity_embs = nn.Embedding(
            len(entity2id),
            e_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.s_entity_embs.weight)
        self.s_relation_embs = nn.Embedding(
            2 * len(relation2id), 
            r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.s_relation_embs.weight)

        # 图像模态 
        # DB15K数据集 转为256维
        img_pool = torch.nn.AvgPool2d(4, stride=4) # 池化处理 降维压缩 原为4096维 
        img = img_pool(img_features.to(self.device).view(-1, 64, 64)) # batchsize,64,64 池化后变为 batchsize,16,16
        img = img.view(img.size(0), -1) # batchsize, 16*16  == batchsize,256
        self.img_entity_embs = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embs = nn.Embedding(
            2 * len(relation2id),
            r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embs.weight)

        # 文本模态 
        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64)) # 4,64
        txt = txt_pool(text_features.to(self.device).view(-1, 12, 64)) # batchsize,12,64 池化后变为 batchsize,4,16
        txt = txt.view(txt.size(0), -1) # batchsize,4*64 == batchsize,256
        self.txt_entity_embs = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embs = nn.Embedding(
            2 * len(relation2id),
            r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embs.weight)

        # 关系引导
        self.rel_gate = nn.Embedding(2 * len(relation2id), 1, padding_idx=None) 

        # 记一下维度
        self.s_e_dim = e_dim # 三元组模态实体维度
        self.r_dim = r_dim # 所有模态的关系维度
        self.img_e_dim = self.img_entity_embs.weight.data.shape[1] # 图像模态实体维度
        self.txt_e_dim = self.txt_entity_embs.weight.data.shape[1] # 文本模态实体维度
        self.j_e_dim = e_dim # 融合模态实体维度

        # 每种模态下多专家学习嵌入
        self.s_remoke = ReMoKE(n_exps=K, layer_size=[self.s_e_dim, self.s_e_dim]) # 三元组
        self.i_remoke = ReMoKE(n_exps=K, layer_size=[self.img_e_dim, self.img_e_dim]) # 图像
        self.t_remoke = ReMoKE(n_exps=K, layer_size=[self.txt_e_dim, self.txt_e_dim]) # 文本
        # 混合模态
        self.e_fusion = ModalFusionJ(multi_head_num=2, s_dim=self.s_e_dim, img_dim=self.img_e_dim, txt_dim=self.txt_e_dim, out_dim=self.j_e_dim)
        self.r_fusion = ModalFusionJ(multi_head_num=2, s_dim=self.r_dim, img_dim=self.r_dim, txt_dim=self.r_dim, out_dim=self.r_dim)
        
        # 每种模态预测模型
        self.s_predtail = PredTail(self.s_e_dim, self.r_dim)
        self.i_predtail = PredTail(self.img_e_dim, self.r_dim)
        self.t_predtail = PredTail(self.txt_e_dim, self.r_dim)
        self.j_predtail = PredTail(self.j_e_dim, self.r_dim) 

        # 二元交叉熵损失函数
        self.bceloss = nn.BCELoss() 

    def forward(self,batch_data):
        source_es = batch_data[:, 0]
        relations = batch_data[:, 1]
        rel_gate = self.rel_gate(relations) 

        # 模态嵌入
        # 三元组模态
        s_e_embs, s_e_experts_embs = self.s_remoke(self.s_entity_embs(source_es), rel_gate)
        s_r_embs = self.s_relation_embs(relations)
        # 图像模态
        i_e_embs, i_e_experts_embs = self.i_remoke(self.img_entity_embs(source_es), rel_gate)
        i_r_embs = self.img_relation_embs(relations)
        # 文本模态
        t_e_embs, t_e_experts_embs = self.t_remoke(self.txt_entity_embs(source_es), rel_gate)
        t_r_embs = self.txt_relation_embs(relations)
        # 融合模态
        j_e_embs = self.e_fusion(s_e_embs, i_e_embs, t_e_embs)
        j_r_embs = self.r_fusion(s_r_embs, i_r_embs, t_r_embs)

        # 预测的尾实体嵌入
        s_pred_tail = self.s_predtail(s_e_embs, s_r_embs)
        i_pred_tail = self.i_predtail(i_e_embs, i_r_embs)
        t_pred_tail = self.t_predtail(t_e_embs, t_r_embs)
        j_pred_tail = self.j_predtail(j_e_embs, j_r_embs)

        # 标准嵌入
        s_e_embs_all, _ = self.s_remoke(self.s_entity_embs.weight)
        i_e_embs_all, _ = self.i_remoke(self.img_entity_embs.weight)
        t_e_embs_all, _= self.t_remoke(self.txt_entity_embs.weight)
        s_e_embs_all = self.e_fusion(s_e_embs_all, i_e_embs_all, t_e_embs_all)

        # 点积得分
        s_pred_sco = torch.mm(s_pred_tail, s_e_embs_all.transpose(1, 0)) # [B, dim] * [dim, N] -> [B, N]
        i_pred_sco = torch.mm(i_pred_tail, i_e_embs_all.transpose(1, 0))
        t_pred_sco = torch.mm(t_pred_tail, t_e_embs_all.transpose(1, 0))
        j_pred_sco = torch.mm(j_pred_tail, s_e_embs_all.transpose(1, 0))
        s_pred_sco = torch.sigmoid(s_pred_sco)
        i_pred_sco = torch.sigmoid(i_pred_sco)
        t_pred_sco = torch.sigmoid(t_pred_sco)
        j_pred_sco = torch.sigmoid(j_pred_sco)

        return [s_pred_sco, i_pred_sco, t_pred_sco, j_pred_sco], [s_e_experts_embs, i_e_experts_embs, t_e_experts_embs]

    # 获取所有专家学习嵌入 用于拟合嵌入分布
    def get_batch_embs(self, batch_data): 
        s_e = batch_data[:, 0]
        _, s_e_exps_embs = self.s_remoke(self.s_entity_embs(s_e))
        _, i_e_exps_embs = self.i_remoke(self.img_entity_embs(s_e))
        _, t_e_exps_embs = self.t_remoke(self.txt_entity_embs(s_e))
        return [s_e_exps_embs, i_e_exps_embs, t_e_exps_embs]

    def loss_kgc(self, pred, target):
        loss_s = self.bceloss(pred[0], target) # [B, N] [B, N]
        loss_i = self.bceloss(pred[1], target)
        loss_t = self.bceloss(pred[2], target)
        loss_j = self.bceloss(pred[3], target)
        return loss_s + loss_i + loss_t + loss_j
