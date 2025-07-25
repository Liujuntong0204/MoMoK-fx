# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# from tqdm import tqdm
# # 设置种子
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

import torch
import numpy as np
import random

# 获取数据
def get_data(path, sparse_ratio):
    e2id = {}
    with open(path + 'entity2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            e2id[instance[0]] = int(instance[1])
    r2id = {}
    with open(path + 'relation2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            r2id[instance[0]] = int(instance[1])  

    # train     
    train_triples = []
    with open(path+'train.txt', 'r', encoding='utf-8') as f: 
        for line in f:
            instance = line.strip().split(' ')
            e1, r, e2 = instance[0], instance[1], instance[2]  
            train_triples.append((e2id[e1], r2id[r], e2id[e2]))

    # 稀疏处理
    total = len(train_triples)
    keep_num = int(total * (1 - sparse_ratio))
    train_triples = random.sample(train_triples, keep_num)

    # test
    test_triples = []
    with open(path+'test.txt', 'r', encoding='utf-8') as f: 
        for line in f:
            instance = line.strip().split(' ')
            e1, r, e2 = instance[0], instance[1], instance[2]  
            test_triples.append((e2id[e1], r2id[r], e2id[e2]))

    # 图像和文本特征
    img_features = torch.load(open(path+'img_features.pth', 'rb'))
    text_features = torch.load(open(path+'text_features.pth', 'rb'))

    return e2id, r2id, img_features, text_features, train_triples, test_triples



class TrainModel:
    def __init__(self, device, batch_size, train_triples, test_triples, e2id, r2id):
        self.device = device
        self.batch_size = batch_size
        self.train_triples = train_triples
        self.test_triples = test_triples
        self.entity2id = {k: v for k, v in e2id.items()}
        self.relation2id = {k: v for k, v in r2id.items()}

        # 添加逆关系 全部改为 s+r->t 形式
        r_num = len(r2id)
        for k, v in r2id.items():
            self.relation2id[k+'_reverse'] = v+r_num

        # 训练集
        # h+r -> set(t)
        hr2t = {}
        for (h, r, t) in self.train_triples:
            if (h, r) not in hr2t.keys():
                hr2t[(h, r)] = set()
            if (t, r+r_num) not in hr2t.keys():
                hr2t[(t, r+r_num)] = set()
            hr2t[(h, r)].add(t)
            hr2t[(t, r+r_num)].add(h)
        self.train_data = [{'triple': (h, r, -1), 'label': list(hr2t[(h, r)])}
                              for (h, r), t in hr2t.items()] # triple:题目 label:set(tail)

        # 测试集
        for (h, r, t) in self.test_triples:
            if (h, r) not in hr2t.keys():
                hr2t[(h, r)] = set()
            if (t, r+r_num) not in hr2t.keys():
                hr2t[(t, r+r_num)] = set()
            hr2t[(h, r)].add(t)
            hr2t[(t, r+r_num)].add(h)

        # 测试集
        self.test_head_data = [{'triple': (t, r+r_num, h), 'label': list(hr2t[(t, r+r_num)])}
                                 for (h, r, t) in self.test_triples]
        self.test_tail_data = [{'triple': (h, r, t), 'label': list(hr2t[(h, r)])}
                                 for (h, r, t) in self.test_triples]
        
        # 计算轮数
        if len(self.train_data) % self.batch_size == 0:
            self.batch_num = len(self.train_data) // self.batch_size
        else:
            self.batch_num = len(self.train_data) // self.batch_size + 1

    # 获取当前batch的训练数据
    def get_batch(self, batch_i): 
        if (batch_i + 1) * self.batch_size <= len(self.train_data):
            batch_data = self.train_data[batch_i * self.batch_size: (batch_i+1) * self.batch_size]
        else:
            batch_data = self.train_data[batch_i * self.batch_size:]
        batch_triples = torch.LongTensor([item['triple'] for item in batch_data]) 
        labels = [np.int32(item['label']) for item in batch_data] 
        # 矩阵形式 标准值
        label_mtx = np.zeros((len(batch_data), len(self.entity2id)), dtype=np.float32)
        for idx in range(len(labels)):
            for e in labels[idx]:
                label_mtx[idx][e] = 1.0
        label_mtx = 0.9 * label_mtx + (1.0 / len(self.entity2id))  
        batch_values = torch.FloatTensor(label_mtx)
        return batch_triples, batch_values
    
    # 测试，计算性能指标
    def get_pred_result(self,model): 
        head_ranks = [] 
        head_r_ranks = [] 
        tail_ranks = [] 
        tail_r_ranks = [] 

        head_data = self.test_head_data
        tail_data = self.test_tail_data

        if len(head_data) % self.batch_size == 0:  
            batch_n = len(head_data) // self.batch_size
        else:
            batch_n = len(head_data) // self.batch_size + 1

        for batch_i in range(batch_n):
            if (batch_i + 1) * self.batch_size <= len(head_data):
                head_batch = head_data[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                tail_batch = tail_data[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
            else:
                head_batch = head_data[batch_i * self.batch_size:]
                tail_batch = tail_data[batch_i * self.batch_size:]

            # 头预测 ( ? , r , t )
            head_batch_triples = torch.LongTensor([item['triple'] for item in head_batch])
            head_batch_triples = head_batch_triples.to(self.device)

            pred_head_s, _ = model.forward(head_batch_triples) # 预测结果
            pred_head = (pred_head_s[0] + pred_head_s[1] + pred_head_s[2] + pred_head_s[3]) / 4.0  

            label = [np.int32(item['label']) for item in head_batch]
            label_trx = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32) # [batch_size, num_entities] 
            for idx in range(len(label)):
                for l in label[idx]:
                    label_trx[idx][l] = 1.0
            label_trx = torch.FloatTensor(label_trx).to(self.device)

            # 预测目标得分
            target = head_batch_triples[:, 2] 
            index = torch.arange(pred_head.shape[0], device=self.device)
            target_pred = pred_head[index, target] # 目标实体预测得分
            pred_head = torch.where(label_trx.bool(), torch.zeros_like(pred_head), pred_head) # 将所有的label置0
            pred_head[index, target] = target_pred # 赋值目标实体分数
            pred_head = pred_head.cpu().numpy()
            target = target.cpu().numpy()

            for i in range(pred_head.shape[0]):
                scores = pred_head[i] # [num_entities]
                tar = target[i] # 目标实体
                sorted_idxs = np.argsort(-scores, kind='stable') 
                rank = np.where(sorted_idxs == tar)[0][0] + 1 
                head_ranks.append(rank) 
                head_r_ranks.append(1.0 / rank)

            # 尾预测 ( s , r , ? )
            tail_batch_triples = torch.LongTensor([item['triple'] for item in tail_batch])
            tail_batch_triples = tail_batch_triples.to(self.device)

            pred_tail_s, _ = model.forward(tail_batch_triples) # 预测结果
            pred_tail = (pred_tail_s[0] + pred_tail_s[1] + pred_tail_s[2] + pred_tail_s[3]) / 4.0  

            label = [np.int32(item['label']) for item in tail_batch]
            label_trx = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32) # [batch_size, num_entities] 
            for idx in range(len(label)):
                for l in label[idx]:
                    label_trx[idx][l] = 1.0
            label_trx = torch.FloatTensor(label_trx).to(self.device)

            target = tail_batch_triples[:, 2] 
            index = torch.arange(pred_tail.shape[0], device=self.device)
            target_pred = pred_tail[index, target] # 目标实体预测得分

            pred_tail = torch.where(label_trx.bool(), torch.zeros_like(pred_tail), pred_tail) # 将所有的label置0
            pred_tail[index, target] = target_pred # 赋值目标实体分数
            pred_tail = pred_tail.cpu().numpy()
            target = target.cpu().numpy()

            for i in range(pred_tail.shape[0]): 
                scores = pred_tail[i]
                tar = target[i] 
                sorted_idxs = np.argsort(-scores, kind='stable')
                rank = np.where(sorted_idxs == tar)[0][0]+1
                tail_ranks.append(rank) 
                tail_r_ranks.append(1.0 / rank)

        # 所有batch得分得到后
        hits_at_10_head = 0
        hits_at_3_head = 0
        hits_at_1_head = 0
        for i in range(len(head_ranks)):
            if head_ranks[i] <= 10:
                hits_at_10_head +=1
            if head_ranks[i] <= 3:
                hits_at_3_head +=1 
            if head_ranks[i] == 1:
                hits_at_1_head +=1  

        hits_at_10_tail = 0
        hits_at_3_tail = 0
        hits_at_1_tail = 0
        for i in range(len(tail_ranks)):
            if tail_ranks[i] <= 10:
                hits_at_10_tail +=1
            if tail_ranks[i] <= 3:
                hits_at_3_tail +=1 
            if tail_ranks[i] == 1:
                hits_at_1_tail +=1   

        hits_at_10 = (hits_at_10_head / len(head_ranks) + hits_at_10_tail / len(tail_ranks)) / 2
        hits_at_3 = (hits_at_3_head / len(head_ranks) + hits_at_3_tail / len(tail_ranks)) / 2
        hits_at_1 = (hits_at_1_head / len(head_ranks) + hits_at_1_tail / len(tail_ranks)) / 2
        mean_r_rank = (sum(head_r_ranks) / len(head_r_ranks) + sum(tail_r_ranks) / len(tail_r_ranks)) / 2

        result = {"Hits@10": hits_at_10, "Hits@3": hits_at_3, "Hits@1": hits_at_1, "MRR": mean_r_rank }
        return result
