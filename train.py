import torch
from utils import get_data, TrainModel
import torch.nn.functional as F
from model import MuJoD
from exid import ExID
from tqdm import tqdm
import numpy as np
import random

# 超参数
lr = 0.0005
epochs = 2000
e_dim = 256 # 三元组模态实体嵌入维度
r_dim = 256 # 所有模态的关系嵌入维度
dataset_path = '/kaggle/input/momok-dataset/DB15K/'
batch_size = 1024
K = 3 # 专家数
mu = 0.0001
eval_freq = 100
sparse_ratio = 0.20 # 稀疏场景舍弃的数据比例

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# 读取数据
e2id, r2id, img_features, text_features, train_triples, test_triples = get_data(dataset_path, sparse_ratio)

print("训练集三元组数 ",len(train_triples))

train_model = TrainModel(device, batch_size, train_triples, test_triples, e2id, r2id)

# 图片和文本特征L2归一化
img_features = F.normalize(torch.Tensor(img_features), p=2, dim=1)
text_features = F.normalize(torch.Tensor(text_features), p=2, dim=1)

# MuJoD学习嵌入
mujod_model = MuJoD(device, e2id, r2id, e_dim, r_dim, img_features, text_features, K)

img_e_dim = mujod_model.img_e_dim # 图像模态嵌入维度 256
txt_e_dim = mujod_model.txt_e_dim # 文本模态嵌入维度 256

# ExID优化模型拟合和模态内专家去相关性
exid_model = ExID(n_exp=K, s_e_dim=e_dim, i_e_dim=img_e_dim, t_e_dim=txt_e_dim)

# 两个优化器
mujod_optimizer = torch.optim.Adam(params=mujod_model.parameters(), lr=lr, weight_decay=0)
exid_optimizer = torch.optim.Adam(params=exid_model.parameters(), lr=lr, weight_decay=0)
mujod_model = mujod_model.to(device)
exid_model = exid_model.to(device)

# 记录最佳结果
best_test_result = {"Hits@10": -1, "Hits@3": -1, "Hits@1": -1,"MRR": -1}

# 训练 需要训练mujod_model和exid_model两个模型
train_epochs = tqdm(range(epochs))
for epoch in train_epochs:
    epoch_mujod_loss = []
    epoch_exid_loss = []
    mujod_model.train()
    np.random.shuffle(train_model.train_data)
    for batch_i in range(train_model.batch_num):
        # 训练mujod_model  将exid_model设为评估模式不更新
        exid_model.eval()
        mujod_optimizer.zero_grad()
        # 获取batch的训练数据
        batch_triples, batch_values = train_model.get_batch(batch_i) 
        batch_triples = torch.LongTensor(batch_triples)
        batch_triples = batch_triples.to(device)
        batch_values = batch_values.to(device)
        pred_sco, experts_embs, modal_embs = mujod_model.forward(batch_triples)
        mujod_loss = mujod_model.compute_total_loss(pred_sco, batch_values, modal_embs ) + mu * exid_model(experts_embs) # 提高预测准度以及降低模态内专家互信息
        mujod_loss.backward()
        mujod_optimizer.step()

        # 训练exid_model
        exid_model.train()
        exid_optimizer.zero_grad()
        with torch.no_grad():
            exps_embs = mujod_model.get_batch_embs(batch_triples)
        exid_loss = exid_model.loss_exid(exps_embs)
        exid_loss.backward()
        exid_optimizer.step()
        epoch_mujod_loss.append(mujod_loss.item())
        epoch_exid_loss.append(exid_loss.item())
    train_epochs.set_postfix(loss="mujod: {:.5} exid: {:.5}".format(sum(epoch_mujod_loss), sum(epoch_exid_loss)))

    # 每eval_freq轮在测试集上测试一次
    if (epoch + 1) % eval_freq == 0:
        mujod_model.eval()
        with torch.no_grad():
            result = train_model.get_pred_result(mujod_model)
        if result['MRR'] > best_test_result['MRR']:
            best_test_result['MRR'] = result['MRR']
        if result['Hits@1'] > best_test_result['Hits@1']:
            best_test_result['Hits@1'] = result['Hits@1']
        if result['Hits@3'] > best_test_result['Hits@3']:
            best_test_result['Hits@3'] = result['Hits@3']
        if result['Hits@10'] > best_test_result['Hits@10']:
            best_test_result['Hits@10'] = result['Hits@10']
        print(f"Epoch {epoch + 1:04d} | " + " | ".join([f"{k}: {v:.4f}" for k, v in result.items()]))

print("best_test_result:")
print(f" | ".join([f"{k}: {v:.4f}" for k, v in best_test_result.items()])) 

torch.save(mujod_model.state_dict(), f'mujod_model.pth')
print('Saved model!')