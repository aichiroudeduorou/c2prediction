import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.spatial.distance import cdist
import os, json, random
import time

# ==================== 1. 数据预处理 ====================
def load_data(train_path, test_path):
    """加载并预处理数据集"""
    def preprocess(csv_path, scaler=None, encoders=None, is_train=False):
        df = pd.read_csv(csv_path, header=0, na_values=' ?', skipinitialspace=True)
        # df.columns = [
        #     'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        #     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        #     'hours-per-week', 'native-country', 'income'
        # ]
        df.dropna(inplace=True)
        
        # 目标变量
        # 获取最后一列的列名
        last_col = df.columns[-1]

        # 删除最后一列
        y = df.iloc[:, -1]
        df.drop(columns=[last_col], axis=1, inplace=True)
        # df.drop(['income', 'fnlwgt'], axis=1, inplace=True)  # 移除无关特征for adult
        
        # 特征分类

        # adult
        # num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        # cat_cols = [c for c in df.columns if c not in num_cols]

        # synthetic
        cat_cols = []
        num_cols = df.columns.tolist()
        
        # 数值特征标准化
        if is_train:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(df[num_cols].values)
        else:
            X_num = scaler.transform(df[num_cols].values)
        
        # 类别特征编码
        if cat_cols:
            if is_train:
                encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in cat_cols}
                X_cat = np.column_stack([
                    encoders[col].transform(df[col].astype(str)) for col in cat_cols
                ])
            else:
                X_cat = np.column_stack([
                    encoders[col].transform(df[col].astype(str)) for col in cat_cols
                ])
            X = np.hstack([X_num, X_cat])
        else:
            if is_train:
                encoders = {}
            X = X_num
        feature_names = num_cols + cat_cols
        cat_mask = np.array([False]*len(num_cols) + [True]*len(cat_cols))
        
        return X.astype(np.float32), y.astype(np.int64).values, scaler, encoders, feature_names, cat_mask
    
    # 训练集
    X_train, y_train, scaler, encoders, feature_names, cat_mask = preprocess(train_path, is_train=True)
    # 测试集
    X_test, y_test, _, _, _, _ = preprocess(test_path, scaler=scaler, encoders=encoders, is_train=False)
    
    return (X_train, y_train), (X_test, y_test), feature_names, cat_mask, scaler

# ==================== 2. 核心模型定义 ====================
class CausalDiscovery(nn.Module):
    """因果发现模块：MLP 生成因果概率图 + Gumbel-Softmax 采样"""
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_features * n_features)
        )
        self.tau = 1.0  # Gumbel-Softmax 温度（训练时从1.0退火到0.1）
    
    def forward(self, X):
        """
        X: [batch_size, n_features]
        返回: P (因果概率图), G_hat (采样得到的二值因果图)
        """
        # 特征级聚合：对 batch 取均值作为输入
        theta = self.mlp(X.mean(dim=0, keepdim=True)).view(self.n_features, self.n_features)
        P = torch.sigmoid(theta)
        P = P * (1 - torch.eye(self.n_features, device=P.device))  # 移除自环
        
        # Gumbel-Softmax 采样（训练时使用，测试时用确定性采样）
        if self.training:
            logits = torch.log(P + 1e-20) - torch.log(1 - P + 1e-20)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            sample = torch.sigmoid((logits + gumbel_noise) / self.tau)
            G_hat = (sample > 0.5).float()
        else:
            G_hat = (P > 0.5).float()
        
        return P, G_hat
    
    def update_tau(self, epoch, max_epochs):
        """温度退火：从1.0线性降至0.1"""
        self.tau = max(0.1, 1.0 - 0.9 * epoch / max_epochs)

class EventPredictor(nn.Module):
    """因果感知的事件预测器"""
    def __init__(self, n_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 2)  # 二分类（>50K 或 <=50K）
        )
    
    def forward(self, X):
        return self.mlp(X)

# ==================== 3. 无环约束 (Acyclicity Constraint) ====================
def acyclicity_constraint(P):
    """
    Eq.(10): h(G) = tr((I + G/|X|)^|X|) - |X|
    确保因果图为 DAG（有向无环图）
    """
    n = P.size(0)
    I = torch.eye(n, device=P.device)
    # 使用 NOTEARS 的数值稳定版本: h(G) = tr(expm(G ◦ G)) - n
    M = torch.matrix_exp(P * P)  # expm(G ◦ G)
    h = torch.trace(M) - n
    return h**2  # Lc = h(G)^2

# ==================== 4. 反事实解释器（内生验证核心） ====================
class CounterfactualExplainer:
    """内生反事实解释器：用于评估因果图质量 + 检测过拟合"""
    def __init__(self, causal_graph, predictor, cat_mask, device='cpu'):
        self.G = causal_graph.cpu().numpy()  # [n, n] 二值矩阵
        self.predictor = predictor
        self.cat_mask = cat_mask
        self.device = device
    
    def heom_distance(self, x, y, feature_ranges):
        """异构欧氏重叠度量 (HEOM)"""
        dist = 0.0
        for i in range(len(x)):
            if self.cat_mask[i]:  # 类别特征
                dist += 0 if abs(x[i] - y[i]) < 1e-5 else 1
            else:  # 数值特征
                if feature_ranges[i] > 1e-5:
                    dist += abs(x[i] - y[i]) / feature_ranges[i]
                else:
                    dist += 0
        return dist
    
    def find_unlike_neighbor(self, instance, dataset_X, dataset_y, feature_ranges, current_pred=None):
        """找到预测结果不同的最近邻（异类最近邻)"""
        if current_pred is None:
            with torch.no_grad():
                pred = torch.argmax(self.predictor(
                    torch.FloatTensor(instance).unsqueeze(0).to(self.device)
                ), dim=1).item()
        else:
            pred = current_pred
        
        # 筛选异类样本
        unlike_mask = (dataset_y != pred)
        if not np.any(unlike_mask):
            return None
            
        candidates = dataset_X[unlike_mask]
        
        # 向量化 HEOM 计算
        # diffs: (n_candidates, n_features)
        diffs = np.abs(candidates - instance)
        
        # 1. Categorical: 1 if diff > epsilon
        # cat_mask broadcasting
        cat_dist = (diffs > 1e-5) * self.cat_mask
        
        # 2. Numerical: diff / range
        # feature_ranges broadcasting
        # Avoid div by zero
        safe_ranges = feature_ranges.copy()
        safe_ranges[safe_ranges < 1e-5] = 1.0
        num_dist = (diffs / safe_ranges) * (~self.cat_mask)
        
        # Sum
        dists = (cat_dist + num_dist).sum(axis=1)
        
        # Find min
        min_idx = np.argmin(dists)
        return candidates[min_idx]
    
    def generate(self, instance, dataset_X, dataset_y, feature_ranges, target_idx=None, current_pred=None):
        """
        生成反事实样本（遵循因果约束）：
        1. 仅修改与目标有直接因果边的特征 (G[:, target_idx] == 1)
        2. 若修改 xi，需同步调整其直接子节点 xj (因果传播)
        """
        # 1. 获取原始预测
        if current_pred is None:
            with torch.no_grad():
                orig_pred = torch.argmax(self.predictor(
                    torch.FloatTensor(instance).unsqueeze(0).to(self.device)
                ), dim=1).item()
        else:
            orig_pred = current_pred

        st = self.find_unlike_neighbor(instance, dataset_X, dataset_y, feature_ranges, current_pred=orig_pred)
        if st is None:
            return None
        
        # 筛选可修改特征
        if target_idx is not None:
            # 仅修改与目标有直接因果边的特征 (G[i, target_idx] == 1)
            modifiable_indices = [i for i in range(len(instance)) if self.G[i, target_idx] > 0.5]
        else:
            modifiable_indices = range(len(instance))
        
        # 批量生成候选样本
        candidates = []
        valid_indices = []
        
        for i in modifiable_indices:
            if np.abs(instance[i] - st[i]) < 1e-5:
                continue
            
            sc = instance.copy()
            sc[i] = st[i]
            candidates.append(sc)
            valid_indices.append(i)
        
        if not candidates:
            return None
            
        # 批量预测 (优化：一次性预测所有候选样本)
        candidates_tensor = torch.FloatTensor(np.array(candidates)).to(self.device)
        with torch.no_grad():
            batch_logits = self.predictor(candidates_tensor)
            batch_preds = torch.argmax(batch_logits, dim=1).cpu().numpy()
            
        # 找到第一个翻转的样本
        for idx, new_pred in enumerate(batch_preds):
            if new_pred != orig_pred:
                return candidates[idx]
        
        return None  # 未能生成有效反事实

def compute_discriminative_power(counterfactuals, test_X, train_X, train_y, predictor, device='cpu'):
    """
    计算 Discriminative Power (dispo):
    用 counterfactuals + test_X 训练 1NN，在 S_eq ∪ S_neq 上评估分类准确率
    """
    if len(counterfactuals) == 0:
        return 0.0
    
    # 构建训练集: test instances (label=1) + counterfactuals (label=0)
    train_set_X = np.vstack([test_X, counterfactuals])
    train_set_y = np.hstack([np.ones(len(test_X)), np.zeros(len(counterfactuals))])
    
    # 为每个测试样本构建 S_eq (同类最近邻) 和 S_neq (异类最近邻)
    test_eval_X = []
    test_eval_y = []
    
    with torch.no_grad():
        test_preds = torch.argmax(predictor(
            torch.FloatTensor(test_X).to(device)
        ), dim=1).cpu().numpy()
    
    for i, (x_test, pred) in enumerate(zip(test_X, test_preds)):
        # 同类样本
        same_idx = np.where(train_y == pred)[0]
        if len(same_idx) > 0:
            same_dists = cdist([x_test], train_X[same_idx], metric='euclidean')[0]
            s_eq = train_X[same_idx[np.argmin(same_dists)]]
            test_eval_X.append(s_eq)
            test_eval_y.append(1)
        
        # 异类样本
        diff_idx = np.where(train_y != pred)[0]
        if len(diff_idx) > 0:
            diff_dists = cdist([x_test], train_X[diff_idx], metric='euclidean')[0]
            s_neq = train_X[diff_idx[np.argmin(diff_dists)]]
            test_eval_X.append(s_neq)
            test_eval_y.append(0)
    
    if len(test_eval_X) == 0:
        return 0.0
    
    # 1NN 分类
    eval_X = np.array(test_eval_X)
    eval_y = np.array(test_eval_y)
    preds = []
    for x in eval_X:
        dists = cdist([x], train_set_X, metric='euclidean')[0]
        nn_idx = np.argmin(dists)
        preds.append(train_set_y[nn_idx])
    
    return accuracy_score(eval_y, preds)

# ==================== 5. 完整训练流程 ====================
def train_interpet(
    X_train, y_train, 
    X_val=None, y_val=None,  # 可选：用于早停监控（非必须，InterPet 使用内生验证）
    n_epochs=200,
    lr_causal=1e-3,
    lr_predictor=1e-3,
    lambda_c=0.1,   # 无环约束权重
    lambda_s=0.01,  # 稀疏性约束权重
    batch_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    
    # 初始化模型
    causal_module = CausalDiscovery(n_features).to(device)
    predictor = EventPredictor(n_features).to(device)
    
    # 优化器（交替优化）
    optimizer_causal = optim.Adam(causal_module.parameters(), lr=lr_causal)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=lr_predictor)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 特征范围（用于 HEOM）
    feature_ranges = X_train.max(axis=0) - X_train.min(axis=0) + 1e-5
    
    # 训练循环（EM 风格交替优化）
    best_dispo = 0.0
    patience = 50
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        causal_module.train()
        predictor.train()
        
        # ===== 阶段1: 固定因果图，优化预测器 =====
        epoch_loss_pred = 0.0
        for _ in range(5):  # 每轮优化预测器5次
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = torch.FloatTensor(X_train[idx]).to(device)
            y_batch = torch.LongTensor(y_train[idx]).to(device)
            
            with torch.no_grad():
                _, G_hat = causal_module(X_batch)
            
            # 特征过滤：仅保留与目标相关的特征（简化：使用全连接）
            # 实际中应使用 G_hat[:, target_idx] 作为掩码
            X_filtered = X_batch  # * G_hat[:, target_idx].unsqueeze(0)
            
            optimizer_predictor.zero_grad()
            logits = predictor(X_filtered)
            loss_pred = criterion(logits, y_batch)
            loss_pred.backward()
            optimizer_predictor.step()
            
            epoch_loss_pred += loss_pred.item()
        
        # ===== 阶段2: 固定预测器，优化因果图 =====
        epoch_loss_graph = 0.0
        for _ in range(1):  # 每轮优化因果图1次
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = torch.FloatTensor(X_train[idx]).to(device)
            y_batch = torch.LongTensor(y_train[idx]).to(device)
            
            optimizer_causal.zero_grad()
            
            # 前向传播
            P, G_hat = causal_module(X_batch)
            X_filtered = X_batch  # * G_hat[:, target_idx].unsqueeze(0)
            logits = predictor(X_filtered)
            
            # 损失计算
            loss_pred = criterion(logits, y_batch)
            loss_c = lambda_c * acyclicity_constraint(P)  # 无环约束
            loss_s = lambda_s * torch.norm(P, p=1)       # 稀疏性约束
            loss_graph = loss_pred + loss_c + loss_s
            
            loss_graph.backward()
            optimizer_causal.step()
            
            epoch_loss_graph += loss_graph.item()
        
        # 温度退火
        causal_module.update_tau(epoch, n_epochs)
        
        # ===== 阶段3: 反事实内生验证（每5轮） =====
        dispo = 0.0
        if epoch % 5 == 0:
            causal_module.eval()
            predictor.eval()
            
            with torch.no_grad():
                _, G_hat = causal_module(torch.FloatTensor(X_train).to(device))
            
            # 采样少量测试样本生成反事实
            sample_size = int(n_samples * 0.3)
            test_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_test_sample = X_train[test_indices]
            y_test_sample = y_train[test_indices]
            
            # 批量预测优化
            with torch.no_grad():
                sample_tensor = torch.FloatTensor(X_test_sample).to(device)
                sample_preds = torch.argmax(predictor(sample_tensor), dim=1).cpu().numpy()
            
            explainer = CounterfactualExplainer(
                causal_graph=G_hat,
                predictor=predictor,
                cat_mask=np.array([False]*5 + [True]*(n_features-5)),
                device=device
            )
            
            counterfactuals = []
            for idx, i in enumerate(test_indices):
                cf = explainer.generate(
                    X_train[i], X_train, y_train, feature_ranges, current_pred=sample_preds[idx]
                )
                if cf is not None:
                    counterfactuals.append(cf)
            
            if len(counterfactuals) > 0:
                dispo = compute_discriminative_power(
                    np.array(counterfactuals),
                    X_test_sample,
                    X_train,
                    y_train,
                    predictor,
                    device
                )
            
            causal_module.train()
            predictor.train()
        
        # ===== 早停判断（基于 dispo）=====
        if dispo > best_dispo:
            best_dispo = dispo
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'causal_state_dict': causal_module.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'dispo': dispo,
                'lambda_c': lambda_c,
                'lambda_s': lambda_s
            }, f'{checkpoint_dir}/interpet_syn_dataset_best.pth')
        else:
            patience_counter += 1
        
        # 打印训练日志
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss_pred: {epoch_loss_pred/5:.4f} | "
                  f"Loss_graph: {epoch_loss_graph:.4f} | Dispo: {dispo:.4f} | "
                  f"Tau: {causal_module.tau:.2f}")
        
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in dispo for {patience} rounds)")
            break
    
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    print(f"\nTraining completed. Best dispo: {best_dispo:.4f}")
    return causal_module, predictor, best_dispo

# ==================== 6. 测试函数 ====================
def test_interpet(model_path, X_test, y_test, feature_names, cat_mask, device='cpu'):
    n_features = X_test.shape[1]
    
    # 加载模型
    causal_module = CausalDiscovery(n_features).to(device)
    predictor = EventPredictor(n_features).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    causal_module.load_state_dict(checkpoint['causal_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    
    causal_module.eval()
    predictor.eval()
    
    # 生成因果图
    with torch.no_grad():
        _, causal_graph = causal_module(torch.FloatTensor(X_test).to(device))
        logits = predictor(torch.FloatTensor(X_test).to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    # 评估指标
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # 可视化因果图（简化：打印前10个强因果边）
    print("\nTop 10 causal edges (feature_i -> feature_j):")
    edges = []
    for i in range(n_features):
        for j in range(n_features):
            if causal_graph[i, j] > 0.5:
                edges.append((i, j, causal_graph[i, j].item()))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:10]
    for i, j, weight in edges:
        print(f"  {feature_names[i]:20s} -> {feature_names[j]:20s} (weight={weight:.2f})")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'dispo': checkpoint.get('dispo', 0.0),
        'causal_graph': causal_graph.cpu().numpy().tolist()
    }

# ==================== 7. 主程序 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 配置路径
    n_samples = 10000
    n_features = 300
    TRAIN_CSV = f"/workspace/causal_discovery/dataset/syn_dataset/{n_samples}samples_{n_features}features/scm_train.csv"
    TEST_CSV = f"/workspace/causal_discovery/dataset/syn_dataset/{n_samples}samples_{n_features}features/scm_test.csv"
    
    # 1. 加载数据
    print("Loading dataset...")
    (X_train, y_train), (X_test, y_test), feature_names, cat_mask, scaler = load_data(
        TRAIN_CSV, TEST_CSV
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. 训练模型
    print("\n" + "="*60)
    print("Training InterPet...")
    print("="*60)
    causal_module, predictor, best_dispo = train_interpet(
        X_train, y_train,
        n_epochs=1000,
        lambda_c=0.1,
        lambda_s=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=f'/workspace/causal_discovery/result/syn_dataset/{n_samples}samples_{n_features}features/checkpoints'
    )
    
    # 3. 测试模型
    print("\n" + "="*60)
    print("Testing InterPet...")
    print("="*60)
    results = test_interpet(
        f'/workspace/causal_discovery/result/syn_dataset/{n_samples}samples_{n_features}features/checkpoints/interpet_syn_dataset_best.pth',
        X_test, y_test,
        feature_names, cat_mask,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 4. 保存结果
    with open(f'/workspace/causal_discovery/result/syn_dataset/{n_samples}samples_{n_features}features/interpet_syn_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to interpet_syn_dataset_results.json")