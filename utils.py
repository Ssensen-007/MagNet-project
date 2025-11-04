import torch
import numpy as np
import random
import os
import csv
from sklearn.metrics import f1_score

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def step(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"⏸️ 早停触发计数：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0

def evaluate(logits, labels, index):
    preds = torch.argmax(logits[index], dim=1)
    correct = (preds == labels[index]).sum().item()
    return correct / len(index)

def evaluate_f1(logits, labels, index):
    preds = torch.argmax(logits[index], dim=1).cpu().numpy()
    true = labels[index].cpu().numpy()
    macro_f1 = f1_score(true, preds, average='macro')
    micro_f1 = f1_score(true, preds, average='micro')
    return macro_f1, micro_f1

def scipy_coo_to_torch_sparse(coo):
    if hasattr(coo, "tocoo"):  # 是scipy稀疏
        coo = coo.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        indices = torch.stack([row, col], dim=0)
        values = torch.tensor(coo.data, dtype=torch.float)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    elif hasattr(coo, "coalesce") and hasattr(coo, "indices"):
        return coo
    else:
        raise TypeError(f"未知类型的稀疏矩阵: {type(coo)}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_results(epoch, loss, train_acc, val_acc, train_f1, val_f1, file_path):
    header = ['epoch', 'loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1']
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            epoch,
            f"{loss:.4f}",
            f"{train_acc:.4f}",
            f"{val_acc:.4f}",
            f"{train_f1:.4f}",
            f"{val_f1:.4f}"
        ])

# === 加载全部embedding、邻接矩阵，并转换为数字key ===
def load_all_data(params):
    import pickle, json
    data_dir = params.data_dir

    # === 读取节点向量 ===
    with open(os.path.join(data_dir, 'word_emb.pkl'), 'rb') as f:
        word_emb = pickle.load(f)
    with open(os.path.join(data_dir, 'pos_emb.pkl'), 'rb') as f:
        pos_emb = pickle.load(f)
    with open(os.path.join(data_dir, 'entity_emb.pkl'), 'rb') as f:
        entity_emb = pickle.load(f)

    # === 读取邻接矩阵 ===
    with open(os.path.join(data_dir, 'adj_word2word.pkl'), 'rb') as f:
        adj_ww = pickle.load(f)
    with open(os.path.join(data_dir, 'adj_pos2pos.pkl'), 'rb') as f:
        adj_pp = pickle.load(f)
    with open(os.path.join(data_dir, 'adj_entity2entity.pkl'), 'rb') as f:
        adj_ee = pickle.load(f)

    # === 读取标签和索引 ===
    with open(os.path.join(data_dir, 'labels.json'), 'r') as f:
        label_dict = json.load(f)
        labels = torch.tensor([label_dict[str(i)] for i in range(len(label_dict))], device=params.device)
    with open(os.path.join(data_dir, 'train_idx.json'), 'r') as f:
        train_idx = torch.tensor(json.load(f), dtype=torch.long, device=params.device)
    with open(os.path.join(data_dir, 'val_idx.json'), 'r') as f:
        val_idx = torch.tensor(json.load(f), dtype=torch.long, device=params.device)
    with open(os.path.join(data_dir, 'test_idx.json'), 'r') as f:
        test_idx = torch.tensor(json.load(f), dtype=torch.long, device=params.device)
    with open(os.path.join(data_dir, 'augmented_flags.json'), "r") as f:
        augmented = torch.tensor(json.load(f), dtype=torch.long, device=params.device)
    # === 转为数字key的输入字典 ===
    h_dict = {
        0: torch.tensor(np.array(word_emb), dtype=torch.float, device=params.device),
        1: torch.tensor(np.array(pos_emb), dtype=torch.float, device=params.device),
        2: torch.tensor(np.array(entity_emb), dtype=torch.float, device=params.device),
    }
    adj_dicts = {
        0: scipy_coo_to_torch_sparse(adj_ww),
        1: scipy_coo_to_torch_sparse(adj_pp),
        2: scipy_coo_to_torch_sparse(adj_ee),
    }

    return h_dict, adj_dicts, labels, train_idx, val_idx, test_idx, augmented

# === 加载TF-IDF池化稀疏矩阵，并做shape检查 ===
def load_tfidf_matrices(data_dir, print_check=True):
    import pickle
    with open(os.path.join(data_dir, "tfidf_word.pkl"), "rb") as f:
        tfidf_word = pickle.load(f)
    with open(os.path.join(data_dir, "tfidf_pos.pkl"), "rb") as f:
        tfidf_pos = pickle.load(f)
    with open(os.path.join(data_dir, "tfidf_entity.pkl"), "rb") as f:
        tfidf_entity = pickle.load(f)
    # === 转为torch稀疏张量 ===
    tfidf_word = scipy_coo_to_torch_sparse(tfidf_word)
    tfidf_pos = scipy_coo_to_torch_sparse(tfidf_pos)
    tfidf_entity = scipy_coo_to_torch_sparse(tfidf_entity)

    # === 自动shape检查（务必保证列数与embedding数量一致） ===
    if print_check:
        print(f"[SHAPE CHECK] tfidf_word.shape = {tfidf_word.shape}")
        print(f"[SHAPE CHECK] tfidf_pos.shape  = {tfidf_pos.shape}")
        print(f"[SHAPE CHECK] tfidf_entity.shape = {tfidf_entity.shape}")

    return tfidf_word, tfidf_pos, tfidf_entity
