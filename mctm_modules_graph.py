import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoEFeatureFusion(nn.Module):
    def __init__(self, feat_dim, num_experts=3, hidden_dim=None, k=2, moe_temperature=1.0):
        """
        feat_dim: 每个输入通道的维度（默认3个通道）
        hidden_dim: expert中间层维度
        k: top-k 稀疏门控（每个样本最多使用k个专家）
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.moe_temperature = moe_temperature
        self.experts = nn.ModuleList([
            Expert(feat_dim, hidden_dim) for _ in range(self.num_experts)
        ])

        gate_input_dim = feat_dim * self.num_experts
       
        self.gate = nn.Linear(gate_input_dim, self.num_experts)

    def forward(self, doc_word, doc_pos, doc_entity):
        # === 每通道输入分别送入专家 ===
        expert_inputs = [doc_word, doc_pos, doc_entity]
        expert_outputs = [expert(x) for expert, x in zip(self.experts, expert_inputs)]  # list of [B, D]
        feats = torch.cat(expert_inputs, dim=1)  # [B, D*3]

        gate_logits = self.gate(feats)  # [B, 3]

        # === Top-k 稀疏门控 ===
        topk = min(self.k, self.num_experts)
        topk_vals, topk_idx = gate_logits.topk(topk, dim=1)  # [B, k]
        mask = torch.zeros_like(gate_logits).scatter(1, topk_idx, 1.0)
        gated_logits_masked = gate_logits * mask
        gate_weights = F.softmax(gated_logits_masked / self.moe_temperature, dim=1)  # 稀疏权重 [B, 3]

        # === 融合专家输出 ===
        fused = sum(gate_weights[:, i:i+1] * expert_outputs[i] for i in range(self.num_experts))  # [B, D]

        return fused, gate_weights

def gate_entropy_loss(gate_weights):
    # 熵越小越稀疏
    entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=1).mean()
    return entropy

def expert_balance_loss(gate_weights):
    """
    负载均衡损失：鼓励不同专家被均衡使用
    例如：若某专家被所有样本使用，loss会增大
    """
    avg = gate_weights.mean(dim=0)  # [3]
    loss = (avg * avg).sum()  # 越均衡越小
    return loss
