import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class InstanceContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        mask = (~torch.eye(2 * batch_size, device=z.device).bool()).float()
        sim_matrix = sim_matrix * mask
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        denom = sim_matrix.sum(dim=1)
        loss = -torch.log(pos_sim / (denom + 1e-8)).mean()
        return loss

def generate_pseudo_labels(features, k=5):
    features = F.normalize(features, dim=1).detach().cpu()
    sim = torch.mm(features, features.T)
    sim.fill_diagonal_(-2)
    b = sim.size(0)
    x = torch.arange(b).unsqueeze(1).repeat(1, 1).flatten()
    y = torch.topk(sim, k, dim=1, largest=True)[1].flatten()
    rx = torch.cat([x, y]).numpy()
    ry = torch.cat([y, x]).numpy()
    v = np.ones(rx.shape[0])
    graph = csr_matrix((v, (rx, ry)), shape=(b, b))
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return torch.tensor(labels, dtype=torch.long)

class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, cluster_labels):
        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.t()) / self.temperature
        labels = cluster_labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = sim
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        logits = logits * logits_mask
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
