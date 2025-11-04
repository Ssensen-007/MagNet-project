import torch
import torch.nn as nn
from GCNEncoder import GCNEncoder
from mctm_modules_graph import MoEFeatureFusion


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = GCNEncoder(
            in_dims=[params.word_dim, params.pos_dim, params.entity_dim], 
            hidden_dim=params.hidden_dim,
            out_dim=params.out_dim,
            num_types=3,
            num_layers=2,
        )
        self.moe_fusion = MoEFeatureFusion(params.out_dim)
        self.x_layernorm = nn.LayerNorm(params.out_dim)
        self.cls_layer = nn.Sequential(
            nn.Linear(params.out_dim, params.out_dim),
            nn.ReLU(),
            nn.LayerNorm(params.out_dim),
            nn.Dropout(params.dropout),
            nn.Linear(params.out_dim, params.num_classes)
        )
        self.proj_head = ProjectionHead(params.out_dim, params.out_dim)
    
    def forward(self, node_feats, adj_dict, tfidf_word, tfidf_pos, tfidf_entity, 
                return_feats=False, return_proj=False):
        device = next(self.parameters()).device
        tfidf_word = tfidf_word.to(device)
        tfidf_pos = tfidf_pos.to(device)
        tfidf_entity = tfidf_entity.to(device)
        for k in node_feats:
            node_feats[k] = node_feats[k].to(device)
        for k in adj_dict:
            adj_dict[k] = adj_dict[k].to(device)
        all_emb = self.encoder(node_feats, adj_dict)
        N_word, N_pos, N_entity = node_feats[0].shape[0], node_feats[1].shape[0], node_feats[2].shape[0]
        word_emb, pos_emb, entity_emb = all_emb[:N_word], all_emb[N_word:N_word+N_pos], all_emb[N_word+N_pos:]
        doc_word   = torch.sparse.mm(tfidf_word, word_emb)
        doc_pos    = torch.matmul(tfidf_pos, pos_emb)
        doc_entity = torch.matmul(tfidf_entity, entity_emb)
        doc_fea, gate_weights = self.moe_fusion(doc_word, doc_pos, doc_entity)
        doc_fea = self.x_layernorm(doc_fea)
        logits = self.cls_layer(doc_fea)
        if return_proj:
            proj_fea = self.proj_head(doc_fea)
            return logits, doc_fea, proj_fea, gate_weights
        elif return_feats:
            return logits, doc_fea
        else:
            return logits
