import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dims, out_dim):
        """
        in_dims: list[int], 每种类型节点的输入维度
        out_dim: int, 输出维度
        """
        super().__init__()
        self.num_types = len(in_dims)
        self.out_dim = out_dim

        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for in_dim in in_dims
        ])
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(out_dim)
            for _ in range(self.num_types)
        ])

        # 仅当 in_dim == out_dim 时可直接残差相加
        self.residuals = nn.ModuleList([
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
            for in_dim in in_dims
        ])

    def forward(self, node_feats, adj_dict):
        out_list = []
        for t in range(self.num_types):
            x = node_feats[t]                  # [N_t, in_dim]
            adj = adj_dict[t]                  # [N_t, N_t] 稀疏邻接矩阵

            h_proj = self.linears[t](x)        # [N_t, out_dim]
            h_agg = torch.spmm(adj, h_proj) if adj.is_sparse else adj @ h_proj

            h_res = self.residuals[t](x)       # residual path
            h_out = h_agg + h_res              # residual connection
            h_out = self.layernorms[t](h_out)  # layer norm

            out_list.append(h_out)

        return torch.cat(out_list, dim=0)      # [N_total, out_dim]

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GCNLayer(nn.Module):
#     def __init__(self, in_dims, out_dim, dropout=0.1):
#         super().__init__()
#         self.num_types = len(in_dims)
#         self.out_dim = out_dim
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.ReLU()

#         self.linears = nn.ModuleList([
#             nn.Linear(in_dim, out_dim, bias=True)
#             for in_dim in in_dims
#         ])

#         self.residuals = nn.ModuleList([
#             nn.Linear(in_dim, out_dim, bias=True) if in_dim != out_dim else nn.Identity()
#             for in_dim in in_dims
#         ])

#         self.layernorms = nn.ModuleList([
#             nn.LayerNorm(out_dim)
#             for _ in range(self.num_types)
#         ])

#         self.type_weight = nn.Parameter(torch.ones(self.num_types))

#     def forward(self, node_feats, adj_dict):
#         out_list = []
#         for t in range(self.num_types):
#             x = node_feats[t]
#             adj = adj_dict[t]

#             h_proj = self.linears[t](x)
#             h_agg = torch.spmm(adj, h_proj) if adj.is_sparse else adj @ h_proj

#             res = self.residuals[t](x)

#             h = h_agg + res
#             h = self.layernorms[t](h)
#             h = self.activation(h)
#             h = self.dropout(h)
#             out_list.append(h)

#         weights = F.softmax(self.type_weight, dim=0)
#         h_final = torch.cat([w * h for w, h in zip(weights, out_list)], dim=0)
#         return h_final
