import torch
import torch.nn as nn
from GCNLayer import GCNLayer

class GCNEncoder(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, num_types, num_layers=2):
        super().__init__()
        self.num_types = num_types
        self.out_dim = out_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dims = in_dims if i == 0 else [hidden_dim] * num_types
            layer_out_dim = hidden_dim if i < num_layers - 1 else out_dim
            self.layers.append(GCNLayer(layer_in_dims, layer_out_dim))  

    def forward(self, node_feats, adj_dict):
        h = node_feats
        for layer in self.layers:
            h_out = layer(h, adj_dict)
            split_sizes = [h[t].shape[0] for t in range(self.num_types)]
            h_split = torch.split(h_out, split_sizes, dim=0)
            h = {t: h_split[t] for t in range(self.num_types)}
        return h_out  # shape: [N_total, out_dim]

# import torch
# import torch.nn as nn
# from GCNLayer import GCNLayer

# class GCNEncoder(nn.Module):
#     def __init__(self, in_dims, hidden_dim, out_dim, num_types, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.num_types = num_types
#         self.out_dim = out_dim
#         self.layers = nn.ModuleList()

#         for i in range(num_layers):
#             layer_in_dims = in_dims if i == 0 else [hidden_dim] * num_types
#             layer_out_dim = hidden_dim if i < num_layers - 1 else out_dim
#             self.layers.append(GCNLayer(layer_in_dims, layer_out_dim, dropout=dropout))

#     def forward(self, node_feats, adj_dict):
#         h = node_feats
#         for layer in self.layers:
#             h_out = layer(h, adj_dict)
#             split_sizes = [h[i].shape[0] for i in range(self.num_types)]
#             h = {i: h_out[start:start + size]
#                  for i, (start, size) in enumerate(zip(torch.cumsum(torch.tensor([0] + split_sizes[:-1]), dim=0), split_sizes))}
#         return h_out
