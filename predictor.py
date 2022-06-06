import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class GraphPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class SimpleMLPGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super(SimpleMLPGraphPredictor, self).__init__(dim_node_features, dim_edge_features, dim_target, config)

        hidden_units = config['hidden_units']

        self.fc_global = nn.Linear(dim_node_features, hidden_units)
        self.out = nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x.float(), batch)
        out = self.out(F.relu(self.fc_global(x)))
        return out, x
