import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import torch.nn.functional as F

from layers import *


class Net_GCN(torch.nn.Module):
    def __init__(self, args):
        super(Net_GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu((self.conv1(x, edge_index)))
        x = F.relu((self.conv2(x, edge_index)))
        x = F.relu((self.conv3(x, edge_index)))

        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GIN(torch.nn.Module):
    def __init__(self, args):
        super(Net_GIN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GINConv(
            Sequential(Linear(self.num_features, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GIN_W(torch.nn.Module):
    def __init__(self, args):
        super(Net_GIN_W, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GINConvWeight(
            Sequential(Linear(self.num_features, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv2 = GINConvWeight(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv3 = GINConvWeight(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.conv3(x, edge_index, edge_weight)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_SGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_SGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.conv1 = SGConv(self.num_features, self.nhid, K=self.K, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = (self.conv1(x, edge_index))

        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_SSGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_SSGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.conv1 = SSGConv(self.num_features, self.nhid, K=self.K, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = (self.conv1(x, edge_index))

        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_CTGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_CTGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.alpha = args.alpha
        self.aggr_type = args.aggr_type
        self.norm_type = args.norm_type

        self.conv1 = CTGConv(self.num_features, self.nhid, K=self.K, alpha=self.alpha, aggr_type=args.aggr_type,
                             norm_type=args.norm_type, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = (self.conv1(x, edge_index, edge_weight))

        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x
