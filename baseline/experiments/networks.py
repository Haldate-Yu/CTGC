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
        self.bn1 = BatchNorm1d(self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.bn2 = BatchNorm1d(self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.bn3 = BatchNorm1d(self.nhid)

        self.bn_pool = BatchNorm1d(self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)
        x = self.bn_pool(x)
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
        self.bn1 = BatchNorm1d(self.nhid)

        # self.convs = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList()
        # for i in range(self.K):
        #     self.convs.append(SGConv(self.num_features, self.nhid, K=1, cached=False))
        #     self.bns.append(BatchNorm1d(self.num_features))

        self.bn_pool = BatchNorm1d(self.nhid)

        # self.lin = torch.nn.Linear(self.num_features, self.nhid)
        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.bn1(self.conv1(x, edge_index))
        # for i in range(self.K):
        #     x = self.bns[i](self.convs[i](x, edge_index))
        # x = self.lin(x)

        x = global_mean_pool(x, batch)
        # x = self.bn_pool(x)
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
        self.bn1 = BatchNorm1d(self.nhid)
        self.bn_pool = BatchNorm1d(self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.bn1(self.conv1(x, edge_index))

        x = global_mean_pool(x, batch)
        # x = self.bn_pool(x)
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
        self.bn1 = BatchNorm1d(self.nhid)
        self.bn_pool = BatchNorm1d(self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

        # self.edge_emb = torch.nn.Linear(4, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.bn1(self.conv1(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        # x = self.bn_pool(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x
