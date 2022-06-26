import os.path

import torch
# from torch_geometric.datasets import TUDataset
from tu_dataset import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from utils import adj_pinv


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False, normalize=False, pinv=False, topk=10):
    dataset = TUDataset(os.path.join('../data', name), name, use_node_attr=True, cleaned=cleaned)

    # set edge attr
    edge_weight = []
    edge_index = []

    for i, data in enumerate(dataset):
        if pinv is True:
            edge_index_i, edge_weight_i = adj_pinv(data, name, i, topk=topk)
            edge_weight.append(edge_weight_i)
            edge_index.append(edge_index_i)
        else:
            edge_weight.append(torch.ones((data.edge_index.size(1),), dtype=torch.float32,
                                          device=data.edge_index.device))
            edge_index.append(data.edge_index)

    dataset = dataset.set_edge_weight(edge_weight)
    dataset = dataset.set_edge_index(edge_index)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    elif normalize:

        dataset.data.x -= torch.mean(dataset.data.x, axis=0)
        dataset.data.x /= torch.std(dataset.data.x, axis=0)

    if not sparse:
        max_num_nodes = 0
        for data in dataset:
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        if dataset.transform is None:
            dataset.transform = T.ToDense(max_num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(max_num_nodes)])

    return dataset


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
