import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import random_split
import copy

import torch_geometric.transforms as T
from torch_geometric.datasets import PPI, Planetoid, Coauthor, Amazon, Reddit
from torch_geometric.utils import remove_isolated_nodes
from utils import *

from torch.utils.data import random_split
import torch_geometric.transforms as T


def load_citation(dataname, path, opt='none'):
    if opt == 'gcn':
        dataset = Planetoid(path, dataname, transform=T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, dataname)
    
    data = dataset[0]
   
    if dataname == 'citeseer':
        data.edge_index, data.edge_attr, node_mask = remove_isolated_nodes(data.edge_index, data.edge_attr)
        data.x = data.x[node_mask]
        data.y = data.y[node_mask]
        data.train_mask = data.train_mask[node_mask]
        data.val_mask = data.val_mask[node_mask]
        data.test_mask = data.test_mask[node_mask]
        
    soft_y = label_propagation(data)[0]
    # print(soft_y)
    if opt == 'add-label':
        # num_node_features = dataset.num_node_features + dataset.num_classes
        # dataset.num_node_features = num_node_features

        temp_y = torch.zeros(data.num_nodes, dataset.num_classes)
        temp_y[data.train_mask] = soft_y[data.train_mask]
        data.x = torch.cat((data.x, temp_y), dim=1)
    elif opt == 'label-only':
        # dataset.num_node_features = dataset.num_classes
        temp_y = torch.zeros(data.num_nodes, dataset.num_classes)
        # temp_y[data.train_mask | data.val_mask] = soft_y[data.train_mask | data.val_mask]
        # data.x = temp_y
        data.x = soft_y

    return dataset, data

def load_others(dataname, path):
    if dataname in ['CS', 'Physics']:
        dataset = Coauthor(root='./data', name=dataname, transform=T.NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='./data', name=dataname, transform=T.NormalizeFeatures())
    
    data = dataset[0]

    # split train/test/val
    data.train_mask, data.val_mask, data.test_mask = eq_split(data.num_nodes, labels=data.y, nclass=dataset.num_classes)
    print(data)

    return dataset, data

def eq_split(num_node, train_num=20, valid_num=50, labels=None, nclass=0):
    all_idx = np.random.permutation(num_node).tolist()
    all_label = labels.tolist()

    train_list = [0 for _ in range(nclass)]
    train_idx = []

    # training set split
    for i in all_idx:
        iter_label = all_label[i]
        if train_list[iter_label] < train_num:
            train_list[iter_label] += 1
            train_idx.append(i)

        if sum(train_list) == train_num * nclass:
            break
    
    # print(sum(train_list))
    assert sum(train_list) == train_num * nclass
    
    # valid + test
    after_train_idx = list(set(all_idx) - set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for i in after_train_idx:
        iter_label = all_label[i]
        if valid_list[iter_label] < valid_num:
            valid_list[iter_label] += 1
            valid_idx.append(i)

        if sum(valid_list) == valid_num * nclass:
            break
    
    # print(sum(valid_list))
    assert sum(valid_list) == valid_num * nclass
    # test
    test_idx = list(set(after_train_idx) - set(valid_idx))
    
    train_idx = torch.from_numpy(np.sort(train_idx))
    valid_idx = torch.from_numpy(np.sort(valid_idx))
    test_idx = torch.from_numpy(np.sort(test_idx))

    train_mask = get_mask(train_idx, num_node)
    valid_mask = get_mask(valid_idx, num_node)
    test_mask = get_mask(test_idx, num_node)
    
    return train_mask, valid_mask, test_mask

def get_mask(idx, num_node):
    mask = torch.zeros((num_node, )).to(torch.bool)
    
    mask[idx] = True
    
    return mask


