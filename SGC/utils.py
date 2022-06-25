import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter

from torch.utils.data import random_split
from torch_sparse import SparseTensor
from torch_geometric.datasets import Coauthor, WebKB, Amazon
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True, pinv=False):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if pinv:
        adj = pesudoinverse(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    # print(features)
    # print(adj, type(adj))
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_pyg_dataset(dataname, args, normalization="AugNormAdj", split=0, cuda=True, ectd_data=''):
    if dataname in ['CS', 'Physics']:
        dataset = Coauthor(root='./data', name=dataname)
    elif dataname in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data', name=dataname)
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='./data', name=dataname)

    data = dataset[0]
    transform = T.RandomNodeSplit(split='test_rest')
    data = transform(data)
    print(data)
    # process adj
    # adj = to_dense_adj(data.edge_index).numpy()
    nnode = data.x.size(0)
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(nnode, nnode)).to_dense().numpy()
    adj = sp.csr_matrix(adj)
    # load features, labels
    features = data.x
    labels = data.y

    train_idx, val_idx, test_idx = data.train_mask, data.val_mask, data.test_mask
    # preprocess adj & features
    # adj, features = preprocess_citation(adj, features, args, normalization, ectd_data)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    # load idx
    # if dataname in ['CS', 'Physics', 'Computers', 'Photo']:
    #     train_mask, val_mask, test_mask = eq_split(data.num_nodes, labels=data.y, nclass=dataset.num_classes)
    #     train_idx = train_mask
    #     val_idx = val_mask
    #     test_idx = test_mask
    # elif dataname in ['Cornell', 'Texas', 'Wisconsin']:
    #     # print(data.train_mask.shape)
    #     train_idx = data.train_mask[:, split]
    #     val_idx = data.val_mask[:, split]
    #     test_idx = data.test_mask[:, split]

    if cuda:
        # print(adj)
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        test_idx = test_idx.cuda()

    return adj, features, labels, train_idx, val_idx, test_idx


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
    mask = torch.zeros((num_node,)).to(torch.bool)

    mask[idx] = True

    return mask


def split_gen(num_nodes, train_ratio=0.1, val_ratio=0.1):
    train_len = int(num_nodes * train_ratio)
    val_len = int(num_nodes * val_ratio)
    test_len = num_nodes - train_len - val_len

    train_set, test_set, val_set = random_split(dataset=torch.arange(0, num_nodes),
                                                lengths=(train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_nodes,)).to(torch.bool)
    test_mask = torch.zeros((num_nodes,)).to(torch.bool)
    val_mask = torch.zeros((num_nodes,)).to(torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return train_mask, test_mask, val_mask


def pesudoinverse(adj):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_mx = sp.coo_matrix(adj).toarray()
    degree = np.diag(np.array(adj_mx.sum(0)))
    # print(degree)
    # print(adj_mx)
    laplacian = degree - adj_mx
    # print(laplacian)
    # n = adj.shape[0]
    # print(n)
    # I = np.eye(adj.shape[0])
    # J = (1 / n) * np.ones((n, n))
    # L_i = np.linalg.solve(laplacian + J, I) - J
    L_i = np.linalg.pinv(laplacian)
    # print(L_i[L_i<1 && L_i>0])
    adj_new = degree - L_i
    adj_new[adj_new < 0] = 0
    # topk, _ = L_i.topk(k=20, dim=1)
    # topk_min = torch.min(topk, dim=-1).values
    # topk_min = topk_min.unsqueeze(-1).repeat(1, n)
    # mask1 = torch.ge(L_i, topk_min)
    # mask2 = torch.zeros_like(L_i)

    # L_i = torch.where(mask1, L_i, mask2)
    # cora - 84(k=2)
    # citeseer - 175(k=2)
    # pubmed - 125(k=2)
    L_i = np.apply_along_axis(topk_values, 1, adj_new, topk=40)
    print(L_i)

    # return sigmoid(L_i)
    return L_i


def topk_values(array, topk=84):
    indexes = array.argsort()[-topk:][::-1]
    a = set(indexes)
    b = set(list(range(array.shape[0])))
    array[list(b.difference(a))] = 0

    return array


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter() - t
    return features, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T

    # adj = pesudoinverse(adj)

    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
