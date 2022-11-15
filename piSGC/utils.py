import numpy as np
import scipy.sparse as sp
import torch
import sys, os, re
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from sklearn.model_selection import ShuffleSplit
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_sparse import SparseTensor
from torch_geometric.datasets import Coauthor, WebKB, Amazon
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T


# import matplotlib.pyplot as plt


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess(adj):
    adj_normalizer = fetch_normalization('OriNormAdj')
    adj = adj_normalizer(adj)
    return adj


def preprocess_citation(adj, features, args, normalization="FirstOrderGCN", ectd_data=''):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, ectd_data, args)
    features = row_normalize(features)
    return adj, features


def preprocess_test(adj, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", args=None, normalization="AugNormAdj", cuda=True, ectd_data=''):
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

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, args, normalization, ectd_data)

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
    transform = T.RandomNodeSplit(split='test_rest')

    if dataname in ['CS', 'Physics']:
        dataset = Coauthor(root='./data', name=dataname, transform=T.NormalizeFeatures())
    elif dataname in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data', name=dataname)
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='./data', name=dataname, transform=T.NormalizeFeatures())

    data = dataset[0]
    data = transform(data)
    nnode = data.x.size(0)
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(nnode, nnode)).to_dense().numpy()
    adj = sp.csr_matrix(adj)
    # load features, labels
    features = data.x
    labels = data.y

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, ectd_data, args)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if cuda:
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        test_idx = test_idx.cuda()

    return adj, features, labels, train_idx, val_idx, test_idx


def sgc_precompute(features, adj, degree, args=None):
    print("pre-compute adj:\n {}".format(adj))
    t = perf_counter()
    # ssgc like - init residual
    alpha = args.alpha

    ori = alpha * features
    emb = alpha * features
    # standard sgc
    for i in range(degree):
        features = torch.spmm(adj, features)
        # ssgc w/o degree
        if args.aggr_type == 'ssgc_no_avg':
            emb = emb + (1. - alpha) * features
        # ssgc
        if args.aggr_type == 'ssgc':
            emb = emb + (1 - alpha) * features / degree
        # sgc
        elif args.aggr_type == 'sgc':
            emb = alpha * ori + (1 - alpha) * features
    precompute_time = perf_counter() - t

    # Normalization - arctan/f2/tanh
    if args.norm_type == 'Arctan':
        emb = 2 * torch.arctan(emb) / torch.pi
    elif args.norm_type == 'F2':
        emb = F.normalize(emb, p=2, dim=1)
    elif args.norm_type == 'Tanh':
        emb = torch.tanh(emb)

    return emb, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def logger(args, acc_val, acc_test, precompute_time, train_time):
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/{}'.format(args.dataset), exist_ok=True)

    headline = '\n==========\n'
    parameters = 'Seed={}, Learning Rate={}, Weight_decay={}, Epochs={}'.format(args.seed, args.lr, args.weight_decay,
                                                                                args.epochs)
    summary1 = 'Model: {}, Degree: {}, Alpha: {}, Aggr_type: {}, Norm_type: {}'.format(args.model, args.degree,
                                                                                       args.alpha, args.aggr_type,
                                                                                       args.norm_type)
    summary2 = 'Validation Accuracy: {:.4f}, Test Accuracy: {:.4f}\nPre-compute Time: {:.4f}s, Train Time: {:.4f}s'.format(
        acc_val, acc_test, precompute_time, precompute_time + train_time)

    results_file = './results/{}/{}.txt'.format(args.dataset, args.model)
    with open(results_file, 'a+') as f:
        f.write('{}Parameters: {}\nModel Details: {}\nResult: {}\n'.format(headline, parameters, summary1, summary2))


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']
