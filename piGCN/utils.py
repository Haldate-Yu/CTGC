import os
import random
import math
import numpy as np
import scipy.sparse as sp
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_sparse import SparseTensor


def label_propagation(data, k=3, alpha=0.2):
    adj = to_dense_adj(data.edge_index)
    # train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    y0 = torch.zeros(data.y.shape[0], data.y.max().item() + 1)
    y = F.one_hot(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    y0[data.train_mask] = y[data.train_mask]
    y = y0

    for _ in range(k):
        y = torch.matmul(adj, y)
        # y[data.train_mask] = y0[data.train_mask]
        y = (1 - alpha) * y + alpha * y0
        y.clamp_(0., 1.)

    return y


def adj_pinv(dataname, data, topk_nodes=100):
    os.makedirs("./pinv-dataset", exist_ok=True)
    path = './pinv-dataset/{}.npz'.format(dataname)
    nnode = data.x.size(0)

    if os.path.exists(path):
        pinv = sp.load_npz(path)
    else:
        # adj = to_dense_adj(data.edge_index).numpy()
        nnode = data.x.size(0)
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                           sparse_sizes=(nnode, nnode)).to_dense().numpy()
        # print(to_scipy_sparse_matrix(data.edge_index), type(to_scipy_sparse_matrix(data.edge_index)))
        adj = sp.csr_matrix(adj)
        pinv = cal_pinv(adj)
        # print(pinv)

        print(pinv, type(pinv))
        # print(from_scipy_sparse_matrix(pinv))
        sp.save_npz(path, pinv)

    # sparse_pinv = torch.from_numpy(pinv).to_sparse()
    print(from_scipy_sparse_matrix(pinv))
    edge_index, edge_attr = from_scipy_sparse_matrix(pinv)  # np.vstack((pinv.row, pinv.col)), pinv.data

    # edge_index, edge_attr = sparse_pinv.indices(), sparse_pinv.values().to(torch.float32)

    return edge_index, edge_attr.float()


def topk(array, topk=80):
    index = array.argsort()[-topk:][::-1]
    a = set(index)
    b = set(list(range(array.shape[0])))
    array[list(b.difference(a))] = 0

    return array


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def cal_pinv(adj):
    # record values
    adj = (adj + sp.eye(adj.shape[0])).toarray()
    i, j = np.nonzero(adj)
    # print(i, j)
    values = zip(i, j)

    # print(len(list(values)))
    # calculate norm lap
    degree = np.diag(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    pinv = scipy.linalg.pinvh(lap)

    # print('laplacian: {}'.format(lap))
    # print('pinv: {}'.format(pinv))
    # calculate ectd

    ectd = calculate_ectd(nnodes=adj.shape[0], values=values, pinv=pinv)
    ectd = np.around(ectd, 3)
    # print('ectd: {} type: {}'.format(ectd, type(ectd)))
    adj = sp.coo_matrix(ectd)
    # print('adj: {}'.format(adj))
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return adj.tocoo()


def calculate_ectd(nnodes, values, pinv):
    ectd = np.zeros((nnodes, nnodes), )

    for i, j in values:
        # print('i, j: {} {}'.format(i, j))
        eij = np.zeros((nnodes, 1), )
        eij[i, 0] = 1.
        eij[j, 0] = -1. if i != j else 0.

        ectd[i, j] = eij.T @ pinv @ eij
    print(np.nonzero(ectd))
    ectd_norm = np.power(ectd, -1.)
    ectd_norm[np.isinf(ectd_norm)] = 0.
    ectd_norm[ectd_norm < 0] = 0.
    np.fill_diagonal(ectd_norm, 1.)
    return ectd_norm


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
