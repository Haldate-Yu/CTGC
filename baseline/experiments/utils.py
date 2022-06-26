import os

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor


def adj_pinv(data, dataname=None, split=0, topk=10):
    os.makedirs('./pinv-dataset', exist_ok=True)
    # dir for ectd-adj filter
    # os.makedirs('./pinv-dataset/test-{}'.format(dataname), exist_ok=True)
    # file = './pinv-dataset/test-{}/{}-{}.npz'.format(dataname, dataname, split)

    # dir for ectd-topk/bottomk filter
    # os.makedirs('./pinv-dataset/{}-ectd-bottomk{}'.format(dataname, topk), exist_ok=True)
    # file = './pinv-dataset/{}-ectd-bottomk{}/{}-{}.npz'.format(dataname, topk, dataname, split)

    # dir for mfpt-topk/bottomk filter,
    # it records fot mfpt-adj filter while topk=666
    os.makedirs('./pinv-dataset/{}-topk{}'.format(dataname, topk), exist_ok=True)
    file = './pinv-dataset/{}-topk{}/{}-{}.npz'.format(dataname, topk, dataname, split)

    if os.path.exists(file):
        pinv = sp.load_npz(file)
    else:
        nnode = data.x.size(0)
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                           sparse_sizes=(nnode, nnode)).to_dense().numpy()
        adj = sp.csr_matrix(adj)
        # mfpt
        pinv = cal_mfpt(adj, topk)
        # ectd-adj
        # pinv = cal_pinv(adj)
        # ectd-topk/botttomk
        # pinv = cal_full_pinv(adj, topk)
        sp.save_npz(file, pinv)

    edge_index, edge_attr = from_scipy_sparse_matrix(pinv)

    return edge_index, edge_attr.float()


# ECTD ADJ Filter
def cal_pinv(adj):
    adj = (adj + sp.eye(adj.shape[0])).toarray()
    i, j = np.nonzero(adj)
    values = zip(i, j)

    deg = np.diag(adj.sum(1))
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    pinv = scipy.linalg.pinvh(lap)

    ectd = cal_ectd(nnode=adj.shape[0], values=values, pinv=pinv)
    ectd = np.around(ectd, 3)
    adj = sp.coo_matrix(ectd)

    return adj.tocoo()


# ECTD TopK/BottomK Filter
def cal_full_pinv(adj, topk=10):
    adj = (adj + sp.eye(adj.shape[0])).toarray()

    full = np.ones_like(adj)
    i, j = np.nonzero(full)
    values = zip(i, j)

    deg = np.diag(adj.sum(1))
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    pinv = scipy.linalg.pinvh(lap)

    ectd = cal_ectd(nnode=adj.shape[0], values=values, pinv=pinv, topk=topk)
    ectd = np.around(ectd, 3)
    # topk filter
    ectd = np.apply_along_axis(get_topk_matrix, 1, ectd, k=topk)
    adj = sp.coo_matrix(ectd)

    return adj.tocoo()


# ectd filter
def cal_ectd(nnode, values, pinv, topk=0):
    ectd = np.zeros((nnode, nnode), )

    for i, j in values:
        eij = np.zeros((nnode, 1), )
        eij[i, 0] = 1.
        eij[j, 0] = -1. if i != j else 0.
        ectd[i, j] = eij.T @ pinv @ eij

    # bottom k
    # ectd = np.apply_along_axis(get_topk_matrix, 1, ectd, k=topk)

    ectd_norm = np.power(ectd, -1)
    ectd_norm[np.isinf(ectd_norm)] = 0
    # for some reason, part of the distances are computed a negative number
    # so just keep the original edge weight
    ectd_norm[ectd_norm < 0.] = 1.

    return ectd_norm


# full matrix of mfpt
def cal_mfpt(adj, topk=20):
    adj = (adj + sp.eye(adj.shape[0])).toarray()

    nnodes = adj.shape[0]
    deg = np.diag(adj.sum(1))

    # Standard Form
    # lap = deg - adj
    # lap_pinv = np.linalg.pinv(lap)

    # Symmetric Form
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    lap = np.identity(nnodes) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    lap_pinv = np.linalg.pinv(lap)

    volG = deg.sum().sum()
    tmp = lap_pinv @ deg @ np.ones((nnodes, 1))
    AFT = -lap_pinv * volG + tmp - tmp.T + np.diag(lap_pinv) * volG
    AFT = AFT / volG

    mfpt = AFT + AFT.T

    # bottomk_filter
    # mfpt = np.apply_along_axis(get_topk_matrix, 1, mfpt, k=topk)

    # mfpt = np.power(mfpt, -1.)
    mfpt[np.isinf(mfpt)] = 0.
    # check if there exists negative values or
    # small values?
    # print(np.count_nonzero(mfpt < 0), mfpt)
    mfpt[mfpt < 0] = 0.

    # topk_filter
    mfpt = np.apply_along_axis(get_topk_matrix, 1, mfpt, k=topk)

    # adj_filter
    # mfpt = np.where(adj <= 0, 0., mfpt)

    mfpt = sp.coo_matrix(mfpt)

    return mfpt.tocoo()


def get_topk_matrix(adj, k=20):
    indexes = adj.argsort()[-k:][::-1]
    a = set(indexes)
    b = set(list(range(adj.shape[0])))
    adj[list(b.difference(a))] = 0

    return adj
