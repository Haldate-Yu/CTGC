import os

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor


def adj_pinv(data, dataname=None, split=0):
    os.makedirs('./pinv-dataset', exist_ok=True)
    os.makedirs('./pinv-dataset/{}'.format(dataname), exist_ok=True)
    file = './pinv-dataset/{}/{}-{}.npz'.format(dataname, dataname, split)

    if os.path.exists(file):
        pinv = sp.load_npz(file)
    else:
        nnode = data.x.size(0)
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                           sparse_sizes=(nnode, nnode)).to_dense().numpy()
        adj = sp.csr_matrix(adj)

        pinv = cal_pinv(adj)
        sp.save_npz(file, pinv)

    edge_index, edge_attr = from_scipy_sparse_matrix(pinv)

    return edge_index, edge_attr.float()


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


def cal_ectd(nnode, values, pinv):
    ectd = np.zeros((nnode, nnode), )

    for i, j in values:
        eij = np.zeros((nnode, 1), )
        eij[i, 0] = 1.
        eij[j, 0] = -1. if i != j else 0.
        ectd[i, j] = eij.T @ pinv @ eij

    ectd_norm = np.power(ectd, -1)
    ectd_norm[np.isinf(ectd_norm)] = 0
    ectd_norm[ectd_norm < 0.] = 0.

    return ectd_norm
