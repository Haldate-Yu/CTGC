import numpy as np
import networkx as nx
import scipy.linalg
import scipy.sparse as sp
from scipy.linalg import pinvh, pinv
from tqdm import tqdm
from scipy import optimize
import scipy.io as sio
import os

np.set_printoptions(formatter={'all': lambda x: str(x)}, threshold=100)


def aug_normalized_adjacency(adj, ectd_data, args):
    # self-loops
    # adj = adj + sp.eye(adj.shape[0])

    # temp adj + self loop
    temp = (adj + sp.eye(adj.shape[0])).toarray()

    if args.adj == 'A1':
        adj_2nd = temp
    elif args.adj == 'A1_2':
        adj_2nd = temp @ temp
    elif args.adj == 'A1_3':
        adj_2nd = temp @ temp @ temp
    elif args.adj == 'A1_4':
        adj_2nd = temp @ temp @ temp @ temp
    elif args.adj == 'A1_5':
        adj_2nd = temp @ temp @ temp @ temp @ temp
    elif args.adj == 'A1_6':
        adj_2nd = temp @ temp @ temp @ temp @ temp @ temp

    adj_2nd = np.where(adj_2nd <= 0, adj_2nd, 1.)

    # 3-layer hierachical
    # adj = adj + sp.eye(adj.shape[0])
    # 2-pow adj?
    # adj = adj.toarray()
    # split_list = [adj, adj @ adj, adj @ adj @ adj]
    # split_list = [np.where(a <= 0, a, 1.) for a in split_list]
    # for index in range(len(split_list) - 1, -1, -1):
    #     if index > 0:
    #         split_list[index] -= split_list[index - 1]

    # value_list = []
    # for adj in split_list:
    #     print(np.count_nonzero(adj))
    #     i, j = np.nonzero(adj)
    #     value_list.append(zip(i, j))

    '''
    ectd_data_path:
        ectd + 
        {dataset} +
        {is_vg} +
        {fun} +
        {An}
    '''
    # ectd_data = './pinv-dataset/ectd_cora_vg_log_A1.txt'
    # adj[i, j] = 1. record indexes
    print(np.count_nonzero(adj_2nd))
    # print(adj_2nd)
    i, j = np.nonzero(adj_2nd)
    values = zip(i, j)
    
    # deg = np.diag(adj_2nd.sum(1))
    deg = np.diag(adj.sum(1))
    
    # standard laplacian
    # lap = deg - adj

    # symmetric laplacian
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    # for temp
    # lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    # lap = np.around(lap, 3)
    # sio.savemat('lap_A3.mat', {'lap': lap})

    
    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj_2nd).dot(d_inv_sqrt)
    pinv = pinvh(lap)
    
    # ectd = calculate_ectd_hie(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg,
    #                       save=ectd_data, full_adj=adj_2nd)

    ectd = calculate_ectd(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg, save=ectd_data)
    ectd = np.around(ectd, 3)

    adj = ectd
    # adj = np.where(ectd > 0, adj_2nd, 0.)
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()  # D^-1/2*A*D^-1/2
    # return d_mat_inv.dot(adj).tocoo()  # A*D^-1
    # return adj.tocoo()


def rw_normalized_adjacency(adj):
    # self-loops
    adj = adj + sp.eye(adj.shape[0])

    # 2-pow adj?
    adj = adj.toarray()

    adj_2nd = adj
    adj_2nd = np.where(adj_2nd <= 0, adj_2nd, 1.)
    # adj[i, j] = 1. record indexes
    print(np.count_nonzero(adj_2nd))
    i, j = np.nonzero(adj_2nd)
    values = zip(i, j)

    deg = np.diag(adj_2nd.sum(1))
    vg = deg.sum().sum()

    # standard laplacian
    lap = deg - adj_2nd
    print("laplacian:\n {}".format(lap))

    # symmetric laplacian
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj_2nd).dot(d_inv_sqrt)
    print("symmetric laplacian:\n {}".format(lap))

    pinv = pinvh(lap)
    # ectd = calculate_ectd(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg)
    ectd = calculate_ectd_hie(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg)

    adj = ectd
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1))

    # D^-1
    d_inv = np.power(row_sum, -1.).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return adj.dot(d_mat_inv).tocoo()  # A*D^-1


def ori_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    # print(sp.coo_matrix(adj))
    adj = sp.coo_matrix(adj)

    # print(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'RWNormAdj': rw_normalized_adjacency,  # A' = ( A + I ) * (D + I)^-1
        'OriNormAdj': ori_normalized_adjacency,
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def calculate_ectd(nnodes, values, pinv, vg=1., save=None):
    if os.path.exists(save):
        ectd = np.loadtxt(save, delimiter=',')
    else:
        ectd = np.zeros((nnodes, nnodes), )

        for i, j in tqdm(values):
            eij = np.zeros((nnodes, 1), )
            eij[i, 0] = 1.
            eij[j, 0] = -1. if i != j else 0.

            ectd[i, j] = vg * eij.T @ pinv @ eij

        np.savetxt(save, ectd, fmt='%f', delimiter=',')

    # for some reason, part of the distance are negative, 
    # so just keep it original. 
    ectd[ectd < 0.] = 1.

    ectd_norm = np.power(ectd, -1.)
    ectd_norm[np.isinf(ectd_norm)] = 0.

    # np.fill_diagonal(ectd_norm, np.max(ectd, axis=1))
    np.fill_diagonal(ectd_norm, 1.)

    return ectd_norm


def calculate_ectd_hie(nnodes, values, pinv, vg=1., save=None, full_adj=None):
    # G = nx.from_numpy_array(full_adj)
    # spd = dict(nx.all_pairs_shortest_path_length(G))

    if os.path.exists(save):  # no save
        ectd = np.loadtxt(save, delimiter=',')
    else:
        ectd = np.zeros((nnodes, nnodes), )
        # adj_1, adj_2, adj_3 = list(split[0]), list(split[1]), list(split[2])

        for i, j in tqdm(values):
            eij = np.zeros((nnodes, 1), )
            eij[i, 0] = 1.
            eij[j, 0] = -1. if i != j else 0.

            ectd[i, j] = vg * eij.T @ pinv @ eij

            # if (i, j) in adj_1:
            # print('\n1-a\n')
            # ectd[i, j] = np.power((1 / ectd[i, j]), np.log(1.))
            # ectd[i, j] *= 100.
            # ectd[i, j] += spd[i][j]
            # elif (i, j) in adj_2:
            # print('\n2-a\n')
            # ectd[i, j] = np.power((1 / ectd[i, j]), np.log(2.))
            # ectd[i, j] *= 10.
            # ectd[i, j] = ectd[i, j] + 2 * spd[i][j]
            # elif (i, j) in adj_3:
            # print('\n3-a\n')
            # ectd[i, j] = np.power((1 / ectd[i, j]), np.log(3.))
            # ectd[i, j] = ectd[i, j] + 3 * spd[i][j]
            # ectd[i, j] = np.exp(-ectd[i, j]) if i != j else 0.
            # ectd[i, j] = eij.T @ pinv @ eij - vg if i != j else 0.
            # ectd[i, j] = np.log(ectd[i, j])
            # a - [0.4, 0.9]
            # a = 2 * np.e
            # ectd[i, j] = np.power(a, np.log(ectd[i, j]))
            # ectd[i, j] = np.power(a, ectd[i, j])
            # ectd[i, j] = np.exp(-np.log2(ectd[i, j]))
            # ectd[i, j] = np.sqrt(1. / ectd[i, j])
            # ectd[i, j] = alpha_exp(ectd[i, j])

            # sigmoid
            # ectd[i, j] = sigmoid(-ectd[i, j])
            # ectd[i, j] = sigmoid(1. / ectd[i, j])

            # tanh
            # ectd[i, j] = np.tanh(1. / ectd[i, j])
            # ectd[i, j] = 1. / np.tanh(ectd[i, j])

            # arctan
            # ectd[i, j] = arctan(1. / ectd[i, j])
            # ectd[i, j] = cos(ectd[i, j])
            # ectd[i, j] = arctan(1. / ectd[i, j])
        np.savetxt(save, ectd, fmt='%f', delimiter=',')

    # reciprocal
    ectd = np.power(ectd, -1.)
    ectd[ectd < 0.] = 0.
    ectd[np.isinf(ectd)] = 0.
    # set diagonal to 1
    np.fill_diagonal(ectd, 1.)
    # print(np.max(ectd, axis=1))
    # np.fill_diagonal(ectd, np.max(ectd, axis=1))

    # ectd_norm = (ectd - min) / (ectd.max() - min)
    # ectd_norm[ectd_norm < 0] = 0.
    # max-min normalization
    # ectd_norm = (ectd - ectd.max()) / (min - ectd.max())
    # ectd_norm[ectd_norm == (ectd.max() / (ectd.max() - min))] = 0.

    # row-sum normalization
    # ectd_norm = row_normalize(ectd)

    # no normalization
    ectd_norm = ectd

    return ectd_norm
