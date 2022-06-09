# @Time     : 2022/6/6
# @Author   : Haldate
import numpy as np
from numpy.linalg import norm


def similarity(matA, matB):
    if np.allclose(matA, matB):
        return 100.0

    flatA = matA.ravel()
    flatB = matB.ravel()
    sim = np.sum(flatA * flatB) / (norm(flatA) * norm(flatB))
    return sim * 100.0


file1 = './mat/cora/ori_cora.txt'
file2 = './mat/cora/pinv_cora.txt'

ori_mat = np.loadtxt(file1, delimiter=',')
pinv_mat = np.loadtxt(file2, delimiter=',')

print(similarity(ori_mat, pinv_mat))
