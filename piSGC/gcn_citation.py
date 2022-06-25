import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)

data_path = './pinv-dataset/ectd_' + args.dataset + '_'
if args.using_vg: data_path += 'vg' + str(args.vg) + '_'
data_path += args.method + '_' + args.adj + '.txt'

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args, args.normalization, args.cuda,
                                                                    data_path)

model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC":
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))


def train_regression(model, adj, features,
                     idx_train, train_labels,
                     idx_val, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        acc_val = accuracy(output[idx_val], val_labels)

    return model, acc_val, train_time


def test_regression(model, adj, features, idx_test, test_labels):
    model.eval()
    return accuracy(model(features, adj)[idx_test], test_labels)


if args.model == "GCN":
    model, acc_val, train_time = train_regression(model, adj, features, idx_train, labels[idx_train], idx_val,
                                                  labels[idx_val],
                                                  args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, adj, features, idx_test, labels[idx_test])

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
# print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
