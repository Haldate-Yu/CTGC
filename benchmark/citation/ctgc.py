import argparse

import torch
import torch.nn.functional as F
from datasets import get_planetoid_dataset, get_amazon_dataset, get_coauthor_dataset
from train_eval import random_planetoid_splits, run

from layers import CTGConv
from utils import logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--aggr_type', type=str, default='sgc', choices=['sgc', 'ssgc', 'ssgc_no_avg'])
parser.add_argument('--norm_type', type=str, default='arctan', choices=['arctan', 'f2', 'none'])
parser.add_argument('--pinv', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = CTGConv(dataset.num_features, dataset.num_classes,
                             K=args.K, alpha=args.alpha, aggr_type=args.aggr_type, norm_type=args.norm_type,
                             cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
elif args.dataset in ['Computers', 'Photo']:
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)
elif args.dataset in ['CS', 'Physics']:
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)

if args.pinv:
    print('using pinv!')
permute_masks = random_planetoid_splits if args.random_splits else None
loss, acc, duration = run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
                          args.early_stopping, permute_masks, pinv=args.pinv, dataname=args.dataset)

logger(Net(dataset).conv1, args, loss, acc, duration)
