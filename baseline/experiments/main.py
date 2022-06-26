import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import utils

import torch.nn.functional as F
import argparse
import os
import random
# from texttable import Texttable
from torch.utils.data import random_split

from tqdm import tqdm, trange
from transformers.optimization import get_cosine_schedule_with_warmup
from functools import reduce

from networks import Net_GCN, Net_GIN, Net_GIN_W, Net_SGC, Net_SSGC, Net_CTGC
from data import get_dataset, num_graphs

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument('--seed_number', type=int, default=10,
                    help='seed number')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=50000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')

parser.add_argument("--normalize", action='store_true')
parser.add_argument("--lr-schedule", action='store_true')

parser.add_argument('--num_layers', type=int, default=3,
                    help='num_layers')

parser.add_argument('--cuda', type=int, default=0,
                    help='cuda number')
parser.add_argument('--model', type=str, default='gcn',
                    help='gcn/gin/sgc')
parser.add_argument('--K', type=int, default='2',
                    help='sgc layers')
parser.add_argument('--alpha', type=float, default='0.1',
                    help='percentage')
parser.add_argument('--aggr_type', type=str, default='ssgc',
                    help='sgc/ssgc/ssgc_no_avg')
parser.add_argument('--norm_type', type=str, default='arctan',
                    help='arctan/f2/none')
parser.add_argument('--pinv', type=bool, default=False)
parser.add_argument('--topk', type=int, default='10',
                    help='topk rate of the matrix(10/20/30)')

parser.add_argument('--train_label_ratio', type=float, default=0.1,
                    help='label ratio')
parser.add_argument('--data_split', type=str, default='standard',
                    help='standard/random')

args = parser.parse_args()
# args.device = 'cpu'
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)
#     args.device = 'cuda:0'


# def tab_printer(args):
#     """
#     Function to print the logs in a nice tabular format.
#     :param args: Parameters used for the model.
#     """
#     args = vars(args)
#     keys = sorted(args.keys())
#     t = Texttable()
#     t.add_rows([["Parameter", "Value"]])
#     t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
#     print(t.draw())

dataset = get_dataset(args.dataset, normalize=args.normalize, pinv=args.pinv, topk=args.topk)
args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
    np.mean([data.num_nodes for data in dataset]))
print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' % (
    dataset, args.num_features, args.num_classes, args.avg_num_nodes))

log_folder_name = os.path.join(*[args.dataset])

if not (os.path.isdir('./results/{}'.format(log_folder_name))):
    os.makedirs(os.path.join('./results/{}'.format(log_folder_name)))

    print("Make Directory {} in  Results Folders".format(log_folder_name))


def load_dataloader(fold_number, val_fold_number, generator):
    train_idxes = torch.as_tensor(np.loadtxt('../datasets/%s/10fold_idx/train_idx-%d.txt' % (args.dataset, fold_number),
                                             dtype=np.int32), dtype=torch.long)
    val_idxes = torch.as_tensor(
        np.loadtxt('../datasets/%s/10fold_idx/test_idx-%d.txt' % (args.dataset, val_fold_number),
                   dtype=np.int32), dtype=torch.long)
    test_idxes = torch.as_tensor(np.loadtxt('../datasets/%s/10fold_idx/test_idx-%d.txt' % (args.dataset, fold_number),
                                            dtype=np.int32), dtype=torch.long)

    all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
    assert len(all_idxes) == len(dataset)

    train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))

    train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              worker_init_fn=seed_worker,
                              generator=generator
                              )
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=generator
                            )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=generator
                             )
    # pin_memory = True
    # num_workers = 8,
    return train_loader, val_loader, test_loader


def load_dataloader_random(fold_number, val_fold_number, generator):
    num_training = int(len(dataset) * args.train_label_ratio)
    num_val = int(len(dataset) * 0.1)
    num_test = int(len(dataset) * 0.1)

    train_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              worker_init_fn=seed_worker,
                              generator=generator
                              )
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=generator
                            )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=generator
                             )
    # pin_memory = True
    # num_workers = 8,
    return train_loader, val_loader, test_loader


def test(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def train(model, optimizer, train_loader, val_loader, scheduler, ):
    min_loss = 1e10
    patience = 0

    # epoch_fold_iter = tqdm(range(0, args.epochs), desc='[Epoch]', position = 1)
    # for epoch in trange(0, (args.epochs), desc = '[Epoch]', position = 1):
    for epoch in range(0, args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            # print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            if args.lr_schedule:
                scheduler.step()

            optimizer.zero_grad()
        val_acc, val_loss = test(model, val_loader)
        # print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))

        # epoch_fold_iter.refresh()
        if val_loss < min_loss:
            torch.save(model.state_dict(),
                       './results/{}/latest_{}_{}.pth'.format(args.dataset, args.dataset, args.model))
            # print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def model_train_and_val(dataset, seed):
    args.device = 'cpu'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args.device = 'cuda:{}'.format(args.cuda)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)

    test_acc_lst = []

    if args.data_split == "standard":

        train_fold_iter = tqdm(range(1, 11), desc='Training')
        val_fold_iter = [i for i in range(1, 11)]

    elif args.data_split == "random":
        train_fold_iter = tqdm(range(1, 2), desc='Training')
        val_fold_iter = [i for i in range(1, 2)]

    for fold_number in train_fold_iter:

        if args.data_split == "standard":
            val_fold_number = val_fold_iter[fold_number - 2]
            train_loader, val_loader, test_loader = load_dataloader(fold_number, val_fold_number, generator)

        elif args.data_split == "random":
            val_fold_number = val_fold_iter[fold_number - 1]
            train_loader, val_loader, test_loader = load_dataloader_random(fold_number, val_fold_number, generator)

        if args.model == "gcn":
            model = Net_GCN(args).to(args.device)
        elif args.model == "gin":
            model = Net_GIN(args).to(args.device)
        elif args.model == 'gin_w':
            model = Net_GIN_W(args).to(args.device)
        elif args.model == 'sgc':
            model = Net_SGC(args).to(args.device)
        elif args.model == 'ssgc':
            model = Net_SSGC(args).to(args.device)
        elif args.model == 'ctgc':
            model = Net_CTGC(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.lr_schedule:
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
                                                        args.epochs * len(train_loader))
        else:
            scheduler = None

        train(model, optimizer, train_loader, val_loader, scheduler)

        if args.model == "gcn":
            model = Net_GCN(args).to(args.device)
        elif args.model == "gin":
            model = Net_GIN(args).to(args.device)
        elif args.model == 'gin_w':
            model = Net_GIN_W(args).to(args.device)
        elif args.model == 'sgc':
            model = Net_SGC(args).to(args.device)
        elif args.model == 'ssgc':
            model = Net_SSGC(args).to(args.device)
        elif args.model == 'ctgc':
            model = Net_CTGC(args).to(args.device)

        model.load_state_dict(
            torch.load('./results/{}/latest_{}_{}.pth'.format(args.dataset, args.dataset, args.model)))
        test_acc, test_loss = test(model, test_loader)

        test_acc_lst.append(test_acc)

        train_fold_iter.refresh()

    return test_acc_lst


def ran_exp_grid(dataset, seed_lst):
    print("Start Training!")

    test_acc_total_lst = []
    for index, seed in enumerate(seed_lst):

        print(" Training cycle {}, Seed:{}".format(index, seed))
        test_acc_lst = model_train_and_val(dataset, seed)
        if args.model in ['gcn', 'gin', 'gin_w']:
            summary_1 = 'Model={}, Data Split={}, Dataset={}, Seed={},'.format(
                args.model, args.data_split, args.dataset, seed)
            if args.pinv:
                summary_1 += ' Using Ectd TOPK={}'.format(args.topk)
        elif args.model in ['sgc', 'ssgc']:
            summary_1 = "Model={}, Layers={}, Data Split={}, Dataset={}, Seed={}".format(
                args.model, args.K, args.data_split, args.dataset, seed)
        elif args.model in ['ctgc']:
            summary_1 = "Model={}, Layers={}, Alpha={}, Aggr_type={}, Norm_type={}, Data Split={}, Dataset={}, Seed={}".format(
                args.model, args.K, args.alpha, args.aggr_type, args.norm_type, args.data_split, args.dataset, seed)

        summary_2 = 'test_acc_main={:.2f} ± {:.2f}'.format(
            np.mean(test_acc_lst) * 100, np.std(test_acc_lst) * 100)
        print('{} : \n{} \n{}: \n{}'.format('Model details', summary_1, "Final result", summary_2))

        final_result_file = "./results/{}/{}-results.txt".format(args.dataset, args.model)
        with open(final_result_file, 'a+') as f:
            f.write('{} :  {} \n{} :  {}\n{}\n'.format(
                'Model details',
                summary_1,
                "Final result",
                summary_2,
                "-" * 30)
            )

        print("Seed {} results have been stored! ".format(seed))

        test_acc_total_lst.append(np.mean(test_acc_lst))

    final_total_result_file = "./results/total_results.txt"

    summary_1_total = 'Model={},Data Split={}, Dataset={}, Num Layers={}'.format(
        args.model, args.data_split, args.dataset, args.num_layers)
    if args.model in ['gcn', 'gin', 'gin_w']:
        summary_1_total = 'Model={}, Data Split={}, Dataset={}, Num Layers={},'.format(
            args.model, args.data_split, args.dataset, args.num_layers)
        if args.pinv:
            summary_1_total += 'Using ECTD TopK={}'.format(args.topk)
    elif args.model in ['sgc', 'ssgc']:
        summary_1_total = "Model={}, Num Layers={}, Data Split={}, Dataset={}".format(
            args.model, args.K, args.data_split, args.dataset)
    elif args.model in ['ctgc']:
        summary_1_total = "Model={}, Num Layers={}, Alpha={}, Aggr_type={}, Norm_type={}, Data Split={}, Dataset={}".format(
            args.model, args.K, args.alpha, args.aggr_type, args.norm_type, args.data_split, args.dataset)
    summary_2_total = 'test_acc_main={:.2f} ± {:.2f}'.format(
        np.mean(test_acc_total_lst) * 100, np.std(test_acc_total_lst) * 100)

    with open(final_total_result_file, 'a+') as f:
        f.write('{} : {} \n{}: {}\n{}\n'.format(
            'Model details',
            summary_1_total,
            "Final result",
            summary_2_total,
            "-" * 30)
        )


if __name__ == '__main__':
    seed_lst = [i + 42 for i in range(args.seed_number)]
    # tab_printer(args)
    if args.pinv is True:
        print('using pinv!')
    ran_exp_grid(dataset, seed_lst)
