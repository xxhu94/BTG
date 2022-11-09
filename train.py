from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("/code/BTG/")
from utils import load_data, accuracy
from models import GCN
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score, roc_auc_score, average_precision_score,classification_report,precision_recall_curve

import  logging
import datetime
def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
# logging setting
log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=24,
    # filename=log_name+'.log',
    # filemode='a'
    )

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_size', type=float, default=0.6,
                    help='training set percentage')
parser.add_argument('--val_size', type=float, default=0.2,
                    help='training set percentage')
parser.add_argument('--edge', type=int, default=1,
                    help='load graph with edge or not,1 for edge,0 for no edge')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    loss_history.append(loss_train.item())
    val_acc_history.append(acc_train.item())


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    f1=f1_score(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1), average='binary')



    forest_auc=roc_auc_score(labels[idx_test].cpu(), output.data[idx_test][:, 1].cpu())

    report = classification_report(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1), target_names=['0','1'], digits=4)
    print("Test set results:\n",
          "train_size:{}\n".format(args.train_size),
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "F1= {:.4f}".format(f1.item()),
          "Auc= {:.4f}".format(forest_auc.item()),
          "\nReport=\n{}".format(report))
    # ===============for batch test=================
    recall=recall_score(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1), average='macro')
    precision=precision_score(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1), average='macro')
    acc=accuracy_score(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1))
    f1=f1_score(labels[idx_test].cpu().detach().numpy(), output[idx_test].cpu().detach().numpy().argmax(axis=1), average='macro')
    logging.log(23,f"train_size= {args.train_size}")
    logging.log(24,f"AUC:{forest_auc:.4f},recall:{recall:.4f},precision:{precision:.4f},accuracy:{acc:.4f},F1:{f1:.4f}")
    logging.log(23,f"\nReport=:\n {report}")


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy{:.4f}s'.format(time.time() - t_total))
    plt.show()


# Train model
t_total = time.time()

loss_history = []
val_acc_history = []
for epoch in range(args.epochs):
    train(epoch)


plot_loss_with_acc(loss_history, val_acc_history)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
test()