import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])    # add self loop
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

def load_data(args,path="./data/user_data/", dataset="all_feat_with_label"):

    print('Loading {} dataset...'.format(dataset))

    # read feature
    idx_features_labels = np.genfromtxt("{}{}.csv".format(path, dataset),
                                        dtype=np.dtype(str), delimiter=',', skip_header=1)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = np.array(idx_features_labels[:, -1],dtype=np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.object)

    if args.edge==1:
        adj = sp.load_npz(path+'node_adj_sparse.npz')
    else:
        adj = sp.load_npz(path+'node_adj_sparse_no_edge.npz')
    adj = adj.toarray()
    adj=sp.coo_matrix(adj)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)

    adj = normalize(adj + sp.eye(adj.shape[0]))


    idx_index=range(len(idx))
    X_train_val, idx_test, y_train_val, y_test = \
        train_test_split(idx_index, labels, stratify=labels, test_size=1 - args.train_size-args.val_size,
                         random_state=48, shuffle=True)
    idx_train, idx_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      stratify=y_train_val,
                                                      train_size=args.train_size/(args.train_size+args.val_size),
                                                      random_state=48,
                                                      shuffle=True)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
