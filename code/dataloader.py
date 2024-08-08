import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_sparse import SparseTensor

resume_path = 'data/movie/movie.csv'
rating_path = 'data/movie/rating.csv'
print('From data: ' + rating_path)


def load_node_csv(path, index_col):
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=6):
    print('we are using rating threshold', rating_threshold)

    df = pd.read_csv(path)
    src, dst = [src_mapping[index] for index in df[src_index_col]], [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold
    #print('edge_attr', edge_attr, 'src', src, 'dst', dst)
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    print("there are", len(edge_index[0]), 'edges')
    return torch.tensor(edge_index)


def load_dynamic_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col,
                          rating_threshold=6):
    print('we are using rating threshold', rating_threshold)
    df = pd.read_csv(path)
    src, dst = [src_mapping[index] for index in df[src_index_col]], [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= 6
    edge_attr_p = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) > 8
    edge_index = [[], []]
    edge_index_p = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    print("there are", len(edge_index[0]), 'edges')

    for i in range(edge_attr_p.shape[0]):
        if edge_attr_p[i]:
            edge_index_p[0].append(src[i])
            edge_index_p[1].append(dst[i])
    print("there are", len(edge_index_p[0]), 'dynamic_edges')
    return torch.tensor(edge_index), torch.tensor(edge_index_p)


def get_user_mapping():
    return load_node_csv(rating_path, index_col='user_id')


def get_resume_mapping():
    return load_node_csv(resume_path, index_col='resume_id')


def get_num_users(user_mapping):
    return len(user_mapping)


def get_num_resumes(resume_mapping):
    return len(resume_mapping)


def get_edge_index():
    edge_index = load_edge_csv(
        rating_path,
        src_mapping=get_user_mapping(),
        src_index_col='user_id',
        dst_mapping=get_resume_mapping(),
        dst_index_col='resume_id',
        link_index_col='rating',
        rating_threshold=4
    )
    return edge_index


def get_dynamic_edge_index():
    edge_index_p = load_dynamic_edge_csv(
        rating_path,
        src_mapping=get_user_mapping(),
        src_index_col='user_id',
        dst_mapping=get_resume_mapping(),
        dst_index_col='resume_id',
        link_index_col='rating',
        rating_threshold=6
    )
    return edge_index_p


def get_sparse_edge_index(n1, n2):
    edge_index = get_edge_index()
    num_interactions = edge_index.shape[1]

    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=None)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=None)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    # print("Train edge Index", train_edge_index)

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1],
                                         sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(n1 + n2, n1 + n2))
    #print(test_sparse_edge_index)

    return [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
            test_edge_index]


def get_sparse_dynamic_edge_index(n1, n2):
    edge_index, edge_index_p = get_dynamic_edge_index()
    num_interactions, num_interactions_p = edge_index.shape[1], edge_index_p.shape[1]

    all_indices, all_indices_p = [i for i in range(num_interactions)], [i for i in range(num_interactions_p)]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=None)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=None)

    train_indices_p, test_indices_p = train_test_split(all_indices_p, test_size=0.3, random_state=None)
    val_indices_p, test_indices_p = train_test_split(test_indices_p, test_size=0.5, random_state=None)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    train_edge_index_p = edge_index_p[:, train_indices_p]
    val_edge_index_p = edge_index_p[:, val_indices_p]
    test_edge_index_p = edge_index_p[:, test_indices_p]

    # print("Train edge Index", train_edge_index)

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1],
                                         sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(n1 + n2, n1 + n2))

    train_sparse_edge_index_p = SparseTensor(row=train_edge_index_p[0], col=train_edge_index_p[1],
                                             sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index_p = SparseTensor(row=val_edge_index_p[0], col=val_edge_index_p[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index_p = SparseTensor(row=test_edge_index_p[0], col=test_edge_index_p[1],
                                            sparse_sizes=(n1 + n2, n1 + n2))

    #print(test_sparse_edge_index)

    return [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
            test_edge_index],[train_sparse_edge_index_p, val_sparse_edge_index_p, test_sparse_edge_index_p, train_edge_index_p, val_edge_index_p,
            test_edge_index_p]
