from abc import ABC

import torch_sparse

import dataloader as dl
import torch
import random
from torch_geometric.nn.conv import MessagePassing
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling, subgraph
from torch_geometric.data import Data

user_mapping = dl.get_user_mapping()
item_mapping = dl.get_resume_mapping()

num_users = dl.get_num_users(user_mapping)
num_items = dl.get_num_resumes(item_mapping)


class LightGCN(MessagePassing):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=64,
            K_layer=3,
            add_self_loops=False):
        """

        :param num_users:
        :param num_items:
        :param embedding_dim:
        :param K_layer:
        :param add_self_loops:
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        """
        gcn_norm
        A_hat= = ( D_hat ) ^ ( -1/2 ) * (A + I) * ( D_hat ) ^ ( 1/2 )
        where ( D_hat )_ii = Sigma_( j = 0 ) ( A_hat )_ij + 1

        This is the process of calculating the symmetrical normalized adjacency matrix
        """
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        # print(edge_index_norm)
        for i in range(self.K_layer):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        # split into  e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class LiuziGCN(MessagePassing):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=64,
            K_layer=3,
            slice_size=2,
            add_self_loops=False):
        """

        :param num_users:
        :param num_items:
        :param embedding_dim:
        :param K_layer:
        :param add_self_loops:
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops
        self.slice_size = slice_size
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor):

        edge_indices = liuzi_tensor(edge_index,slice_size=self.slice_size)
        # edge_indices = liuzi_tensor1(edge_index)
        edge_index_norms = []

        for i in range(self.slice_size):
            norm = gcn_norm(edge_indices[i], add_self_loops=self.add_self_loops)
            edge_index_norms.append(norm)

        """
            gcn_norm
            A_hat= = ( D_hat ) ^ ( -1/2 ) * (A + I) * ( D_hat ) ^ ( 1/2 )
            where ( D_hat )_ii = Sigma_( j = 0 ) ( A_hat )_ij + 1

            This is the process of calculating the symmetrical normalized adjacency matrix
        """
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        emb_ks = []

        for i in range(self.K_layer):
            for j in range(self.slice_size):
                emb_ks.append(self.propagate(edge_index_norms[j], x=emb_k))
                embs.append(emb_ks[j])

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        # split into  e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class kobe(MessagePassing):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=64,
            K_layer=3,
            add_self_loops=False):
        """

        :param num_users:
        :param num_items:
        :param embedding_dim:
        :param K_layer:
        :param add_self_loops:
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor, edge_index_p: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        edge_index_norm_p = gcn_norm(edge_index_p, edge_weight=0.02,add_self_loops=self.add_self_loops)
        """
         print("edge_index", edge_index)
        print("edge_index_p", edge_index_p)
        print("eee", edge_index+edge_index_p)
       
        print("norm", edge_index_norm)
        print("norm_p", edge_index_norm_p)
        print("nnn", edge_index_norm+edge_index_norm_p)
        print("ee_norm", ee_norm) 
        """

        ee_norm = edge_index_norm_p+edge_index_norm #gcn_norm(edge_index+edge_index_p, add_self_loops=self.add_self_loops)

        """
        gcn_norm
        A_hat= = ( D_hat ) ^ ( -1/2 ) * (A + I) * ( D_hat ) ^ ( 1/2 )
        where ( D_hat )_ii = Sigma_( j = 0 ) ( A_hat )_ij + 1

        This is the process of calculating the symmetrical normalized adjacency matrix
        """
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        # print(edge_index_norm)
        for i in range(self.K_layer):
            emb_k = self.propagate(ee_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        # split into  e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)



def sample_mini_batch(batch_size, edge_index):
    """
    Randomly samples indices of a minibatch given an adjacency matrix
    :param batch_size: (int) mini batch size
    :param edge_index:  (torch.Tensor) 2 by N list of edges
    :return:tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(users_emb_k, users_emb_0, pos_item_emb_k, pos_item_emb_0, neg_item_emb_k, neg_item_emb_0, lambda_val):
    """

    :param users_emb_k:
    :param users_emb_0:
    :param pos_item_emb_k:
    :param pos_item_emb_0:
    :param neg_item_emb_k:
    :param neg_item_emb_0:
    :param lambda_val:
    :return:
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) + pos_item_emb_0.norm(2).pow(2) + neg_item_emb_0.norm(2).pow(2))

    # predicted scores of positive and negative samples
    pos_scores = torch.mul(users_emb_k, pos_item_emb_k)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_k, neg_item_emb_k)
    neg_scores = torch.sum(neg_scores, dim=-1)
    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


def get_user_positive_items(edge_index):
    """

    :param edge_index:
    :return:
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def get_user_negative_items(edge_index):
    """

    :param edge_index:
    :return:
    """
    user_neg_items = {}
    """
    to be completed 
    """
    return user_neg_items


def recall_precision_at_K_r(ground_truth, r, k):
    """

    :param ground_truth: (list) of lists containing highly rated items
    :param r: (list) of lists indicating whether each top k item recommended to each user is a top k ground truth item or not
    :param k: top k
    :return:
    """
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(ground_truth[i]) for i in range(len(ground_truth))])
    recall = torch.mean(num_correct_pred / user_num_liked).item()
    precision = torch.mean(num_correct_pred) / k
    return recall, precision.item()


def NDCGat_K_r(ground_truth, r, k):
    """

    :param groundTruth:
    :param r:
    :param k:
    :return:
    """
    assert len(r) == len(ground_truth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(ground_truth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r / torch.log2(torch.arange(2, k + 2)), dim=1)
    dcg = r / torch.log2(torch.arange(2, k + 2))
    dcg = torch.sum(dcg, dim=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def liuzi_tensor(edge_index: torch_sparse.SparseTensor, slice_size: int):
    row_tensor = edge_index.storage.row()
    col_tensor = edge_index.storage.col()

    slices = len(row_tensor) // slice_size
    num_elements = slices * slice_size

    row_list = row_tensor.tolist()
    col_list = col_tensor.tolist()

    # shuffle tensor
    random_seed = 1
    random.seed(random_seed)
    random.shuffle(row_list)
    random.seed(random_seed)
    random.shuffle(col_list)

    row_tensor_shuffled = torch.tensor(row_list)
    col_tensor_shuffled = torch.tensor(col_list)

    row_tensor = row_tensor_shuffled[:num_elements]
    col_tensor = col_tensor_shuffled[:num_elements]

    row_tensor_subsets = [row_tensor[i * slices: (i + 1) * slices] for i in range(slice_size)]
    col_tensor_subsets = [col_tensor[i * slices: (i + 1) * slices] for i in range(slice_size)]

    sp_len = num_items + num_users

    merged_sparse_tensor = []
    for i in range(slice_size):
        merged_sparse_tensor.append(
            SparseTensor(row=row_tensor_subsets[i], col=col_tensor_subsets[i], sparse_sizes=(sp_len, sp_len)))

    return merged_sparse_tensor


def liuzi_tensor1(edge_index: torch_sparse.SparseTensor):

    row_tensor = edge_index.storage.row()
    col_tensor = edge_index.storage.col()

    subset_size = len(row_tensor) // 4
    num_elements = subset_size * 4

    row_list = row_tensor.tolist()
    col_list = col_tensor.tolist()

    random_seed = 1

    random.seed(random_seed)
    random.shuffle(row_list)

    random.seed(random_seed)
    random.shuffle(col_list)

    tensor1_shuffled = torch.tensor(row_list)
    tensor2_shuffled = torch.tensor(col_list)

    row_tensor = tensor1_shuffled[:num_elements]
    col_tensor = tensor2_shuffled[:num_elements]

    subset1_tensor1 = row_tensor[:subset_size]
    subset1_tensor2 = col_tensor[:subset_size]

    subset2_tensor1 = row_tensor[subset_size:2 * subset_size]
    subset2_tensor2 = col_tensor[subset_size:2 * subset_size]

    subset3_tensor1 = row_tensor[2 * subset_size:3 * subset_size]
    subset3_tensor2 = col_tensor[2 * subset_size:3 * subset_size]

    subset4_tensor1 = row_tensor[3 * subset_size:]
    subset4_tensor2 = col_tensor[3 * subset_size:]

    sp_len = num_items + num_users

    sp1 = SparseTensor(row=subset1_tensor1, col=subset1_tensor2, sparse_sizes=(sp_len, sp_len))
    sp2 = SparseTensor(row=subset2_tensor1, col=subset2_tensor2, sparse_sizes=(sp_len, sp_len))
    sp3 = SparseTensor(row=subset3_tensor1, col=subset3_tensor2, sparse_sizes=(sp_len, sp_len))
    sp4 = SparseTensor(row=subset4_tensor1, col=subset4_tensor2, sparse_sizes=(sp_len, sp_len))

    return sp1, sp2, sp3, sp4
