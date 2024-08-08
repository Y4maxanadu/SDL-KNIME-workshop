import torch
import numpy as np

from torch_geometric.utils import structured_negative_sampling
import model as m


def get_metrics(model, edge_index, exclude_edge_indices, k):
    """
    get the evaluation metrics: recall@k, precision@k and ndcg@k
    :param model:
    :param edge_index:
    :param exclude_edge_indices:
    :param k:
    :return:
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = m.get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -1024

    # get the top k recommended items for each user
    _, top_k_items = torch.topk(rating, k=k)

    users = edge_index[0].unique()

    test_user_pos_items = m.get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_k_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = m.recall_precision_at_K_r(test_user_pos_items_list, r, k)
    ndcg = m.NDCGat_K_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg


def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """

    :param model:
    :param edge_index:
    :param sparse_edge_index:
    :param exclude_edge_indices:
    :param k:
    :param lambda_val:
    :return:
    """

    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                               pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                               neg_item_indices], items_emb_0[neg_item_indices]

    loss = m.bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg


def evaluation_p(model, edge_index, sparse_edge_index, sparse_edge_index_p, exclude_edge_indices, k, lambda_val):
    """

    :param model:
    :param edge_index:
    :param sparse_edge_index:
    :param exclude_edge_indices:
    :param k:
    :param lambda_val:
    :return:
    """

    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index,sparse_edge_index_p)
    edges = structured_negative_sampling(
        edge_index)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                               pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                               neg_item_indices], items_emb_0[neg_item_indices]

    loss = m.bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg