import model as m
import dataloader as dl
import torch
from torch import optim
import matplotlib.pyplot as plt

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import utils

ITERATIONS = 6000
LR = 1e-3
BATCH_SIZE = 1024
LAMBDA = 1e-6
ITERS_INTERVAL = 200
ITERS_PER_LR_DECAY = 200
K = 10


def record_model(model_name, iterations, lr, batch_size, lambda_val, iters_interval, iters_per_lr_decay, k,
                 test_results, file_path):
    """

    :param model_name:
    :param iterations:
    :param lr:
    :param batch_size:
    :param lambda_val:
    :param iters_interval:
    :param iters_per_lr_decay:
    :param k:
    :param test_results:
    :param file_path:
    :return:
    """
    model_info = f"Model Name: {model_name}\n"
    model_info += f"ITERATIONS: {iterations}\n"
    model_info += f"LR: {lr}\n"
    model_info += f"BATCH_SIZE: {batch_size}\n"
    model_info += f"LAMBDA: {lambda_val}\n"
    model_info += f"ITERS_INTERVAL: {iters_interval}\n"
    model_info += f"ITERS_PER_LR_DECAY: {iters_per_lr_decay}\n"
    model_info += f"K: {k}\n\n"
    model_info += f"Test Results:\n{test_results}\n"
    with open(file_path, 'a') as f:
        f.write(model_info)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('kobe')

    user_mapping = dl.get_user_mapping()
    item_mapping = dl.get_resume_mapping()

    n1, n2 = dl.get_num_users(user_mapping), dl.get_num_resumes(item_mapping)
    model = m.LiuziGCN(num_users=n1, num_items=n2)

    #model = m.kobe(num_users=n1, num_items=n2)
    # model = m.LightGCN(num_users=n1, num_items=n2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = dl.get_edge_index()
    edge_index = edge_index.to(device)

    [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index,
     train_edge_index, val_edge_index, test_edge_index] = dl.get_sparse_edge_index(n1, n2)
    """
    [[train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
     test_edge_index], [train_sparse_edge_index_p, val_sparse_edge_index_p, test_sparse_edge_index_p,
                        train_edge_index_p, val_edge_index_p,
                        test_edge_index_p]] = dl.get_sparse_dynamic_edge_index(n1, n2)
    """
    train_sparse_edge_index = train_sparse_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)
    # test_sparse_edge_index = train_sparse_edge_index.to(device)

    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    # test_edge_index =test_edge_index.to(device)

    # training loop
    epoch = 0
    epoches = []
    train_losses = []
    val_losses = []

    edge_index = edge_index.to(device)

    for iter in range(ITERATIONS):
        # forward propagating
        users_emb_k, users_emb_0, item_emb_k, items_emb_0 = model.forward(train_sparse_edge_index)
        #users_emb_k, users_emb_0, item_emb_k, items_emb_0 = model.forward(train_sparse_edge_index, train_sparse_edge_index_p)
        # mini batching
        user_indices, pos_item_indices, neg_item_indices = m.sample_mini_batch(BATCH_SIZE, train_edge_index)

        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        neg_item_indices = neg_item_indices.to(device)

        users_emb_k, users_emb_0 = users_emb_k[user_indices], users_emb_0[user_indices]
        pos_item_emb_k, pos_item_emb_0 = item_emb_k[pos_item_indices], items_emb_0[pos_item_indices]
        neg_item_emb_k, neg_item_emb_0 = item_emb_k[neg_item_indices], items_emb_0[neg_item_indices]

        # loss computing
        train_loss = m.bpr_loss(users_emb_k, users_emb_0, pos_item_emb_k, pos_item_emb_0, neg_item_emb_k,
                                neg_item_emb_0, LAMBDA)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % ITERS_INTERVAL == 0:
            model.eval()
            val_loss, recall, precision, ndcg = utils.evaluation(model, val_edge_index, val_sparse_edge_index,
                                                                [train_edge_index], K, LAMBDA)

            #val_loss, recall, precision, ndcg = utils.evaluation_p(model, val_edge_index, val_sparse_edge_index, val_sparse_edge_index_p,[train_edge_index], K, LAMBDA)
            print(
                f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            epoch += 1
            epoches.append(epoch)
            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

    plt.plot(epoches, train_losses, label='train')
    plt.plot(epoches, val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.show()

    model_name = input("model_name = ")
    if model_name == 'kobe':
        print('Mamba out, we can\'t save the model which named kobe.')
        exit()
    torch.save(model.state_dict(), './model/' + model_name + '.pt')

    model.eval()
    test_edge_index = test_edge_index.to(device)
    test_sparse_edge_index = test_sparse_edge_index.to(device)

    test_loss, test_recall, test_precision, test_ndcg = utils.evaluation(model, test_edge_index, test_sparse_edge_index,
                                                                      [train_edge_index, val_edge_index], K, LAMBDA)
    #test_loss, test_recall, test_precision, test_ndcg = utils.evaluation_p(model, test_edge_index, test_sparse_edge_index, test_sparse_edge_index_p, [train_edge_index, val_edge_index], K, LAMBDA)

    test_results = f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}"

    file_path = "model_data.txt"
    record_model(model_name, ITERATIONS, LR, BATCH_SIZE, LAMBDA, ITERS_INTERVAL, ITERS_PER_LR_DECAY, K, test_results,
                 file_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
