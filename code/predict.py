import torch
import pandas as pd
import model as m
import dataloader as dl


resume_path = 'data/resume/resume.csv'
rating_path = 'data/resume/rating.csv'
print('Using data: ' + rating_path)
'''
resume_path = 'data/movie/movie.csv'
rating_path = 'data/movie/rating.csv'

'''

print('Let\'s find it out.')

user_mapping = dl.load_node_csv(rating_path, index_col='user_id')
resume_mapping = dl.load_node_csv(resume_path, index_col='resume_id')

edge_index = dl.get_edge_index()
user_pos_items = m.get_user_positive_items(edge_index)

num_users, num_resumes = len(user_mapping), len(resume_mapping)
num_interactions = edge_index.shape[1]

train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index = \
    dl.get_sparse_edge_index(num_users, num_resumes)[0], dl.get_sparse_edge_index(num_users, num_resumes)[1], \
    dl.get_sparse_edge_index(num_users, num_resumes)[2]

model = m.LightGCN(num_users, num_resumes)

model_name = input('Choose a existing NBA star:')
if model_name == 'kobe':
    print('Oh no, Mamba is out.')
    exit()

model.load_state_dict(torch.load('model/' + model_name + '.pt'))

print('Your are using model: ' + model_name)

df = pd.read_csv(resume_path)
resume_name = pd.Series(df.name.values, index=df.resume_id).to_dict()
resume_tags = pd.Series(df.tags.values, index=df.resume_id).to_dict()


def make_predictions(user_id, num_records):
    """

    :param user_id:
    :param num_records:
    :return:
    """

    user = user_mapping[user_id]
    emb_user = model.users_emb.weight[user]
    scores = model.items_emb.weight @ emb_user
    values, indices = torch.topk(scores, k=len(user_pos_items[user]) + num_records)

    resumes = [index.cpu().item() for index in indices if index in user_pos_items[user]][:num_records]
    resume_ids = [list(resume_mapping.keys())[list(resume_mapping.values()).index(resume)] for resume in resumes]
    name = [resume_name[id] for id in resume_ids]
    tags = [resume_tags[id] for id in resume_ids]
    print(f"User {user_id} rate them highly.")
    for i in range(num_records):
        print(f"name: {name[i]}, tags: {tags[i]} ")

    resumes = [index.cpu().item() for index in indices if index not in user_pos_items[user]][:num_records]
    resume_ids = [list(resume_mapping.keys())[list(resume_mapping.values()).index(resume)] for resume in resumes]
    name = [resume_name[id] for id in resume_ids]
    tags = [resume_tags[id] for id in resume_ids]

    print(f"User {user_id} might like to see these:")
    for i in range(num_records):
        print(f"name: {name[i]}, tags: {tags[i]} ")

model.eval()

sel = 'y'
while sel == 'y':
    USER = int(input('Which user:'))

    print(make_predictions(USER, 10))


