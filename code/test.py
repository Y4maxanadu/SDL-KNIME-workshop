import dataloader as dl

um,rm = dl.get_user_mapping(), dl.get_resume_mapping()
dl.get_dynamic_edge_index()
n1,n2 = dl.get_num_users(um), dl.get_num_resumes(rm)


dl.get_sparse_edge_index(n1, n2)