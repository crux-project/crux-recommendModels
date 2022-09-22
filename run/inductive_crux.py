from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config, get_crux_config
import numpy as np
import time

import scipy.sparse as sp

def main():
    '''
    test case


    n_users= 2
    n_items = 2
    user_map={0:0,1:1}
    item_map = {0:0,1:1}
    train_array = [[0,0],[0,1],[1,1]]
    user_embeds = np.array([[0,0,0],[1,1,1]])
    item_embeds = np.array([[2,2,2],[3,3,3]])
    user_embeds_dim = user_embeds[0].shape[0]
    model_embeds_dim =item_embeds[0].shape[0]

    user_dim, item_dim = len(user_map), len(item_map)
    data = []
    indices = []
    for user, item in train_array:
        if item in item_map:
            for i in range(user_embeds_dim):
                indices.append([user, (user_dim + item_map[item])*user_embeds_dim + i])
                user_r,user_c = [user,i]
                user_embed = user_embeds[user_r,user_c]
                item_r, item_c = [item_map[item],i]
                model_embed = item_embeds[item_r,item_c]
                data.append(user_embed + model_embed)
        if user in user_map:
            for i in range(user_embeds_dim):
                indices.append([n_users + item, user_map[user]* user_embeds_dim+i])
                user_r,user_c = [user_map[user],i]
                user_embed= user_embeds[user_r,user_c]
                item_r, item_c = [item,i]
                model_embed = item_embeds[item_r,item_c]
                data.append(user_embed + model_embed)
    print(indices)

    for user in range(n_users):
        for i in range(user_embeds_dim):
            indices.append([user, (user_dim + item_dim)*user_embeds_dim + i])
            data.append(1)
    for item in range(n_items):
        for i in range(user_embeds_dim):
            indices.append([n_users + item, (user_dim + item_dim + 1)*user_embeds_dim + i])
            data.append(1)
    print(data)
    print(indices)
    print(np.array(indices).T)
    feat = sp.coo_matrix((data, np.array(indices).T),
                         shape=(n_users + n_items, (user_dim + item_dim + 2)*user_embeds_dim), dtype=np.float32).tocsr()
    print(feat.toarray())
    row_sum = np.array(np.sum(feat, axis=1)).squeeze()
    print(row_sum)
    '''



    log_path = __file__[:-3]
    #embed = np.load('/Users/patrick/Downloads/igcn_cf/data/Crux/2/data_embed.npy',allow_pickle=True)
    #embed.item().get(0)
    init_run(log_path, 2021)
    #choose the #s split
    split =7

    #device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config = get_gowalla_config(device)
    config = get_crux_config(device)
    dataset_config, model_config, trainer_config, metric_config = config[8]
    #choose the configure file as IGCN
    print(dataset_config['path'])
    #omit the time, change the directory to 'data/chosen_data/1
    dataset_config['path'] = dataset_config['path'][:-4] + str(split)

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()
    #get the current training ids from trainer
    '''
    exist_train_users = trainer.dataset.train_ids
    exist_val_users = trainer.dataset.val_ids
    exist_testing_users = trainer.dataset.test_ids
    '''

    #transductive setting
    #results, _ = trainer.eval('test')
    #inductive setting


    #in the inductive setting, we need to consider a new set of training/validation/testing sets
    dataset_config['path'] = dataset_config['path'][:-1]+ str('time')
    new_dataset = get_dataset(dataset_config)



    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    start_time = time.time()
    ##determine which method you have:
    if model_config['name']== 'IGCN':
        model.norm_adj = model.generate_graph(new_dataset)
        '''
        with torch.no_grad():
            old_embedding = model.embedding.weight
            ## note that the original IGCN model with lightgcn the embedding is hardcoded with n_users+n_items+2
            model.embedding = torch.nn.Embedding(new_dataset.n_users + new_dataset.n_items+2, model.embedding_size,
                                                 device=device)
            model.embedding.weight[:, :] = old_embedding.mean(dim=0)[None, :].expand(model.embedding.weight.shape)
    
            model.embedding.weight[:dataset.n_users, :] = old_embedding[:dataset.n_users, :]
            model.embedding.weight[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
                old_embedding[dataset.n_users:dataset.n_users+dataset.n_items, :]
        '''
        '''
        user_embeds_dim = 1
        with torch.no_grad():
            old_embedding = model.embedding.weight
            ## note that the original IGCN model with lightgcn the embedding is hardcoded with n_users+n_items+2
            model.embedding = torch.nn.Embedding((new_dataset.n_users + new_dataset.n_items + 2)*user_embeds_dim, model.embedding_size,
                                                 device=device)
            model.embedding.weight[:, :] = old_embedding.mean(dim=0)[None, :].expand(model.embedding.weight.shape)

            model.embedding.weight[:dataset.n_users, :] = old_embedding[:dataset.n_users, :]
            model.embedding.weight[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
                old_embedding[dataset.n_users:dataset.n_users + dataset.n_items, :]
        '''
        if model_config['use_feats'] == True:
            model.feat_mat, _, _, model.row_sum = model.generate_feat_with_embed(new_dataset, is_updating=True)
        else:
            model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
        model.update_feat_mat()
        trainer = get_trainer(trainer_config, new_dataset, model)
        print('Inductive results.')
        trainer.inductive_eval(dataset.n_users, dataset.n_items)

        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))
    #results = trainer.inductive_eval(exist_train_users, exist_val_users, testing_users)


    #results= trainer.inductive_eval(exist_train_users,exist_val_users, testing_users)
    #print('Test result. {:s}'.format(results))
    elif model_config['name']== 'LightGCN':
        model.norm_adj = model.generate_graph(new_dataset)
        with torch.no_grad():
            old_embedding = model.embedding.weight
            model.embedding = torch.nn.Embedding(new_dataset.n_users + new_dataset.n_items, model.embedding_size,
                                                 device=device)
            model.embedding.weight[:, :] = old_embedding.mean(dim=0)[None, :].expand(model.embedding.weight.shape)
            model.embedding.weight[:dataset.n_users, :] = old_embedding[:dataset.n_users, :]
            model.embedding.weight[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
                old_embedding[dataset.n_users:, :]
        trainer = get_trainer(trainer_config, new_dataset, model)
        print('Inductive results.')
        trainer.inductive_eval(dataset.n_users, dataset.n_items)
        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))

    elif model_config['name']== 'IDCF_LGCN':
        model.norm_adj = model.generate_graph(new_dataset)
        model.feat_mat = model.generate_feat(new_dataset)
        trainer = get_trainer(trainer_config, new_dataset, model)
        print('Inductive results.')
        trainer.inductive_eval(dataset.n_users, dataset.n_items)
        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))


if __name__ == '__main__':
    main()
