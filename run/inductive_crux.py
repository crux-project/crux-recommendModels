from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config, get_crux_config,get_material_classification_config,get_material_regression_config,get_kaggle_config,get_kag_update_config, get_hugging_update_config
import numpy as np
import time
import os

import scipy.sparse as sp

def main():


    train_start_time = time.time()
    #dataset_name ='kaggle'
    dataset_name = 'pka_update'
    #dataset_name = 'kag_update'
    #dataset_name = 'hugging_update'

    log_path = __file__[:-3]
    #embed = np.load('/Users/patrick/Downloads/igcn_cf/data/Crux/2/data_embed.npy',allow_pickle=True)
    #embed.item().get(0)
    init_run(log_path, 2021)
    #choose the #s split
    split =1

    #device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config = get_gowalla_config(device)
    #config = get_crux_config(device, split)
    #config = get_material_classification_config(device, split)
    if dataset_name =='kaggle':
        config = get_kaggle_config(device, split)
    elif dataset_name =='pka_update':
        config = get_crux_config(device, split)
    elif dataset_name =='kag_update':
        config = get_kag_update_config(device,split)
    elif dataset_name == 'hugging_update':
        config = get_hugging_update_config(device, split)

    dataset_config, model_config, trainer_config, metric_config = config[3]
    print(dataset_config['path'])
    #omit the time, change the directory to 'data/chosen_data/1
    #dataset_config['path'] = dataset_config['path'][:-4] + str(split)

    #method_name =model_config['name']
    ##automatic the processing of the raw dataset, only for method -- IGCN, we adopt the threshold processing to save computation and change the validation process metric

    writer = SummaryWriter(log_path)

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)

    print("training # of datasets is {0:d} and # of models is {1:d}".format(model.n_users, model.n_items))
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)

    writer.close()
    training_time = time.time() - train_start_time
    print('Training Time: {:.3f}s'.format(training_time))


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

    if split < 10:
        #dataset_config['path'] = dataset_config['path'][:-33]+ str('inductive')+'/table-kag-oracle-report-results'
        #dataset_config['path'] = dataset_config['path'][:-20] + str('inductive') + '/sampling_ratio=0.2'
        #dataset_config['path'] = dataset_config['path'][:-11] + str('inductive') + '/ratio_1.0'
        #dataset_config['path'] = dataset_config['path'][:-9] + str('inductive') + '/dummy25'
        #dataset_config['path'] = dataset_config['path'][:-31] + str('inductive') + '/sampling_ratio=1.0/70-probing'
        dataset_config['path'] = dataset_config['path'][:-1] + str('inductive')
    else:
        #dataset_config['path'] = dataset_config['path'][:-34] + str('inductive')+'/table-kag-oracle-report-results'
        #dataset_config['path'] = dataset_config['path'][:-21] + str('inductive') + '/sampling_ratio=0.2'
        #dataset_config['path'] = dataset_config['path'][:-12] + str('inductive') + '/ratio_1.0'
        #dataset_config['path'] = dataset_config['path'][:-10] + str('inductive') + '/dummy25'
        #dataset_config['path'] = dataset_config['path'][:-32] + str('inductive') + '/sampling_ratio=1.0/70-probing'
        dataset_config['path'] = dataset_config['path'][:-2] + str('inductive')

    start_time = time.time()
    new_dataset = get_dataset(dataset_config)

    model.config['dataset'] = new_dataset
    training_users = model.n_users
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    print("testing # of datasets is {0:d} and # of models is {1:d}".format(model.n_users, model.n_items))
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
        trainer.inductive_eval(training_users, new_dataset.n_items)

        '''
        ##prepare saving the prediction results as csv file
        ##load the data_map and model_map
        output_data_map_file_path = os.path.join('../../data/crux/output', 'data_map.npy')
        np.save(output_data_map_file_path, data_map)

        ##save model_map
        output_model_map_file_path = os.path.join('../../data/crux/output', 'model_map.npy')
        '''
        output_prediction_path = os.path.join('../../data/kaggle/output', 'orcal_pred_kaggle.csv')


        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))
    #results = trainer.inductive_eval(exist_train_users, exist_val_users, testing_users)


    #results= trainer.inductive_eval(exist_train_users,exist_val_users, testing_users)
    #print('Test result. {:s}'.format(results))
    elif model_config['name']== 'LightGCN':
        model.norm_adj = model.generate_graph(new_dataset)
        trainer = get_trainer(trainer_config, new_dataset, model)
        #model.load('/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/run/checkpoints/LightGCN_BPRTrainer_ProcessedDataset_50.518.pth')
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
        trainer.inductive_eval(training_users, new_dataset.n_items,start_time)
        #inference_time = time.time() - start_time
        #print('Inference Time: {:.3f}s'.format(inference_time))

    elif model_config['name']== 'IDCF_LGCN':
        model.norm_adj = model.generate_graph(new_dataset)
        model.feat_mat = model.generate_feat(new_dataset)
        trainer = get_trainer(trainer_config, new_dataset, model)
        print('Inductive results.')
        trainer.inductive_eval(training_users, new_dataset.n_items)
        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))

    elif model_config['name']=='IGCN_MLP':
        model.norm_adj = model.generate_graph(new_dataset)
        model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
        model.update_feat_mat()
        trainer = get_trainer(trainer_config, new_dataset, model)
        print('Inductive results.')
        #trainer.train(verbose=True, writer=writer)
        trainer.inductive_eval(training_users, new_dataset.n_items,start_time)
        #inference_time = time.time() - start_time
        #print('Inference Time: {:.3f}s'.format(inference_time))


if __name__ == '__main__':
    main()
