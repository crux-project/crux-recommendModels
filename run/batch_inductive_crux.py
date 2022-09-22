from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config, get_crux_config
import numpy as np
import time

def main():
    log_path = __file__[:-3]
    #embed = np.load('/Users/patrick/Downloads/igcn_cf/data/Crux/2/data_embed.npy',allow_pickle=True)
    #embed.item().get(0)
    init_run(log_path, 2021)
    #choose the #s split
    split =2

    #device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config = get_gowalla_config(device)
    config = get_crux_config(device)
    dataset_config, model_config, trainer_config, metric_config = config[2]
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
    start_time = time.time()
    for i in range(1,5):
        dataset_config['path'] = dataset_config['path'][:-1]+ str(i)
        new_dataset = get_dataset(dataset_config)

        model.config['dataset'] = new_dataset
        model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
        model.norm_adj = model.generate_graph(new_dataset)
        model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset)
        model.update_feat_mat()
        with torch.no_grad():
            old_embedding = model.embedding.weight
            ## note that the original IGCN model with lightgcn the embedding is hardcoded with n_users+n_items+2
            model.embedding = torch.nn.Embedding(new_dataset.n_users + new_dataset.n_items+2, model.embedding_size,
                                                 device=device)
            model.embedding.weight[:, :] = old_embedding.mean(dim=0)[None, :].expand(model.embedding.weight.shape)

            model.embedding.weight[:dataset.n_users, :] = old_embedding[:dataset.n_users, :]
            model.embedding.weight[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
                old_embedding[dataset.n_users:dataset.n_users+dataset.n_items, :]

        trainer = get_trainer(trainer_config, new_dataset, model)

        print('Inductive results.')
        trainer.inductive_eval(dataset.n_users, dataset.n_items)
    inference_time = time.time() - start_time
    print('Inference Time: {:.3f}s'.format(inference_time))
    #results = trainer.inductive_eval(exist_train_users, exist_val_users, testing_users)


    #results= trainer.inductive_eval(exist_train_users,exist_val_users, testing_users)
    #print('Test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
