def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 2, 'batch_size': 2048, 'dataloader_num_workers': 0,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.1}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.1, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg':  1.e-4,
                      'contrastive_reg':  1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_yelp_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    yelp_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.5, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg':  1.e-4,
                      'contrastive_reg':  1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512, 'topks': [20]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    yelp_config.append((dataset_config, model_config, trainer_config))
    return yelp_config


def get_amazon_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/time',
                      'device': device}
    amazon_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 256, 'n_layers': 3, 'device': device,
                    'dropout': 0., 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    amazon_config.append((dataset_config, model_config, trainer_config))
    return amazon_config

def get_crux_config(device,split):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/pka_update/'+str(split),
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    crux_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 1, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 2, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'IGCN_MLP', 'embedding_size': 256, 'n_layers': 3, 'device': device,
                    'dropout': 0.7, 'feature_ratio': 0.80}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.002, 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 6, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config, metric_config))



    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    crux_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    crux_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    crux_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/PKA-LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 3, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 100, 'topks': [20]}

    crux_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    crux_config.append((dataset_config, model_config, trainer_config, metric_config))


    return crux_config


def get_material_classification_config(device,split):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/material_classification/'+str(split),
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    material_classification_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_classification_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 30, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [30]}
    material_classification_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 30, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [20]}
    material_classification_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    material_classification_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    material_classification_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_classification_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_classification_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    material_classification_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 30, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 100, 'topks': [30]}

    material_classification_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    material_classification_config.append((dataset_config, model_config, trainer_config, metric_config))


    return material_classification_config


def get_material_regression_config(device,split):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/material_regression/'+str(split),
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    material_regression_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_regression_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 30, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [30]}
    material_regression_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 128, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 30, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [30]}
    material_regression_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    material_regression_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    material_regression_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_regression_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    material_regression_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    material_regression_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 30, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 100, 'topks': [30]}

    material_regression_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    material_regression_config.append((dataset_config, model_config, trainer_config, metric_config))


    return material_regression_config


def get_kaggle_config(device,split):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/kaggle/'+str(split),
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    kaggle_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 100, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'IGCN_MLP', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.9}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 100, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    kaggle_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 100, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 5, 'topks': [20]}

    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))


    return kaggle_config


def get_kag_update_config(device,split):

    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/kag_update/'+str(split)+'/table-kag-oracle-report-results',
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}
    '''
    dataset_config = {'name': 'ProcessedDataset',
                      'path': '/Users/patrick/Downloads/igcn_cf/data/kag_update/' + str(split) + '/sampling_ratio=1.0/70-probing',
                      'device': device,
                      'data_embed': None,
                      'model_embed': None}
    '''

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    kaggle_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 100, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config,metric_config))

    ###~~ Best table hyper-parameters~~~~~~##
    '''
    model_config = {'name': 'IGCN_MLP', 'embedding_size': 16, 'n_layers': 15, 'device': device,
                    'dropout': 0.9, 'feature_ratio': 0.95}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 4.e-3, 'l2_reg': 0.002, 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1, 'batch_size': 5, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))
    '''

    ####~~~Best optimization hyper-parameters~~~~##

    model_config = {'name': 'IGCN_MLP', 'embedding_size': 16, 'n_layers': 30, 'device': device,
                    'dropout': 0.7, 'feature_ratio': 0.95}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 4.e-3, 'l2_reg': 0.002, 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 5, 'batch_size': 5, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    ####~~~Best optimization hyper-parameters~~~~##
    '''
    model_config = {'name': 'IGCN_MLP', 'embedding_size': 16, 'n_layers': 30, 'device': device,
                    'dropout': 0.7, 'feature_ratio': 0.93}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 4.e-3, 'l2_reg': 0.002, 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 5, 'batch_size': 5, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))
    '''



    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    kaggle_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    kaggle_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 100, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 5, 'topks': [20]}

    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    kaggle_config.append((dataset_config, model_config, trainer_config, metric_config))


    return kaggle_config


def get_hugging_update_config(device,split):
    dataset_config = {'name': 'ProcessedDataset', 'path': '/Users/patrick/Downloads/igcn_cf/data/hugging_update/'+str(split)+ '/ratio_1.0',
                      'device': device,
                      'data_embed':None,
                      'model_embed':None}

    metric_config = {'name': 'metricByUser', 'metric':'F1_score'}
    hugging_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 100, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats':False}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1, 'batch_size': 50, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'IGCN_MLP', 'embedding_size': 32, 'n_layers': 2, 'device': device,
                    'dropout': 0.9, 'feature_ratio': 0.85}
    trainer_config = {'name': 'IGCN_MLPTrainer', 'optimizer': 'Adam', 'lr': 0.98e-3, 'l2_reg': 0.004, 'aux_reg': 0.001,
                      'device': device, 'n_epochs': 20, 'batch_size': 5, 'dataloader_num_workers': 0,
                      'test_batch_size': 5, 'topks': [10]}
    hugging_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [7]}
    hugging_config.append((dataset_config, model_config, trainer_config,metric_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    hugging_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    hugging_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': '/Users/patrick/Downloads/igcn_cf-master/crux-recommendModels/savedModel/LightGCN.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 100, 'batch_size': 50,
                      'dataloader_num_workers': 0, 'test_batch_size': 5, 'topks': [20]}

    hugging_config.append((dataset_config, model_config, trainer_config, metric_config))

    model_config = {'name': 'InductGNN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.0, 'use_feats': True, 'user_embeds_dim': 256, 'model_embeds_dim':256}
    trainer_config = {'name': 'InductGNNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 20, 'batch_size': 100, 'dataloader_num_workers': 0,
                      'test_batch_size': 100, 'topks': [3]}
    hugging_config.append((dataset_config, model_config, trainer_config, metric_config))


    return hugging_config