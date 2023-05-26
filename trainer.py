import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
import time
import numpy as np
import os
from utils import AverageMeter, get_sparse_tensor, get_exclude_items, transform_edge_list,check_subset,transform_edge_to_dict,check_equalset
import torch.nn.functional as F
import scipy.sparse as sp
from dataset import AuxiliaryDataset
from ranx import Qrels, Run
from ranx import evaluate
from sklearn.metrics import mean_squared_error



def get_trainer(config, dataset, model):
    config = config.copy()
    config['dataset'] = dataset
    config['model'] = model
    trainer = getattr(sys.modules['trainer'], config['name'])
    trainer = trainer(config)
    return trainer


class BasicTrainer:
    def __init__(self, trainer_config):
        print(trainer_config)
        self.config = trainer_config
        self.name = trainer_config['name']
        self.dataset = trainer_config['dataset']
        self.model = trainer_config['model']
        self.topks = trainer_config['topks']
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.max_patience = trainer_config.get('max_patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.epoch = 0
        self.best_ndcg = -np.inf
        self.best_rmse = np.inf
        self.best_loss = np.inf
        self.save_path = None
        self.opt = None

        test_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=trainer_config['test_batch_size'])



    def initialize_optimizer(self):
        opt = getattr(sys.modules[__name__], self.config['optimizer'])
        self.opt = opt(self.model.parameters(), lr=self.config['lr'])

    def train_one_epoch(self):
        raise NotImplementedError

    def record(self, writer, stage, metrics):
        for metric in metrics:
            writer.add_scalar('{:s}_{:s}/{:s}_{:s}'
                .format(self.model.name, self.name, stage, metric)
                              , metrics[metric], self.epoch)

    def train(self, verbose=True, writer=None):
        if not self.model.trainable:
            results, metrics = self.eval('val')
            if verbose:
                print('Validation result. {:s}'.format(results))
            rmse = metrics['RMSE']
            return rmse

        if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
        patience = self.max_patience
        for self.epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            loss = self.train_one_epoch()
            _, metrics = self.eval('train')
            ##update the dataset to
            #_, metrics = self.constrained_eval('train')
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}s'
                      .format(self.epoch, self.n_epochs, loss, consumed_time))
            if writer:
                writer.add_scalar('{:s}_{:s}/train_loss'.format(self.model.name, self.name), loss, self.epoch)
                self.record(writer, 'train', metrics)

            if (self.epoch + 1) % self.val_interval != 0:
                continue

            start_time = time.time()
            results, metrics = self.eval('val')
            #results, metrics = self.constrained_eval('val')
            consumed_time = time.time() - start_time
            if verbose:
                print('Validation result. {:s}Time: {:.3f}s'.format(results, consumed_time))
            if writer:
                self.record(writer, 'validation', metrics)



            rmse = metrics['RMSE']
            if rmse < self.best_rmse:
                if self.save_path:
                    os.remove(self.save_path)
                self.save_path = os.path.join('checkpoints', '{:s}_{:s}_{:s}_{:.3f}.pth'
                                              .format(self.model.name, self.name, self.dataset.name, rmse))
                self.best_rmse = rmse
                self.model.save(self.save_path)
                patience = self.max_patience
                print('Best rmse, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping!')
                    break

            '''
            if loss < self.best_loss:
                if self.save_path:
                    os.remove(self.save_path)
                self.save_path = os.path.join('checkpoints', '{:s}_{:s}_{:s}_{:.3f}.pth'
                                              .format(self.model.name, self.name, self.dataset.name, loss))
                self.best_loss = loss
                self.model.save(self.save_path)
                patience = self.max_patience
                print('Best loss, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping!')
                    break
            '''

        print("The best model is:")
        print(self.save_path)
        self.model.load(self.save_path)
        return self.best_rmse

    def calculate_metrics(self, eval_data, rec_items):
        results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for user in range(rec_items.shape[0]):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[user, item_idx] in eval_data[user]:
                    hit_matrix[user, item_idx] = 1.
        eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

        for k in self.topks:
            hit_num = np.sum(hit_matrix[:, :k], axis=1)
            precisions = hit_num / k
            with np.errstate(invalid='ignore'):
                recalls = hit_num / eval_data_len

            max_hit_num = np.minimum(eval_data_len, k)
            max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
            for user, num in enumerate(max_hit_num):
                max_hit_matrix[user, :num] = 1.
            denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
            dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
            idcgs = np.sum(max_hit_matrix / denominator, axis=1)
            with np.errstate(invalid='ignore'):
                ndcgs = dcgs / idcgs

            user_masks = (max_hit_num > 0)
            results['Precision'][k] = precisions[user_masks].mean()
            results['Recall'][k] = recalls[user_masks].mean()
            results['NDCG'][k] = ndcgs[user_masks].mean()
        return results

    '''
    def eval(self, val_or_test, banned_items=None):
        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                #print(scores[0])
                #print(scores[0].shape)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        #print(rec_items)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics
    '''

    def eval(self,val_or_test):
        self.model.eval()
        label_lst, pred_lst = [], []
        rmse, mse= 0, 0
        count = 0
        metrics = {'MSE': {}, 'RMSE': {}}
        ##get the data loader
        eval_data = getattr(self.dataset, 'complete_'+val_or_test + '_data')
        batch_size = self.config['batch_size']
        eval_steps, eval_list = self.get_eval_data_list(eval_data, batch_size)
        eval_dataloader=self.get_batch_instances(eval_list, batch_size)
        with torch.no_grad():
            for batch_data in eval_dataloader:
                batch_data = np.array(batch_data)
                batch_users = torch.tensor(batch_data[:,0].astype(np.int64))
                batch_items = torch.tensor(batch_data[:,1].astype(np.int64))
                batch_labels = torch.FloatTensor(batch_data[:,2])
                batch_users_r, batch_items_r, _ = self.model.bpr_forward(batch_users, batch_items)
                batch_preds = torch.sum(batch_users_r * batch_items_r, dim=1)
                pred_lst.append(batch_preds.detach().numpy())
                label_lst.append(batch_labels.detach().numpy())
            pred_lst = np.concatenate(pred_lst)
            label_lst = np.concatenate(label_lst)
            mse = mean_squared_error(label_lst, pred_lst)
            rmse = np.sqrt(mse)

        results = 'MSE: {:.4f}RMSE: {:.4f}'.format(mse, rmse)
        metrics['MSE']=mse
        metrics['RMSE']=rmse
        return results, metrics

    def constrained_eval(self, val_or_test, banned_items=None):
        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        evaluation_constrained = []
        train_ground_truth = self.dataset.train_ground_truth
        train_ground_truth = transform_edge_list(train_ground_truth, self.topks[0])
        valid_ground_truth = self.dataset.val_ground_truth
        valid_ground_truth = transform_edge_list(valid_ground_truth, self.topks[0])
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                users = users.cpu().numpy().tolist()
                #print(scores[0])
                #print(scores[0].shape)
                n_items = scores.shape[1]
                exclude_user_indexes = []
                exclude_items = []
                if  val_or_test == 'train':
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        evaluation_items = train_ground_truth[user]
                        evaluation_constrained.append(evaluation_items)
                        assert check_subset(evaluation_items, items), "Invalid evaluation items"
                        e_items = get_exclude_items(n_items, items)
                        exclude_user_indexes.extend([user_idx] * len(e_items))
                        exclude_items.extend(e_items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                elif val_or_test != 'train':
                    #users = users.cpu().numpy().tolist()
                    #exclude_user_indexes = []
                    #exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.val_data[user]
                        evaluation_items = valid_ground_truth[user]
                        evaluation_constrained.append(evaluation_items)
                        e_items = get_exclude_items(n_items, items)
                        assert check_subset(evaluation_items, items), "Invalid evaluation items"
                        '''
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        '''
                        exclude_user_indexes.extend([user_idx] * len(e_items))
                        exclude_items.extend(e_items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                #scores[exclude_user_indexes, exclude_items] = -np.inf
                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        #print(rec_items)
        metrics = self.calculate_metrics(evaluation_constrained, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics


    def model_recommend(self, val_or_test, banned_items=None):
        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                #print(scores[0])
                #print(scores[0].shape)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    '''
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                    '''
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        print(rec_items)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics

    def inductive_model_recommend(self,n_old_users, n_old_items):
        results, _ = self.model_recommend('test')
        print('All users and all items result. {:s}'.format(results))

        '''
        test_data = self.dataset.test_data.copy()
        self.dataset.test_data = test_data.copy()
        for user in range(n_old_users):
            self.dataset.test_data[user] = []
        results, _ = self.eval('test')
        print('New users and all items result. {:s}'.format(results))
        '''

    def inductive_eval(self, n_old_users, n_new_items, start_time):
        test_data = self.dataset.test_data.copy()
        '''
        results, _ = self.eval('test')
        print('All users and all items result. {:s}'.format(results))
        '''

        '''
        for user in range(n_old_users, self.dataset.n_users):
            self.dataset.test_data[user] = []
        results, _ = self.eval('test')
        print('Old users and all items result. {:s}'.format(results))
        '''

        #self.dataset.test_data = test_data.copy()
        #train_data = self.dataset.train_data.copy()
        #self.dataset.train_data = train_data.copy()
        '''
        for user in range(n_old_users):
            #self.dataset.test_data[user] = []
            #remove the data list
            del self.dataset.test_data[0]
        '''
        del self.dataset.test_data[:n_old_users]
        #del self.dataset.train_data[:n_old_users]
        ##need to update the test_user_loader
        test_user = TensorDataset(torch.arange(n_old_users, self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size= self.config['test_batch_size'])
        '''
        results, _ = self.induct_eval('val',self.dataset.test_data)
        print('metric 1, New users and all items result. {:s}'.format(results))
        '''

        ##save the prediction results
        output_prediction_path = os.path.join('../data/prediction/output', 'oracle_pred.csv')
        #prediction_with_ranking = self.save_inductive_recommendations(self.dataset.test_data, output_prediction_path)
        results= self.save_inductive_with_rates_recommend(n_new_items, self.dataset.test_data, output_prediction_path,start_time)
        print('metric 2, New users and all items result. {:.3f}'.format(results))


        '''
        self.dataset.test_data = test_data.copy()
        for user in range(self.dataset.n_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items < n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items, self.dataset.n_items))
        print('All users and old items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(self.dataset.n_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items >= n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items))
        print('All users and new items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(n_old_users, self.dataset.n_users):
            self.dataset.test_data[user] = []
        for user in range(n_old_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items < n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items, self.dataset.n_items))
        print('Old users and old items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        '''

    def induct_eval(self, val_or_test, eval_data,banned_items=None):
        self.model.eval()
        #eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                #print(scores[0])
                #print(scores[0].shape)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        #print(rec_items)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics

    def save_inductive_with_rates_recommend(self,n_items,eval_data,output_prediction_path,start_time, banned_items=None):
        print("start saving predictions")
        self.model.eval()
        rec_items = []
        saved_preds =[]
        evaluation_constrained = []
        test_ground_matrix = self.dataset.test_cutoff_ground_truth
        #test_truth =transform_edge_list(test_ground_truth, self.topks[0])
        test_ground_truth =transform_edge_to_dict(test_ground_matrix)

        run_dict = {}
        pred_prob_matrix = np.zeros(shape=(1,3))

        qrels_dict = {}
        prediction_count =0

        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                ##exclude the items ids that not have rated values for user id
                users = users.cpu().numpy().tolist()
                exclude_user_indexes = []
                exclude_items = []

                for user_idx, user in enumerate(users):
                    ##leverage the val_data ==test_data in the split setting
                    items = self.dataset.val_data[user]
                    ##the train_data may cover edges we don't have ground truth
                    #items =items+ self.dataset.train_data[user]
                    #the test ground truth may not
                    if user not in test_ground_truth.keys():
                        continue
                    evaluation_items = test_ground_truth[user]
                    evaluation_constrained.append(evaluation_items)
                    #The trick here is val_data is the same as test_data for these new cold-start data nodes
                    #items = items + self.dataset.val_data[user]
                    #print(check_subset(evaluation_items, items))
                    ##check all evaluation_items should belong to items
                    assert check_subset(evaluation_items, items), "Invalid evaluation items"
                    #calculate exclude items for user id
                    e_items = get_exclude_items(n_items,items)
                    exclude_user_indexes.extend([user_idx] * len(e_items))
                    exclude_items.extend(e_items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                    probs = scores[user_idx].cpu().numpy()
                    items=np.where(np.isinf(probs)==False)
                    prediction_count+=items[0].shape[0]
                    probs = probs[items]
                    for item_idx, item in enumerate(items[0]):
                        if str(user) not in run_dict:
                            run_dict[str(user)]={}
                        run_dict[str(user)][str(item)] = probs[item_idx]
                        pred_prob_matrix =np.concatenate((pred_prob_matrix, np.array([user,item,probs[item_idx]]).reshape(1,-1)), axis=0)
                '''
                #_, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())
                users_id =np.array(users).reshape(-1,1)
                saved_pred=np.concatenate((users_id, items.cpu().numpy()),axis=1)
                saved_preds.append(saved_pred)
                '''
            pred_prob_matrix = np.delete(pred_prob_matrix, 0, axis=0)
            pred_prob_matrix[:,:2] = pred_prob_matrix[:,:2].astype(int)

        print(prediction_count)
        ##get the inference_time
        inference_time = time.time() - start_time
        print('Inference Time: {:.3f}s'.format(inference_time))


        ##save the predicted results after mapping
        np.savetxt(output_prediction_path, pred_prob_matrix, fmt='%i,%i,%f', delimiter=',')

        # sort the test_ground_matrix
        idex=np.lexsort([-1*test_ground_matrix[:,2],test_ground_matrix[:,0]])
        sorted_test_ground_matrix = test_ground_matrix[idex,:]

        ##
        for i in range(sorted_test_ground_matrix .shape[0]):
            data_id = sorted_test_ground_matrix [i][0]
            if str(int(data_id)) not in qrels_dict:
                qrels_dict[str(int(data_id))] ={}
            pred = int(round(sorted_test_ground_matrix [i][2]*100))
            model_id = sorted_test_ground_matrix[i][1]
            qrels_dict[str(int(data_id))][str(int(model_id))] = pred


        ##compute the
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)


        # Compute score for a single metric
        if self.dataset.config['path'].find('hugging_update')!=-1:
            results = evaluate(qrels, run, "ndcg@10")
        elif self.dataset.config['path'].find('kag_update')!=-1:
            results = evaluate(qrels, run, "ndcg@10")
        elif self.dataset.config['path'].find('pka_update')!=-1:
            results = evaluate(qrels, run, "ndcg@10")
        print('ndcg@10')
        print(results)






        '''
        ##eval_data add ids by 72
        evaluation_updates =[]
        for list in eval_data:
            list_values =[]
            for ele in list:
                list_values.append(ele+72)
            evaluation_updates.append(list_values)

        rec_items_updates= np.copy(rec_items)
        rec_items_updates =np.add(rec_items_updates, 72)

        saved_preds_update = np.concatenate((saved_preds[:,0].reshape(-1,1), rec_items_updates),axis=1)

        metrics = self.calculate_metrics(evaluation_constrained, rec_items)
        #metrics = self.calculate_metrics(evaluation_updates, rec_items_updates)


        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)

        np.savetxt(output_prediction_path, saved_preds_update, delimiter=',')
        '''


        return results

    def save_inductive_recommendations(self, eval_data, output_prediction_path, banned_items=None):
        print("start saving predictions")
        self.model.eval()
        rec_items = []
        saved_preds = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())
                users_id = users.cpu().numpy().reshape(-1, 1)
                saved_pred = np.concatenate((users_id, items.cpu().numpy()), axis=1)
                saved_preds.append(saved_pred)

        rec_items = np.concatenate(rec_items, axis=0)
        saved_preds = np.concatenate(saved_preds, axis=0)
        # print(rec_items)

        ##eval_data add ids by 72
        evaluation_updates = []
        for list in eval_data:
            list_values = []
            for ele in list:
                list_values.append(ele + 72)
            evaluation_updates.append(list_values)

        rec_items_updates = np.copy(rec_items)
        rec_items_updates = np.add(rec_items_updates, 72)

        saved_preds_update = np.concatenate((saved_preds[:, 0].reshape(-1, 1), rec_items_updates), axis=1)

        metrics = self.calculate_metrics(evaluation_updates, rec_items_updates)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)

        np.savetxt(output_prediction_path, saved_preds_update, delimiter=',')

        return results, metrics

    def get_eval_data_list(self, eval_data, batch_size):
        eval_data_list =[]
        for key in sorted(eval_data.keys()):
            user = int(key)
            for i in range(eval_data[key]['models'].size):
                item = int (eval_data[key]['models'][i])
                label = eval_data[key]['performance'][i]
                eval_data_list.append([user, item, label])
        print(len(eval_data_list))
        num_batches_per_epoch = int((len(eval_data_list) - 1) / batch_size) + 1
        return num_batches_per_epoch, eval_data_list

    def get_batch_instances(self, instances, batch_size):
        batch = []
        for instance in instances:
            batch.append(instance)
            if len(batch) == batch_size:
                yield batch
                batch=[]
        if len(batch)>0:
            yield batch










class BPRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BPRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class IDCFTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IDCFTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.contrastive_reg = trainer_config['contrastive_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq, contrastive_loss = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.contrastive_reg * contrastive_loss.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class IGCNTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IGCNTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.aux_dataloader = DataLoader(AuxiliaryDataset(self.dataset, self.model.user_map, self.model.item_map),
                                         batch_size=trainer_config['batch_size'],
                                         num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.aux_reg = trainer_config['aux_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data, a_batch_data in zip(self.dataloader, self.aux_dataloader):
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            inputs = a_batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r = self.model.embedding(users)
            pos_items_r = self.model.embedding(pos_items + len(self.model.user_map))
            neg_items_r = self.model.embedding(neg_items + len(self.model.user_map))
            pos_scores = torch.sum(users_r * pos_items_r * self.model.w[None, :], dim=1)
            neg_scores = torch.sum(users_r * neg_items_r * self.model.w[None, :], dim=1)
            aux_loss = F.softplus(neg_scores - pos_scores).mean()

            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.aux_reg * aux_loss
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        self.model.feat_mat_anneal()
        return losses.avg

class IGCN_MLPTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IGCN_MLPTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        '''
        self.aux_dataloader = DataLoader(AuxiliaryDataset(self.dataset, self.model.user_map, self.model.item_map),
                                         batch_size=trainer_config['batch_size'],
                                         num_workers=trainer_config['dataloader_num_workers'])
        '''
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        #self.aux_reg = trainer_config['aux_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        loss_function = torch.nn.MSELoss(size_average=False)
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.float32)
            '''
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            '''
            users, items, labels = inputs[:, 0].to(dtype=torch.int64), inputs[:, 1].to(dtype=torch.int64), inputs[:, 2]
            users_r, items_r, l2_norm_sq = self.model.bpr_forward(users, items)
            preds = torch.sum(users_r * items_r, dim=1)
            pred_loss = loss_function(preds, labels)



            # pos_items_r = self.model.embedding(pos_items + len(self.model.user_map))
            # neg_items_r = self.model.embedding(neg_items + len(self.model.user_map))
            # pos_scores = torch.sum(users_r * pos_items_r * self.model.w[None, :], dim=1)
            # neg_scores = torch.sum(users_r * neg_items_r * self.model.w[None, :], dim=1)
            # aux_loss = F.softplus(neg_scores - pos_scores).mean()

            ##The l2_norm_sq now is defined on the final representations, which may not be correct

            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = pred_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])

            #since the performance metric is not dynamically updated
            '''
            for batch_data, a_batch_data in zip(self.dataloader, self.aux_dataloader):
                inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.float32)
                
                #users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
                #users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
                #pos_scores = torch.sum(users_r * pos_items_r, dim=1)
                #neg_scores = torch.sum(users_r * neg_items_r, dim=1)
                #bpr_loss = F.softplus(neg_scores - pos_scores).mean()
         
                users, items, labels = inputs[:, 0].to(dtype=torch.int64), inputs[:, 1].to(dtype=torch.int64), inputs[:, 2]
                users_r, items_r, l2_norm_sq = self.model.bpr_forward(users, items)
                preds = torch.sum(users_r * items_r, dim=1)
                pred_loss = loss_function(preds, labels)
    
    
                inputs = a_batch_data[:, 0, :].to(device=self.device, dtype=torch.float32)
                #users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
                users, items, labels = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
                users_r = self.model.embedding(users)
                items_r = self.model.embedding(items+ len(self.model.user_map))
                preds = torch.sum(users_r * items_r * self.model.w[None, :], dim=1)
                aux_loss = loss_function(preds, labels)
    
                #pos_items_r = self.model.embedding(pos_items + len(self.model.user_map))
                #neg_items_r = self.model.embedding(neg_items + len(self.model.user_map))
                #pos_scores = torch.sum(users_r * pos_items_r * self.model.w[None, :], dim=1)
                #neg_scores = torch.sum(users_r * neg_items_r * self.model.w[None, :], dim=1)
                #aux_loss = F.softplus(neg_scores - pos_scores).mean()
    
                ##The l2_norm_sq now is defined on the final representations, which may not be correct
    
                reg_loss = self.l2_reg * l2_norm_sq.mean() + self.aux_reg * aux_loss
                loss = pred_loss + reg_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.update(loss.item(), inputs.shape[0])
            '''
        self.model.feat_mat_anneal()
        return losses.avg



class BCETrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BCETrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.mf_pretrain_epochs = trainer_config['mf_pretrain_epochs']
        self.mlp_pretrain_epochs = trainer_config['mlp_pretrain_epochs']

    def train_one_epoch(self):
        if self.epoch == self.mf_pretrain_epochs:
            self.model.arch = 'mlp'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
        if self.epoch == self.mf_pretrain_epochs + self.mlp_pretrain_epochs:
            self.model.arch = 'neumf'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
            self.model.init_mlp_layers()
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data.to(device=self.device, dtype=torch.int64)
            users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            logits, l2_norm_sq_p = self.model.bce_forward(users, pos_items)
            bce_loss_p = F.softplus(-logits)

            inputs = inputs.reshape(-1, 3)
            users, neg_items = inputs[:, 0], inputs[:, 2]
            logits, l2_norm_sq_n = self.model.bce_forward(users, neg_items)
            bce_loss_n = F.softplus(logits)

            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            l2_norm_sq = torch.cat([l2_norm_sq_p, l2_norm_sq_n], dim=0)
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bce_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg


class MLTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(MLTrainer, self).__init__(trainer_config)

        train_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.train_user_loader = DataLoader(train_user, batch_size=trainer_config['batch_size'], shuffle=True)
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.dataset.n_users, self.dataset.n_items), dtype=np.float32).tocsr()
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.kl_reg = trainer_config['kl_reg']

    def train_one_epoch(self):
        kl_reg = min(self.kl_reg, 1. * self.epoch / self.n_epochs)

        losses = AverageMeter()
        for users in self.train_user_loader:
            users = users[0]

            scores, kl, l2_norm_sq = self.model.ml_forward(users)
            scores = F.log_softmax(scores, dim=1)
            users = users.cpu().numpy()
            profiles = self.data_mat[users, :]
            profiles = get_sparse_tensor(profiles, self.device).to_dense()
            ml_loss = -torch.sum(profiles * scores, dim=1).mean()

            reg_loss = kl_reg * kl.mean() + self.l2_reg * l2_norm_sq.mean()
            loss = ml_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), users.shape[0])
        return losses.avg

