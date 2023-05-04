import os
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import random
import sys
import time
import json
import re
import pickle

def compt_ratio(total_budget, num_candidates, nodes):
    budget_per_node = total_budget//nodes
    ratio = budget_per_node *1.0 / num_candidates
    return 1-ratio


def threshold_pruning(edges,cut_off_threshold):
    selected_rows_idx = edges[:, 2] >= cut_off_threshold
    edges_array = edges[selected_rows_idx]
    top_k = 100000000
    edges_array = process_rank_data(edges_array, top_k)
    return edges_array




def process_rank_data(edges,top_k):
    edges = edges[edges[:, 0].argsort()]
    data_ids, indices = np.unique(edges[:, 0], return_index=True)
    test_edges = np.zeros(shape=(1, 3))
    for i in range(indices.shape[0]-1):
        model_pairs_with_same_data_id = edges[indices[i]:indices[i + 1], :]
        #if this node hasn't been mapped to more than 40 models
        if model_pairs_with_same_data_id.shape[0] < top_k:
            test_edges = np.concatenate((test_edges, model_pairs_with_same_data_id), axis=0)
            continue
        else:
            #print("sort the data-model pairs")
            selected_rows = model_pairs_with_same_data_id[model_pairs_with_same_data_id[:, 2].argsort()[::-1][:top_k]]
            test_edges = np.concatenate((test_edges, selected_rows), axis=0)
    #consider the corner case, the last inidces:
    last_indice = indices[-1]
    model_pairs_with_same_data_id = edges[last_indice:, :]
    if model_pairs_with_same_data_id.shape[0] < top_k:
        test_edges = np.concatenate((test_edges, model_pairs_with_same_data_id), axis=0)
    else:
        # print("sort the data-model pairs")
        selected_rows = model_pairs_with_same_data_id[model_pairs_with_same_data_id[:, 2].argsort()[::-1][:top_k]]
        test_edges = np.concatenate((test_edges, selected_rows), axis=0)
    test_edges = np.delete(test_edges, 0, axis=0)
    return test_edges



def append_nodes(test_node_path, data_map, model_map,n_datas, n_models, datas, models, data_embed, model_embed):
    ##get the current data_id and model_id
    data_map = data_map
    model_map = model_map
    n_datas = n_datas
    n_models = n_models
    datas = datas
    models = models
    data_embed = data_embed
    model_embed =model_embed
    assert(n_datas == list(data_map.values())[-1]+1)
    assert(n_models == list(model_map.values())[-1]+1)
    with open(test_node_path, 'r') as f:
        lines = f.read().strip().split('\n')
    data_id = n_datas
    model_id = n_models

    for line in lines:
        node_type = int(line.split('\t')[1])
        items = line.split('\t')[:1]
        embeds = line.split('\t')[2].replace('[','').replace(']','')
        embeds = np.fromstring(embeds, dtype=float, sep=',')
        if node_type ==1:
            #this is the data type
            #print(items[0])
            data_map[items[0]] = data_id
            datas.append(data_id)
            data_embed[data_id] =embeds
            data_id +=1

        else:
            #this is the model type
            model_map[items[0]] = model_id
            models.append(model_id)
            model_embed[model_id] = embeds
            model_id+=1
        datas = [int(item) for item in datas]
        models = [int(item) for item in models]
        if datas:
            n_datas = max(datas) + 1
        if models:
            n_models = max(models) + 1
    return n_datas, n_models, data_map, model_map, datas, models, data_embed, model_embed



def read_nodes(file_path):
    ##make sure that the data_id, model_id are increasing sequentially from 0
    datas = []
    models =[]
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    #build a dictionary starting from 0
    data_id =0
    model_id =0
    data_map = dict()
    model_map = dict()
    n_datas = 0
    n_models = 0
    data_embed =dict()
    model_embed = dict()
    for line in lines:
        node_type = int(line.split('\t')[1])
        items = line.split('\t')[:1]
        embeds = line.split('\t')[2].replace('[','').replace(']','')
        embeds = np.fromstring(embeds, dtype=float, sep=',')
        if node_type ==1:
            #this is the data type
            #print(items[0])
            data_map[items[0]] = data_id
            datas.append(data_id)
            data_embed[data_id] =embeds
            data_id +=1

        else:
            #this is the model type
            model_map[items[0]] = model_id
            models.append(model_id)
            model_embed[model_id] = embeds
            model_id+=1
        datas = [int(item) for item in datas]
        models = [int(item) for item in models]
        if datas:
            n_datas = max(datas) + 1
        if models:
            n_models = max(models) + 1
    return n_datas, n_models, data_map, model_map, datas, models, data_embed, model_embed

def read_edges(file_path, data_map, model_map,metric,test_flag=False):
    edges =[]
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    ##build an ndarray according to the number of lines
    edges_array = np.zeros(shape=(len(lines),3))
    start_row_index = 0
    for line in lines:
        prep_model_data_pair = line.split('\t')[0].replace('(','').replace(')','')
        [model_key, data_key]=prep_model_data_pair.split(",")
        metric_list = line.split('\t')[1]
        #print(model_key)
        model_id = model_map[model_key]
        data_id = data_map[data_key]
        if metric=='f1_score':
            ranking_metric = metric_list.split(',')[5].replace(' ','')
            ranking_metric = float(ranking_metric)
        if metric=='inference_time':
            ranking_metric = metric_list.split(',')[0].replace('[','').replace(']','')
            ranking_metric = float(ranking_metric)
        if metric=='cosine_similarity':
            ranking_metric = metric_list.split(',')[4].replace(' ','')
            ranking_metric = float(ranking_metric)
        if metric=='Jaccard_similarity':
            ranking_metric = metric_list.split(',')[5].replace(' ','').replace('[','').replace(']','')
            ranking_metric = float(ranking_metric)
        if metric=='balanced_accuracy':
            ranking_metric = metric_list.split(',')[2].replace(' ', '')
            ranking_metric = float(ranking_metric)
        edges_array[start_row_index]= [data_id, model_id, ranking_metric]
        #edges.append([data_id, model_id])
        start_row_index+=1
    #rank the edges_array according to the ranking_metric
    sorted_rows_idx= edges_array[:,2].argsort()[::-1]
    edges_array = edges_array[sorted_rows_idx]
    if metric == 'f1_score':
        if not test_flag:
            threshold_f1_score = 0.33
            selected_rows_idx = edges_array[:,2]>threshold_f1_score
            edges_array = edges_array[selected_rows_idx]
        else:
            #keep all the test edges
            edges_array = edges_array
    if metric == 'balanced_accuracy':
        if not test_flag:
            threshold_f1_score = 0.6
            selected_rows_idx = edges_array[:,2]>threshold_f1_score
            edges_array = edges_array[selected_rows_idx]
        else:
            #keep all the test edges
            edges_array = edges_array
    if metric == 'inference_time':
        print('only keep the 60% inference_time that is the fastest')
        threshold_inference_time = edges_array[int(edges_array.shape[0]*0.4),2]
        selected_rows_idx = edges_array[:,2]<=threshold_inference_time
        edges_array = edges_array[selected_rows_idx]
    if metric == 'cosine_similarity':
        threshold_f1_score = 0.7
        selected_rows_idx = edges_array[:, 2] > threshold_f1_score
        edges_array = edges_array[selected_rows_idx]
    if metric == 'Jaccard_similarity':
        threshold_f1_score = 0.5
        selected_rows_idx = edges_array[:, 2] > threshold_f1_score
        edges_array = edges_array[selected_rows_idx]

    edges= np.array(edges_array[:,:3], dtype=float)
    return edges



total_budget = 50
#total_budget =  200
random_state=5
input_file_path = os.path.join('../../data/kaggle/graph_split', 'node.txt')
train_n_datas, train_n_models, data_map, model_map, datas, models, data_embed, model_embed = read_nodes(input_file_path)
input_edge_path = os.path.join('../../data/kaggle/graph_split', 'edge.txt')
##user selects the F-1 score metric
#user_selected_metric = 'f1_score'
user_selected_metric = 'balanced_accuracy'
##user selects the running time score metric
#user_selected_metric = 'inference_time'
#user_selected_metric = 'Jaccard_similarity'
#user_selected_metric = 'cosine_similarity'

test_edge_path = os.path.join('../../data/kaggle/graph_split', 'new_edge.txt')
#process the new_edge.txt to a csv file:
#edges = change_to_csv_test_edges(test_edge_path,user_selected_metric)
test_node_path = os.path.join('../../data/kaggle/graph_split', 'new_node.txt')

n_datas, n_models, data_map, model_map, datas, models, data_embed, model_embed = append_nodes(test_node_path, data_map, model_map,train_n_datas, train_n_models, datas, models, data_embed, model_embed)

##set this test_cut_off_threshold to be used to compute the NDCG with the desired ground truth
test_cut_off_threshold =0.6
sampling_ratio = 0.7
##not pruning any edges
test_flag= False
#use 0.6 as threshold to filter unnecessary regression scores
edges = read_edges(input_edge_path, data_map, model_map,user_selected_metric, test_flag)
test_flag= True
test_raw_edges = read_edges(test_edge_path, data_map, model_map,user_selected_metric, test_flag)
#concatenate edges with test_edges
edges = np.concatenate((edges, test_raw_edges), axis=0)

#for each data, we must make sure it has at least three models
#rank edges according to data id
edges= edges[edges[:,0].argsort()]
data_ids, indices = np.unique(edges[:,0], return_index=True)
if data_ids.shape[0] < n_datas:
    print("Some data ids are missing, we need to regenerate the map to filter such data ids")
second_level_data_map = dict()
#mark the starting test dataset node id
test_start_node_index = int(test_raw_edges[0][0])
train_edges = np.zeros(shape=(1,3))
val_edges = np.zeros(shape=(1,3))
test_edges = np.zeros(shape=(1,3))
train_ground_truth = np.zeros(shape=(1,3))
valid_ground_truth = np.zeros(shape=(1,3))

##This value is for extended evaluation
top_k = 20
test_ground_truth = threshold_pruning(test_raw_edges,0.0)

#selected_ground_truth = process_rank_data(test_raw_edges, top_k)
#assert(selected_ground_truth.shape[0]==top_k*(n_datas-train_n_datas))
selected_ground_truth = threshold_pruning(test_raw_edges,test_cut_off_threshold)


#set the test_top_k be a real large number, make sure no test_ground_edges are not pruned at this stage
test_top_k =500000
test_ground_edges =process_rank_data(test_ground_truth, test_top_k)


#set the valid_top_k be a real large number, make sure no valid_ground_edges are not pruned at this stage
valid_top_k= 500000
added_train_test_phase_edges = 0

id_index = 0
##sort the training edges only
for i in range(indices.shape[0]-1):
    #print("the current data id is {:}".format(data_ids[i]))
    #only considers the training edges file,testing edges files is useless
    if data_ids[i]<train_n_datas and data_ids[i]==id_index:
        id_index+=1
        model_pairs_with_same_data_id = edges[indices[i]:indices[i+1],:]
        #check the size of model_pairs_with_same_data_id >=2
        if model_pairs_with_same_data_id.shape[0]<2:
            print("there may generate unexpected errors,  data id is {}".format(data_ids[i]))
            id_index -= 1
            continue
        else:
            #train_split, test_split = train_test_split(model_pairs_with_same_data_id , test_size=0.2, random_state=random_state)
            #train_split, val_split= train_test_split(train_split, test_size=0.25, random_state=random_state)
            train_split, val_split= train_test_split(model_pairs_with_same_data_id, test_size=1-sampling_ratio,
                                                       random_state=random_state)
            #remap the data node id to make sure that they are consecutive
            train_edges = np.concatenate((train_edges, train_split), axis=0)
            val_split = process_rank_data(val_split, valid_top_k)
            val_edges = np.concatenate((val_edges,val_split),axis =0)
            test_edges = np.concatenate((test_edges, val_split), axis =0)
            train_ground_truth_edges = process_rank_data(train_split, top_k)
            train_ground_truth = np.concatenate((train_ground_truth, train_ground_truth_edges), axis=0)
            valid_ground_truth_edges = process_rank_data(val_split, top_k)
            valid_ground_truth = np.concatenate((valid_ground_truth , valid_ground_truth_edges), axis =0)
    elif data_ids[i]<train_n_datas and data_ids[i]> id_index:
        print("The gap exists, we need to re-map the dataset id")
        key = int(data_ids[i])
        value = int(id_index)
        id_index +=1
        second_level_data_map[key]= value
        model_pairs_with_same_data_id = edges[indices[i]:indices[i + 1], :]
        if model_pairs_with_same_data_id.shape[0] < 2:
            print("there may generate unexpected errors,  data id is {}".format(data_ids[i]))
            id_index -= 1
            continue
        else:
            train_split, val_split = train_test_split(model_pairs_with_same_data_id, test_size=1 - sampling_ratio,
                                                      random_state=random_state)
            train_split[:,0]  = value
            val_split[:,0] = value
            train_edges = np.concatenate((train_edges, train_split), axis=0)
            val_split = process_rank_data(val_split, valid_top_k)
            val_edges = np.concatenate((val_edges, val_split), axis=0)
            test_edges = np.concatenate((test_edges, val_split), axis=0)
            train_ground_truth_edges = process_rank_data(train_split, top_k)
            train_ground_truth = np.concatenate((train_ground_truth, train_ground_truth_edges), axis=0)
            valid_ground_truth_edges = process_rank_data(val_split, top_k)
            valid_ground_truth = np.concatenate((valid_ground_truth, valid_ground_truth_edges), axis=0)
    #this branch should never be iterate in, just for robustness purpose
    elif data_ids[i]>=train_n_datas and data_ids[i]== id_index:
        print("The gap exists, we need to re-map the dataset id")
        key = int(data_ids[i])
        value = int(id_index)
        id_index += 1
        second_level_data_map[key] = value
        model_pairs_with_same_data_id = edges[indices[i]:indices[i + 1], :]
        # check the size of model_pairs_with_same_data_id >=2
        if model_pairs_with_same_data_id.shape[0] < 2:
            print("there may generate unexpected errors,  data id is {}".format(data_ids[i]))
            id_index -= 1
            continue
        else:
            ###The baseline is to use a random probing methodology
            num_candidates = model_pairs_with_same_data_id.shape[0]
            num_test_nodes = n_datas - train_n_datas
            test_size_ratio = compt_ratio(total_budget, num_candidates, num_test_nodes)
            train_split, test_split = train_test_split(model_pairs_with_same_data_id, test_size=test_size_ratio,
                                                      random_state=random_state)
            train_split[:, 0] = value
            added_train_test_phase_edges +=train_split.shape[0]
            index = np.where(test_ground_edges[:,0]==data_ids[i])
            ##The test split contains all possible node pair search space!!
            test_split = test_ground_edges[index]
            test_split[:, 0] = value
            train_edges = np.concatenate((train_edges, train_split), axis=0)
            val_edges = np.concatenate((val_edges, test_split), axis=0)
            test_edges = np.concatenate((test_edges, test_split), axis=0)

    elif data_ids[i] >= train_n_datas and data_ids[i] > id_index:
        print("The gap exists, we need to re-map the dataset id")
        key = int(data_ids[i])
        value = int(id_index)
        id_index += 1
        second_level_data_map[key] = value
        model_pairs_with_same_data_id = edges[indices[i]:indices[i + 1], :]
        # check the size of model_pairs_with_same_data_id >=2
        if model_pairs_with_same_data_id.shape[0] < 2:
            print("there may generate unexpected errors,  data id is {}".format(data_ids[i]))
            id_index -= 1
            continue
        else:
            ###The baseline is to use a random probing methodology
            num_candidates = model_pairs_with_same_data_id.shape[0]
            num_test_nodes = n_datas - train_n_datas
            test_size_ratio = compt_ratio(total_budget, num_candidates, num_test_nodes)
            train_split, test_split = train_test_split(model_pairs_with_same_data_id, test_size=test_size_ratio,
                                                       random_state=random_state)
            train_split[:, 0] = value
            added_train_test_phase_edges += train_split.shape[0]
            index = np.where(test_ground_edges[:, 0] == data_ids[i])
            ##The test split contains all possible node pair search space!!
            test_split = test_ground_edges[index]
            test_split[:, 0] = value
            train_edges = np.concatenate((train_edges, train_split), axis=0)
            val_edges = np.concatenate((val_edges, test_split), axis=0)
            test_edges = np.concatenate((test_edges, test_split), axis=0)



##consider the corner case, the last data id:
last_indice = indices[-1]
model_pairs_with_same_data_id = edges[last_indice:, :]
last_node_id = indices.shape[0] -1
if model_pairs_with_same_data_id.shape[0] >=2:
    key = int(data_ids[last_node_id])
    value = int(id_index)
    id_index += 1
    second_level_data_map[key] = value
    num_candidates = model_pairs_with_same_data_id.shape[0]
    num_test_nodes = n_datas - train_n_datas
    test_size_ratio = compt_ratio(total_budget, num_candidates, num_test_nodes)
    train_split, test_split = train_test_split(model_pairs_with_same_data_id, test_size=test_size_ratio, random_state=random_state)
    train_split[:, 0] = value
    added_train_test_phase_edges += train_split.shape[0]
    index = np.where(test_ground_edges[:, 0] == data_ids[-1])
    test_split = test_ground_edges[index]
    test_split[:, 0] = value
    train_edges = np.concatenate((train_edges, train_split), axis=0)
    val_edges = np.concatenate((val_edges, test_split), axis=0)
    test_edges = np.concatenate((test_edges, test_split), axis=0)
#this branch should never been interated
else:
    key = int(data_ids[last_node_id])
    value = int(id_index)
    id_index += 1
    second_level_data_map[key] = value

print("Total added new train edges in the test phase is {:}".format(added_train_test_phase_edges))

#remove the dummy header of train_edges, val_edges and test_edges
train_edges = np.delete(train_edges, 0, axis=0)
val_edges = np.delete(val_edges, 0, axis=0)
test_edges = np.delete(test_edges, 0, axis=0)
train_ground_truth = np.delete(train_ground_truth, 0, axis=0)
valid_ground_truth = np.delete(valid_ground_truth, 0, axis=0)

##save the selected ground truth
train_ground_truth_file_path = os.path.join('../../data/kaggle/output','train_ground_truth.csv')
train_ground_truth[:,:2] = train_ground_truth[:,:2].astype(int)
np.savetxt(train_ground_truth_file_path,train_ground_truth, fmt='%i,%i,%f',delimiter=',')
valid_ground_truth[:,:2] = valid_ground_truth[:,:2].astype(int)
valid_ground_truth_file_path = os.path.join('../../data/kaggle/output','valid_ground_truth.csv')
np.savetxt(valid_ground_truth_file_path,valid_ground_truth, fmt='%i,%i,%f',delimiter=',')

##save the second_level_data_map
output_second_data_map_file_path = os.path.join('../../data/kaggle/output','second_data_map.txt')
output_second_data_map_file = open(output_second_data_map_file_path,'wb')
pickle.dump(second_level_data_map,output_second_data_map_file)
output_second_data_map_file.close()


#confirm all the data ids are the same
train_data_ids= np.unique(train_edges[:,0])
val_data_ids = np.unique(val_edges[:,0])
test_data_ids= np.unique(test_edges[:,0])

assert(np.array_equal(train_data_ids, val_data_ids))
assert(np.array_equal(train_data_ids, test_data_ids))
assert(np.array_equal(val_data_ids, test_data_ids))




##sort the train_edges, val_edges, and test_edges
train_edges = train_edges[train_edges[:,0].argsort()]
val_edges = val_edges[val_edges[:,0].argsort()]
test_edges = test_edges[test_edges[:,0].argsort()]

##choose the training in the inductive setting according to our partition
train_ids, train_indices = np.unique(train_edges[:,0], return_index=True)
train_inductive_id_index = second_level_data_map[train_n_datas]
remaining_train_ids = train_ids[:train_inductive_id_index]
remaining_train_indices = train_indices[:train_inductive_id_index+1]

val_ids, val_indices = np.unique(val_edges[:,0], return_index=True)
remaining_val_ids = val_ids[:train_inductive_id_index]
remaining_val_indices = val_indices[:train_inductive_id_index+1]

test_ids, test_indices = np.unique(test_edges[:,0], return_index=True)
remaining_test_ids = test_ids[:train_inductive_id_index]
remaining_test_indices = test_indices[:train_inductive_id_index+1]

remaining_train_edges = train_edges[:remaining_train_indices[-1],]
remaining_val_edges =  val_edges[:remaining_val_indices[-1],]
remaining_test_edges = test_edges[:remaining_test_indices[-1],]

remaining_train_data_ids= np.unique(remaining_train_edges[:,0])
remaining_val_data_ids = np.unique(remaining_val_edges[:,0])
remaining_test_data_ids= np.unique(remaining_test_edges[:,0])

assert(np.array_equal(remaining_train_data_ids, remaining_val_data_ids))
assert(np.array_equal(remaining_train_data_ids, remaining_test_data_ids))
assert(np.array_equal(remaining_val_data_ids, remaining_test_data_ids))

##~~~~~~~~~~~~~~~~~~~~~~
##start writing the new data format to store the source, dest, performance pairs
##~~~~~~~~~~~~~~~~~~~~~~

train_edge_list=[]
train_edges_index = train_edges[0][0]
val_edge_list=[]
val_edges_index =val_edges[0][0]
test_edge_list =[]
test_edges_index = test_edges[0][0]

count =0
train_data_index =[]

for i in range(train_edges.shape[0]):
    if train_edges[i][0] == train_edges_index and count==0:
        temp_list= [int(train_edges_index)]
        temp_list.append(int(train_edges[i][1]))
        count+=1
        train_data_index.append(train_edges_index)
    elif train_edges[i][0] == train_edges_index and count!=0:
        temp_list.append(int(train_edges[i][1]))
        if i==train_edges.shape[0]-1:
            train_edge_list.append(temp_list)
    elif train_edges[i][0] > train_edges_index:
        train_edge_list.append(temp_list)
        train_edges_index = train_edges[i][0]
        count =0
        temp_list = [int(train_edges_index)]
        temp_list.append(int(train_edges[i][1]))
        count+=1
        train_data_index.append(train_edges_index)
        if i==train_edges.shape[0]-1:
            train_edge_list.append(temp_list)
#print(train_data_index)
#print(train_edge_list[-1])
##validation needs to make sure the train_data_index remains the same
count =0
for i in range(val_edges.shape[0]):
    if val_edges[i][0] == val_edges_index and (val_edges_index in train_data_index) and count==0:
        temp_list= [int(val_edges_index)]
        temp_list.append(int(val_edges[i][1]))
        count+=1
    elif val_edges[i][0] == val_edges_index and (val_edges_index in train_data_index) and count!=0:
        temp_list.append(int(val_edges[i][1]))
        if i==val_edges.shape[0]-1:
            val_edge_list.append(temp_list)
    elif val_edges[i][0] > val_edges_index and (val_edges[i][0] in train_data_index):
        val_edge_list.append(temp_list)
        val_edges_index = val_edges[i][0]
        count =0
        temp_list = [int(val_edges_index)]
        temp_list.append(int(val_edges[i][1]))
        count+=1
        if i == val_edges.shape[0] - 1:
            val_edge_list.append(temp_list)
    elif val_edges[i][0] > val_edges_index and (val_edges[i][0] not in train_data_index):
        continue

count =0
for i in range(test_edges.shape[0]):
    if test_edges[i][0] == test_edges_index and count==0:
        temp_list= [int(test_edges_index)]
        temp_list.append(int(test_edges[i][1]))
        count+=1
    elif test_edges[i][0] == test_edges_index and count!=0:
        temp_list.append(int(test_edges[i][1]))
        if i==test_edges.shape[0]-1:
            test_edge_list.append(temp_list)
    elif test_edges[i][0] > test_edges_index:
        test_edge_list.append(temp_list)
        test_edges_index = test_edges[i][0]
        count =0
        temp_list = [int(test_edges_index)]
        temp_list.append(int(test_edges[i][1]))
        count+=1
        if i == test_edges.shape[0] - 1:
            test_edge_list.append(temp_list)

###
#~~~~~~~~~~~~~~~~
### start saving the input as pairs
#~~~~~~~~~~~~~~~~
train_dict = dict()
val_dict = dict()
test_dict= dict()
train_edges_index = train_edges[0][0]
val_edges_index =val_edges[0][0]
test_edges_index = test_edges[0][0]

count =0
train_edge_num =0
train_data_index = []

for i in range(train_edges.shape[0]):
    if train_edges[i][0] == train_edges_index and count == 0:
        data_id = int(train_edges_index)
        key = data_id
        train_dict[key] = dict()
        train_dict[key]['models']=dict()
        train_dict[key]['performance'] = dict()
        temp_model_list = [int(train_edges[i][1])]
        temp_performance_list =[train_edges[i][2]]
        count+=1
        train_edge_num+=1
        train_data_index.append(data_id)
    elif train_edges[i][0] == train_edges_index and count != 0:
        temp_model_list.append(int(train_edges[i][1]))
        temp_performance_list.append(train_edges[i][2])
        count += 1
        train_edge_num += 1
        if i==train_edges.shape[0]-1:
            train_dict[key]['models'] = np.array(temp_model_list)
            train_dict[key]['performance'] = np.array(temp_performance_list)
    elif train_edges[i][0] > train_edges_index:
        train_dict[key]['models'] = np.array(temp_model_list)
        train_dict[key]['performance'] = np.array(temp_performance_list)
        train_edges_index = train_edges[i][0]
        count = 0
        data_id = int(train_edges_index)
        key = data_id
        train_dict[key] = dict()
        train_dict[key]['models'] = dict()
        train_dict[key]['performance'] = dict()
        temp_model_list = [int(train_edges[i][1])]
        temp_performance_list = [train_edges[i][2]]
        count += 1
        train_edge_num += 1
        train_data_index.append(data_id)
        if i==train_edges.shape[0]-1:
            train_dict[key]['models'] = np.array(temp_model_list)
            train_dict[key]['performance'] = np.array(temp_performance_list)

assert(train_edge_num==train_edges.shape[0])

count =0
val_edge_num =0
for i in range(val_edges.shape[0]):
    if val_edges[i][0] == val_edges_index and (val_edges_index in train_data_index) and count == 0:
        data_id = int(val_edges_index)
        key = data_id
        val_dict[key] = dict()
        val_dict[key]['models']=dict()
        val_dict[key]['performance'] = dict()
        temp_model_list = [int(val_edges[i][1])]
        temp_performance_list =[val_edges[i][2]]
        count+=1
        val_edge_num+=1
    elif val_edges[i][0] == val_edges_index and (val_edges_index in train_data_index) and count != 0:
        temp_model_list.append(int(val_edges[i][1]))
        temp_performance_list.append(val_edges[i][2])
        count += 1
        val_edge_num += 1
        if i==val_edges.shape[0]-1:
            val_dict[key]['models'] = np.array(temp_model_list)
            val_dict[key]['performance'] = np.array(temp_performance_list)
    elif val_edges[i][0] > val_edges_index and (val_edges[i][0] in train_data_index):
        val_dict[key]['models'] = np.array(temp_model_list)
        val_dict[key]['performance'] = np.array(temp_performance_list)
        val_edges_index = val_edges[i][0]
        count = 0
        data_id = int(val_edges_index)
        key = data_id
        val_dict[key] = dict()
        val_dict[key]['models'] = dict()
        val_dict[key]['performance'] = dict()
        temp_model_list = [int(val_edges[i][1])]
        temp_performance_list = [val_edges[i][2]]
        count += 1
        val_edge_num += 1
        if i==val_edges.shape[0]-1:
            val_dict[key]['models'] = np.array(temp_model_list)
            val_dict[key]['performance'] = np.array(temp_performance_list)
    elif val_edges[i][0] > val_edges_index and (val_edges[i][0] not in train_data_index):
        continue

assert(val_edge_num==val_edges.shape[0])

count =0
test_edge_num =0
for i in range(test_edges.shape[0]):
    if test_edges[i][0] == test_edges_index and (test_edges_index in train_data_index) and count == 0:
        data_id = int(test_edges_index)
        key = data_id
        test_dict[key] = dict()
        test_dict[key]['models']=dict()
        test_dict[key]['performance'] = dict()
        temp_model_list = [int(test_edges[i][1])]
        temp_performance_list =[test_edges[i][2]]
        count+=1
        test_edge_num+=1
    elif test_edges[i][0] == test_edges_index and (test_edges_index in train_data_index) and count != 0:
        temp_model_list.append(int(test_edges[i][1]))
        temp_performance_list.append(test_edges[i][2])
        count += 1
        test_edge_num += 1
        if i==test_edges.shape[0]-1:
            test_dict[key]['models'] = np.array(temp_model_list)
            test_dict[key]['performance'] = np.array(temp_performance_list)
    elif test_edges[i][0] > test_edges_index and (test_edges[i][0] in train_data_index):
        test_dict[key]['models'] = np.array(temp_model_list)
        test_dict[key]['performance'] = np.array(temp_performance_list)
        test_edges_index = test_edges[i][0]
        count = 0
        data_id = int(test_edges_index)
        key = data_id
        test_dict[key] = dict()
        test_dict[key]['models'] = dict()
        test_dict[key]['performance'] = dict()
        temp_model_list = [int(test_edges[i][1])]
        temp_performance_list = [test_edges[i][2]]
        count += 1
        test_edge_num += 1
        if i==test_edges.shape[0]-1:
            test_dict[key]['models'] = np.array(temp_model_list)
            test_dict[key]['performance'] = np.array(temp_performance_list)
    elif test_edges[i][0] > test_edges_index and (test_edges[i][0] not in train_data_index):
        continue

assert(test_edge_num==test_edges.shape[0])




'''
#check which val_edge_list has issue
edge_index_check =[]
for i in range(len(val_edge_list)):
    edge_index_check.append(val_edge_list[i][0])
print(edge_index_check)
for i in range(len(train_data_index)):
    if train_data_index[i] not in edge_index_check:
        print(train_data_index[i])
        #randomly select one edge from edges, use copy to avoid automatic id change
        val_edge_list.append(train_edge_list[i].copy())
#sort val_edge_list according to first value
val_edge_list=sorted(val_edge_list, key=lambda x:x[0])

#need to reindex train and validation users id to make sure that they are consecutive
index_map = dict()
start =0
for index in train_data_index:
    index_map[index] = start
    start+=1
for i in range(len(train_edge_list)):
    origin_value = train_edge_list[i][0]
    train_edge_list[i][0] =index_map[origin_value]
for i in range(len(val_edge_list)):
    origin_value = val_edge_list[i][0]
    if origin_value in index_map:
        val_edge_list[i][0] =index_map[origin_value]
    else:
        index_map[origin_value] = start
        val_edge_list[i][0] = index_map[origin_value]
        start+=1
for i in range(len(test_edge_list)):
    origin_value = test_edge_list[i][0]
    if origin_value in index_map:
        test_edge_list[i][0] =index_map[origin_value]
    else:
        index_map[origin_value] = start
        test_edge_list[i][0] = index_map[origin_value]
        start+=1
##sort test_edge_list according to first value
test_edge_list=sorted(test_edge_list, key=lambda x:x[0])

print("the largest id is %d"%start)
'''
###add logic to save node embeddings



##save train_edge_list , val_edge_list, test_edge_list
output_train_file_path = os.path.join('../../data/kaggle/output', 'inductive_train.txt')
f=open(output_train_file_path, 'w')
for items in train_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()
output_val_file_path = os.path.join('../../data/kaggle/output', 'inductive_val.txt')
f=open(output_val_file_path, 'w')
for items in val_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

output_test_file_path = os.path.join('../../data/kaggle/output', 'inductive_test.txt')
f=open(output_test_file_path, 'w')
for items in test_edge_list:
    for i in range(len(items)):
        if i!=len(items)-1:
            f.write(str(items[i])+ ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

###
#~~~~~~~~~~~~~~~~
### saving the new data format: train_dict, val_dict, test_dict
#~~~~~~~~~~~~~~~~
output_train_data_maps_file_path = os.path.join('../../data/kaggle/output','inductive_train_dict.txt')
output_train_data_map_file = open(output_train_data_maps_file_path,'wb')
pickle.dump(train_dict,output_train_data_map_file)
output_train_data_map_file.close()

output_val_data_maps_file_path = os.path.join('../../data/kaggle/output','inductive_val_dict.txt')
output_val_data_map_file = open(output_val_data_maps_file_path,'wb')
pickle.dump(val_dict,output_val_data_map_file)
output_val_data_map_file.close()

output_test_data_maps_file_path = os.path.join('../../data/kaggle/output','inductive_test_dict.txt')
output_test_data_map_file = open(output_test_data_maps_file_path,'wb')
pickle.dump(test_dict,output_test_data_map_file)
output_test_data_map_file.close()


#~~~~~~~~~~~~~~
#start saving the training part
#~~~~~~~~~~~~~~

sampled_train_edge_list=[]
train_edges_index = remaining_train_edges[0][0]
sampled_val_edge_list=[]
val_edges_index =remaining_val_edges[0][0]
sampled_test_edge_list =[]
test_edges_index = remaining_test_edges[0][0]

count =0
sampled_train_data_index =[]

for i in range(remaining_train_edges.shape[0]):
    if remaining_train_edges[i][0] == train_edges_index and count==0:
        temp_list= [int(train_edges_index)]
        temp_list.append(int(remaining_train_edges[i][1]))
        count+=1
        sampled_train_data_index.append(train_edges_index)
    elif remaining_train_edges[i][0] == train_edges_index and count!=0:
        temp_list.append(int(remaining_train_edges[i][1]))
        if i==remaining_train_edges.shape[0]-1:
            sampled_train_edge_list.append(temp_list)
    elif remaining_train_edges[i][0] > train_edges_index:
        sampled_train_edge_list.append(temp_list)
        train_edges_index = remaining_train_edges[i][0]
        count =0
        temp_list = [int(train_edges_index)]
        temp_list.append(int(remaining_train_edges[i][1]))
        count+=1
        sampled_train_data_index.append(train_edges_index)
        if i==remaining_train_edges.shape[0]-1:
            sampled_train_edge_list.append(temp_list)

print(sampled_train_data_index)
print(sampled_train_edge_list[-1])
##validation needs to make sure the train_data_index remains the same
count =0
for i in range(remaining_val_edges.shape[0]):
    if remaining_val_edges[i][0] == val_edges_index and (val_edges_index in sampled_train_data_index) and count==0:
        temp_list= [int(val_edges_index)]
        temp_list.append(int(remaining_val_edges[i][1]))
        count+=1
    elif remaining_val_edges[i][0] == val_edges_index and (val_edges_index in sampled_train_data_index) and count!=0:
        temp_list.append(int(remaining_val_edges[i][1]))
        if i==remaining_val_edges.shape[0]-1:
            sampled_val_edge_list.append(temp_list)
    elif remaining_val_edges[i][0] > val_edges_index and (remaining_val_edges[i][0] in sampled_train_data_index):
        sampled_val_edge_list.append(temp_list)
        val_edges_index = remaining_val_edges[i][0]
        count =0
        temp_list = [int(val_edges_index)]
        temp_list.append(int(remaining_val_edges[i][1]))
        count+=1
    elif remaining_val_edges[i][0] > val_edges_index and (remaining_val_edges[i][0] not in sampled_train_data_index):
        continue

count =0
for i in range(remaining_test_edges.shape[0]):
    if remaining_test_edges[i][0] == test_edges_index and count==0:
        temp_list= [int(test_edges_index)]
        temp_list.append(int(remaining_test_edges[i][1]))
        count+=1
    elif remaining_test_edges[i][0] == test_edges_index and count!=0:
        temp_list.append(int(remaining_test_edges[i][1]))
        if i==remaining_test_edges.shape[0]-1:
            sampled_test_edge_list.append(temp_list)
    elif remaining_test_edges[i][0] > test_edges_index:
        sampled_test_edge_list.append(temp_list)
        test_edges_index = remaining_test_edges[i][0]
        count =0
        temp_list = [int(test_edges_index)]
        temp_list.append(int(remaining_test_edges[i][1]))
        count+=1
        if i==remaining_test_edges.shape[0]-1:
            sampled_test_edge_list.append(temp_list)

###
#~~~~~~~~~~~~~~~~
### start saving the sampled input as pairs
#~~~~~~~~~~~~~~~~
sampled_train_dict = dict()
sampled_val_dict = dict()
sampled_test_dict= dict()

sampled_train_edges_index = remaining_train_edges[0][0]
sampled_val_edges_index =remaining_val_edges[0][0]
test_edges_index = remaining_test_edges[0][0]

count = 0
sampled_train_data_index = []
sampled_train_edge_num =0

for i in range(remaining_train_edges.shape[0]):
    if remaining_train_edges[i][0] == sampled_train_edges_index and count == 0:
        data_id = int(sampled_train_edges_index)
        key = data_id
        sampled_train_dict[key] = dict()
        sampled_train_dict[key]['models']=dict()
        sampled_train_dict[key]['performance'] = dict()
        temp_model_list = [int(remaining_train_edges[i][1])]
        temp_performance_list =[remaining_train_edges[i][2]]
        count+=1
        sampled_train_edge_num +=1
        sampled_train_data_index.append(data_id)
    elif remaining_train_edges[i][0] == sampled_train_edges_index and count != 0:
        temp_model_list.append(int(remaining_train_edges[i][1]))
        temp_performance_list.append(remaining_train_edges[i][2])
        count += 1
        sampled_train_edge_num += 1
        if i==remaining_train_edges.shape[0]-1:
            sampled_train_dict[key]['models'] = np.array(temp_model_list)
            sampled_train_dict[key]['performance'] = np.array(temp_performance_list)
    elif remaining_train_edges[i][0] > sampled_train_edges_index:
        sampled_train_dict[key]['models'] = np.array(temp_model_list)
        sampled_train_dict[key]['performance'] = np.array(temp_performance_list)
        sampled_train_edges_index = remaining_train_edges[i][0]
        count = 0
        data_id = int(sampled_train_edges_index)
        key = data_id
        sampled_train_dict[key] = dict()
        sampled_train_dict[key]['models'] = dict()
        sampled_train_dict[key]['performance'] = dict()
        temp_model_list = [int(remaining_train_edges[i][1])]
        temp_performance_list = [remaining_train_edges[i][2]]
        count += 1
        sampled_train_edge_num += 1
        sampled_train_data_index.append(data_id)
        if i==remaining_train_edges.shape[0]-1:
            sampled_train_dict[key]['models'] = np.array(temp_model_list)
            sampled_train_dict[key]['performance'] = np.array(temp_performance_list)

assert(sampled_train_edge_num==remaining_train_edges.shape[0])




##save train_edge_list , val_edge_list, test_edge_list
output_train_file_path = os.path.join('../../data/kaggle/output', 'initial_train.txt')
f=open(output_train_file_path, 'w')
for items in sampled_train_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()
output_val_file_path = os.path.join('../../data/kaggle/output', 'initial_val.txt')
f=open(output_val_file_path, 'w')
for items in sampled_val_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

output_test_file_path = os.path.join('../../data/kaggle/output', 'initial_test.txt')
f=open(output_test_file_path, 'w')
for items in sampled_test_edge_list:
    for i in range(len(items)):
        if i!=len(items)-1:
            f.write(str(items[i])+ ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()


for i in range(test_ground_truth.shape[0]):
    key =test_ground_truth[i,0]
    test_ground_truth[i,0]=second_level_data_map[key]
test_ground_truth[:,:2] = test_ground_truth[:,:2].astype(int)
test_ground_truth_file_path = os.path.join('../../data/kaggle/output','test_ground_truth.csv')
np.savetxt(test_ground_truth_file_path,test_ground_truth, fmt='%i,%i,%f',delimiter=',')

for i in range(selected_ground_truth.shape[0]):
    key = selected_ground_truth[i, 0]
    selected_ground_truth[i, 0] = second_level_data_map[key]
##save the selected ground truth
selected_ground_truth[:,:2] = selected_ground_truth[:,:2].astype(int)
output_ground_truth_file_path = os.path.join('../../data/kaggle/output','cutoff_ground_truth.csv')
np.savetxt(output_ground_truth_file_path,selected_ground_truth, fmt='%i,%i,%f',delimiter=',')

###save the data_embedding and model_embedding with numpy

output_data_embed_file_path = os.path.join('../../data/kaggle/output','data_embed.npy')
np.save(output_data_embed_file_path,data_embed)

##save data_map
output_data_map_file_path = os.path.join('../../data/kaggle/output','data_map.txt')
output_data_map_file = open(output_data_map_file_path,'wb')
pickle.dump(data_map,output_data_map_file)
output_data_map_file.close()

##save model_map
output_model_map_file_path = os.path.join('../../data/kaggle/output','model_map.txt')
output_model_map_file = open(output_model_map_file_path,'wb')
pickle.dump(model_map,output_model_map_file)
#np.save(output_model_map_file_path,model_map)
output_model_map_file.close()

output_model_embed_file_path = os.path.join('../../data/kaggle/output','model_embed.npy')
np.save(output_model_embed_file_path,model_embed)


print("Finish the parameterized data preparation")
