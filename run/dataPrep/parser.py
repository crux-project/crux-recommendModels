import os
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import random
import sys
import time
import json
import re





def read_nodes(file_path):
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
    for line in lines:
        node_type = int(line.split('\t')[1])
        items = line.split('\t')[:1]
        if node_type ==1:
            #this is the data type
            #print(items[0])
            data_map[items[0]] = data_id
            datas.append(data_id)
            data_id +=1
        else:
            #this is the model type
            model_map[items[0]] = model_id
            models.append(model_id)
            model_id+=1
        datas = [int(item) for item in datas]
        models = [int(item) for item in models]
        if datas:
            n_datas = max(datas) + 1
        if models:
            n_models = max(models) + 1
    return n_datas, n_models, data_map, model_map, datas, models

def read_edges(file_path, data_map, model_map,metric):
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
            ranking_metric = metric_list.split(',')[1]
            ranking_metric = float(ranking_metric)
        edges_array[start_row_index]= [data_id, model_id, ranking_metric]
        #edges.append([data_id, model_id])
        start_row_index+=1
    #rank the edges_array according to the ranking_metric
    sorted_rows_idx= edges_array[:,2].argsort()[::-1]
    edges_array = edges_array[sorted_rows_idx]
    if metric == 'f1_score':
        threshold_f1_score = 0.5
        selected_rows_idx = edges_array[:,2]>threshold_f1_score
        edges_array = edges_array[selected_rows_idx]
    edges= np.array(edges_array[:,:2], dtype=int)
    return edges





input_file_path = os.path.join('../../data/crux_raw', 'node.txt')
n_datas, n_models, data_map, model_map, datas, models = read_nodes(input_file_path)
input_edge_path = os.path.join('../../data/crux_raw', 'edge.txt')
##user selects the F-1 score metric
user_selected_metric = 'f1_score'
sampling_ratio = 0.8
edges = read_edges(input_edge_path, data_map, model_map,user_selected_metric)

#for each data, we must make sure it has at least three models
#rank edges according to data id
edges= edges[edges[:,0].argsort()]
data_ids, indices = np.unique(edges[:,0], return_index=True)
train_edges = np.zeros(shape=(1,2))
val_edges = np.zeros(shape=(1,2))
test_edges = np.zeros(shape=(1,2))
for i in range(indices.shape[0]-1):
    model_pairs_with_same_data_id = edges[indices[i]:indices[i+1],:]
    #check the size of model_pairs_with_same_data_id >=3
    if model_pairs_with_same_data_id.shape[0]<3:
        continue
    else:
        train_split, test_split = train_test_split(model_pairs_with_same_data_id , test_size=0.2, random_state=2)
        train_split, val_split= train_test_split(train_split, test_size=0.25, random_state=2)
        train_edges = np.concatenate((train_edges, train_split), axis=0)
        val_edges = np.concatenate((val_edges,val_split),axis =0)
        test_edges = np.concatenate((test_edges, test_split), axis =0)

#remove the dummy header of train_edges, val_edges and test_edges
train_edges = np.delete(train_edges, 0, axis=0)
val_edges = np.delete(val_edges, 0, axis=0)
test_edges = np.delete(test_edges, 0, axis=0)

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

##choose the 80% as the inductive setting
train_ids, train_indices = np.unique(train_edges[:,0], return_index=True)
train_inductive_id_index = int(len(train_ids) *sampling_ratio)
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
print(train_data_index)
print(train_edge_list[-1])
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



##save train_edge_list , val_edge_list, test_edge_list
output_train_file_path = os.path.join('../../data/crux/time', 'inductive_train.txt')
f=open(output_train_file_path, 'w')
for items in train_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()
output_val_file_path = os.path.join('../../data/crux/time', 'inductive_val.txt')
f=open(output_val_file_path, 'w')
for items in val_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

output_test_file_path = os.path.join('../../data/crux/time', 'inductive_test.txt')
f=open(output_test_file_path, 'w')
for items in test_edge_list:
    for i in range(len(items)):
        if i!=len(items)-1:
            f.write(str(items[i])+ ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

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

##save train_edge_list , val_edge_list, test_edge_list
output_train_file_path = os.path.join('../../data/crux/time', 'initial_train.txt')
f=open(output_train_file_path, 'w')
for items in sampled_train_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()
output_val_file_path = os.path.join('../../data/crux/time', 'initial_val.txt')
f=open(output_val_file_path, 'w')
for items in sampled_val_edge_list:
    for i in range(len(items)):
        if i != len(items) - 1:
            f.write(str(items[i]) + ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()

output_test_file_path = os.path.join('../../data/crux/time', 'initial_test.txt')
f=open(output_test_file_path, 'w')
for items in sampled_test_edge_list:
    for i in range(len(items)):
        if i!=len(items)-1:
            f.write(str(items[i])+ ' ')
        else:
            f.write(str(items[i]))
    f.write('\n')
f.close()


print("Finish the parameterized data preparation")
