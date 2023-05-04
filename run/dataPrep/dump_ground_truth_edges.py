import os
from sklearn.model_selection import train_test_split
import numpy as np


def find_key(dict, search_item):
    for key, value in dict.iteritems():
        if value == search_item:
            return key


def read_edges(file_path, data_map_file_path, model_map_file_path,metric,pruning_flag=False):
    data_map = np.load(data_map_file_path,allow_pickle=True).item()
    model_map = np.load(model_map_file_path,allow_pickle=True).item()

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
        model_id = model_map[model_key]+72
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
            ranking_metric = metric_list.split(',')[1].replace(' ', '')
            ranking_metric = float(ranking_metric)
            ranking_metric = int(round(ranking_metric,2)*100)
            if ranking_metric ==0:
                ranking_metric=1
        edges_array[start_row_index]= [data_id, model_id, ranking_metric]
        #edges.append([data_id, model_id])
        start_row_index+=1
    #rank the edges_array according to the ranking_metric
    sorted_rows_idx= edges_array[:,2].argsort()[::-1]
    edges_array = edges_array[sorted_rows_idx]
    edges_array= edges_array.astype(int)
    if metric == 'f1_score':
        if pruning_flag:
            threshold_f1_score = 0.33
            selected_rows_idx = edges_array[:,2]>threshold_f1_score
            edges_array = edges_array[selected_rows_idx]
        else:
            #keep all the test edges
            edges_array = edges_array
    if metric == 'balanced_accuracy':
        if pruning_flag:
            threshold_f1_score = 0.25
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

    ##save edges to csv file
    return edges_array

input_file_path = os.path.join('../../data/kaggle/graph_split', 'new_edge.txt')
data_map_file_path = os.path.join('../../data/kaggle/output', 'data_map.npy')
model_map_file_path = os.path.join('../../data/kaggle/output', 'model_map.npy')
user_selected_metric = 'balanced_accuracy'
transferred_edge_file_path = os.path.join('../../data/kaggle/output', 'ground_truth_kaggle.csv')
pruning_flag =False
edges = read_edges(input_file_path, data_map_file_path, model_map_file_path,user_selected_metric,pruning_flag)
#sort by the last selected_metric first, then by the data_node id
edges = edges[np.lexsort((edges[:,-1], edges[:,0]))][::-1]
##save edges to csv file
np.savetxt(transferred_edge_file_path, edges, delimiter=',')