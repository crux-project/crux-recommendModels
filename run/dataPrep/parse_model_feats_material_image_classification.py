import os
import numpy as np
import pandas as pd
import re


###generate model features from results_4pc_lda_test89.csv

###generate edges based on data id and model id pair in the standard edge.txt

input_file_path = '/Users/patrick/Downloads/material_image_classification'
data_file = 'results_4pc_lda_test89.csv'
model_result_file ='image_classi_model.txt'
edges_file = 'edge.txt'
#read csv file line by line
with open(os.path.join(input_file_path,data_file), 'r') as f:
    lines = f.read().strip().split('\n')
#model_feats_vec = np.zeros(shape=(1,24))
material_models_ids =[]
material_model_dict = dict()
value = 0
node_type ='0'
#skip first line since it is header
line_count =0
data_nodes = list(range(0,100))
#In hugh's partition we always have 54 testing data points, whereas in partitaion we generated we have 50 data points
for i in range(1,len(lines)):
    model_feats_idx = lines[i].split('(')[1].split(')')[0]
    model_feats_idx = [int(j) for j in model_feats_idx.split(',')]
    model_feats = np.zeros(shape=(24))
    for idx in model_feats_idx:
        model_feats[idx-1]=1
    #print(model_feats)
    #model_feats_vec = np.concatenate((model_feats_vec , model_feats), axis=0)
    model_id = int(lines[i].split(',')[1])
    material_model_dict[model_id] = value
    test_time = float(lines[i].split(',')[-1])*(50.0/54.0)
    for data_node in data_nodes:
        with open(os.path.join(input_file_path, edges_file), 'a') as edge_file:
            edge_file.write('({},{})\t{}\n'.format(value, data_node, '['+str(test_time)+']'))
            line_count += 1
    '''
    #save model dictionary and model features
    with open(os.path.join(input_file_path, model_result_file), 'a') as text_file:
        text_file.write('{}\t{}\t{}\n'.format(value, node_type, model_feats))
        line_count+=1
    '''
    value += 1

edge_file.close()
#text_file.close()
#check the number of models are equal
assert (len(lines)-1)*100 ==line_count,  f"line number should match, got: {line_count}"





