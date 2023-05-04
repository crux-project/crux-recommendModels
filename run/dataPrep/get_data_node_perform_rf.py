import os
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import random
import sys
import time
import json
import re
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


input_file_path = '/Users/patrick/Downloads/material_image_classification'
model_perf_folder = 'rf_700'
fnames = ['PC-RF-{:03}.csv'.format(i) for i in range(1,701)]
model_id =0
model_dict = dict()
data_id = 0
partition_files =['partition_{:}_meta_data.csv'.format(i) for i in range(1,101)]
if os.path.exists(os.path.join(input_file_path,'edge_material_rf.txt')):
    os.remove(os.path.join(input_file_path,'edge_material_rf.txt'))
else:
    print("the edge file does not exist")

with open(os.path.join(input_file_path,'edge_material_rf.txt'),'a') as f:
    if f.tell()==0:
        print('a new file or the file was empty')
    else:
        raise Exception('file existed, the initial detection is wrong')
line_count =0
with open(os.path.join(input_file_path,'edge_material_rf.txt'),'a') as f:
    for file in fnames:
        df_none = pd.read_csv(os.path.join(input_file_path,model_perf_folder,file))
        for partition_file in partition_files:
            #initialize a performanc_list
            performance_list =[]
            df_data = pd.read_csv(os.path.join(input_file_path,partition_file))
            key ='Labelfine'
            #do inner-join to prepare the performance table for the exact partition_file
            partition_perform = df_none[df_none.apply(key, axis=1).isin(df_data.apply(key,axis=1))]
            y_true = partition_perform['actual'].to_numpy()
            y_pred = partition_perform['prediction'].to_numpy()
            #place holder, add one dummy training time for now
            dummy_training_time = 0.00
            performance_list.append(dummy_training_time)
            #calculate R^2 score, the coefficient of determination
            #r2_score=r2_score(y_true,y_pred)
            #performance_list.append(r2_score)
            #calculate the mean absolute error
            mae = mean_absolute_error(y_true,y_pred)
            performance_list.append(mae)
            #calculate the mean absolute percentage error
            mape = mean_absolute_percentage_error(y_true, y_pred)
            performance_list.append(mape)
            #calculate the RMSE
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            performance_list.append(rmse)
            #save the performance to a txt file and appending write the performance to the txt file
            f.write('({},{})\t{}\n'.format(model_id, data_id, str(performance_list)))
            line_count += 1
            #update the data_id
            data_id+=1
        #reinitialize data_id
        data_id=0
        #update model_id
        model_id+=1
    print("finish the random forest performance building")



