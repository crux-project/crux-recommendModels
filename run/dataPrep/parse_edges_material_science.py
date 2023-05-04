import os
import numpy as np
import pandas as pd
import re


#compute the model's performance for each partition testing set
input_file_path = '/Users/patrick/Downloads/material_image_classification'
model_perf_folder = 'lda_preds_meta_89'
fnames = ['lda_preds_meta89_model{:03}.csv'.format(i) for i in range(1,691)]
model_id =0
model_dict = dict()
for file in fnames:
    df_none = pd.read_csv(os.path.join(input_file_path,model_perf_folder,file),skiprows=1, header=None)
    #print(df_none)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    correct_num =0
    partition_images_col = df_none['0']
    for i in range(df_none.shape[0]):
        image_id =df_none.iloc[[i],[0]]
        pred = df_none.iloc[[i],[1]]
        ground_truth = df_none.iloc[[i],[2]]
        if pred =='low' and ground_truth =='low':
            true_positive +=1
            correct_num+=1
            assert df_none.iloc[[i],[0]]
        elif pred=='low' and
