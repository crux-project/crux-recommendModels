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

np.random.seed(0)
input_file_path = '/Users/patrick/Downloads/material_image_classification'
data_file = 'data.csv'
material_data=pd.read_csv(os.path.join(input_file_path,data_file))
material_data_id = material_data.to_numpy()[:,0]
material_data_id=material_data_id.astype(int)

np.random.choice(material_data_id, 50)
combs = set()
N = 100
while len(combs)< N:
    sample_array =np.sort(np.random.choice(material_data_id, 50,replace=False)).flatten()
    combs.add(tuple(sample_array.tolist()))
#change set to numpy array
samples = np.array(list(combs))

partition_ids=np.array(range(1,samples.shape[0]+1))
partition_ids=partition_ids.reshape((-1,1))
partition_table=np.concatenate((partition_ids, samples),axis=1)

columns =[str(i+1) for i in range(partition_table.shape[1]-1)]
columns = ['partition_id']+ columns
#build a dictionary for each samples
df=pd.DataFrame(partition_table,
                index=[i for i in range(partition_table.shape[0])],
                columns=columns)
#print(df)
#save dataframe to a csv file
partition_file = 'partition.csv'
df.to_csv(os.path.join(input_file_path,partition_file))


##for each partition, save the material data
for i in range(partition_table.shape[0]):
    ## build an empty dataframe for each partition
    df_empty = pd.DataFrame(columns =['image_id', 'Labelfine', 'Label'],
                             index=[i for i in range(partition_table.shape[1]-1)])
    #print(df_empty)

    for j in range(partition_table.shape[1]-1):
        image_id = partition_table[i][j+1]
        df_empty.loc[j,'image_id'] = image_id
        df_empty.loc[j,'Labelfine'] = material_data.loc[image_id-1]['Labelfine']
        df_empty.loc[j, 'Label'] = material_data.loc[image_id - 1]['Label']
    #print(df_empty)
    #save the parition i data frame files
    image_meta_data_file = 'meta_data.csv'
    df_empty.to_csv(os.path.join(input_file_path, 'partition_'+str(i+1)+'_'+image_meta_data_file))




print("test")
#check_unique