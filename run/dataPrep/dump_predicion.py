import os
import numpy as np
import pickle
import pandas as pd

prediction_path = os.path.join('../../data/kaggle_ratio_0.2/output', 'oracle_ratio_0.2.csv')
#prediction_path = os.path.join('../../data/kaggle/output', 'INMO-GCN.csv')
#prediction_path = os.path.join('../../data/kaggle/output', 'LightGCN.csv')
#prediction_path = os.path.join('../../data/kaggle/output', 'IDCF-GCN.csv')
second_level_data_node_map_path  = os.path.join('../../data/kaggle_ratio_0.2/output', 'second_data_map.txt')
data_node_map_path = os.path.join('../../data/kaggle_ratio_0.2/output', 'data_map.txt')
model_node_map_path = os.path.join('../../data/kaggle_ratio_0.2/output', 'model_map.txt')
#train_dict_path = os.path.join('../../data/kaggle/output', 'inductive_train_dict.txt')

'''
train_dict_file = open(train_dict_path , 'rb')
train_dict = pickle.loads(train_dict_file.read())
train_dict_file.close()
'''



prediction_matrix = np.loadtxt(prediction_path , delimiter=',')
second_level_data_node_map_file = open(second_level_data_node_map_path, 'rb')
second_level_data_node_map = pickle.loads(second_level_data_node_map_file.read())
second_level_data_node_map_file.close()
data_node_map_file = open(data_node_map_path,'rb')
data_node_map = pickle.loads(data_node_map_file.read())
data_node_map_file.close()
model_node_map_file = open(model_node_map_path,'rb')
model_node_map = pickle.loads(model_node_map_file.read())
model_node_map_file.close()
transformed_prediction = np.copy(prediction_matrix)

inv_second_level_data_node_map = {v: k for k, v in second_level_data_node_map.items()}
inv_data_node_map = {v: k for k, v in data_node_map.items()}
inv_model_node_map = {v: k for k, v in model_node_map.items()}

for i in range(transformed_prediction.shape[0]):
    for j in range(transformed_prediction.shape[1]):
        if j==0:
            key =transformed_prediction[i][j]
            temp= inv_second_level_data_node_map[key]
            transformed_prediction[i][j] = inv_data_node_map[temp]
        elif j==1:
            key = transformed_prediction[i][j]
            transformed_prediction[i][j] = inv_model_node_map[key]

df = pd.DataFrame(transformed_prediction, columns=['dataset', 'model','balanced_accuracy'])
print(df.dtypes)
df['dataset']=df['dataset'].astype('int')
df['model']=df['model'].astype('int')
transformed_pred_path = os.path.join('../../data/kaggle_ratio_0.2/output/final_result', 'oracle_edge_60.csv')
df.to_csv(transformed_pred_path,index=False)



print("test")