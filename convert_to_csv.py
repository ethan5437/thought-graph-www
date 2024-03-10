import pickle

import sys
sys.path.insert(0, 'src') 
sys.path.insert(0, '/data/ch52669/gene_interaction/tot-gene-sets/MedAgents') 
import os
import json
from functools import partial

import seaborn as sns
import numpy as np

INPUT_FILE_NAME = #'/data/ch52669/gene_interaction/tot-gene-sets/results2/tot2_eval_15gen_5steps_no_certainty_9nodes.pkl'
OUPPUT_FILE_NAME = #

with open(INPUT_FILE_NAME, 'rb') as f:
    data = pickle.load(f)
    data = data[:]
    print(FILE_NAME, len(data))  
    
    
df_list = []
for result in data:
    index = result['index']
    y_true = result['label']
    for index_in_sample, t in enumerate(result['steps']['steps']): 
        x = t['x']
        for node_index, node in enumerate(t['sorted_grouped_items']):
            y_preds = node[1]['grouped_names'].split(', ')
            for index_in_layer, y_pred in enumerate(y_preds):
                df_list.append([index, index_in_sample, index_in_layer, node_index, x, y_pred, y_true])
                
import pandas as pd

# Your list of lists

columns = ['Index', 'Index in Sample', 'Index in Layer', 'Node Index', 'x' , 'y_pred', 'y_true']
# Creating a DataFrame
df = pd.DataFrame(df_list, columns=columns)
df.to_csv(OUPPUT_FILE_NAME)