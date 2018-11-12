from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np
import nltk
import random
import math
import os
import dill as pickle
from tqdm import tqdm

import torch



def pickler(path,pkl_name,obj):
    with open(os.path.join(path, pkl_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickler(path,pkl_name):
    with open(os.path.join(path, pkl_name) ,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_glove(dir,file_name):
    return np.load(dir+file_name)


def minibatch_indices_generator(data_len, batch_size, shuffle=True):
    if(shuffle):
        indices = np.random.permutation(data_len)
    else:
        indices = list(range(data_len))
    for i in range(0, data_len, batch_size):
        if(i + batch_size >= data_len):
            yield indices[i:]
        else:
            yield indices[i:i+batch_size]


            

class MinibatchFromIndices:
    
    def __init__(self, data, device):
        print("Loading data")
        self.keys = list(data.keys())
        self.device = device
        self.data = {}
        #create tensors on given device
        for key in data.keys():
            if(key == "supporting_facts"):
                self.data[key] = torch.tensor(data[key], dtype=torch.float32)
            else:
                self.data[key] = torch.tensor(data[key])
        print("Finished loading data")
            
    def get(self, indices):
        out_dict = {}
        for key in self.data.keys():
            out_dict[key] = self.data[key][indices].to(self.device)
        return out_dict
        

