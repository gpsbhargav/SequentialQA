import os
import pickle
from tqdm import tqdm
import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

from tqdm import tqdm

import pdb

def pickler(path,pkl_name,obj):
    with open(os.path.join(path, pkl_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickler(path,pkl_name):
    with open(os.path.join(path, pkl_name) ,'rb') as f:
        obj = pickle.load(f)
    return obj


TRAINING = False

if TRAINING:
    in_pkl_path = './'
    in_pkl_name = "preprocessed_train.pkl"
    out_pkl_path = './'
    out_pkl_name = "preprocessed_standard_train.pkl"
else:
    in_pkl_path = './'
    in_pkl_name = "preprocessed_dev.pkl"
    out_pkl_path = './'
    out_pkl_name = "preprocessed_standard_dev.pkl"


class Coqa(Dataset):
    
    def __init__(self,dataset):
        self.dataset = dataset
    
    def __len__(self):
        assert(len(self.dataset["data_points"]) == len(self.dataset['supporting_facts']))
        return len(self.dataset["data_points"])
    
    def __getitem__(self,index):
        passage_index, question_index, prev_indices = self.dataset["data_points"][index]
        
        question_word = self.dataset['questions_word'][question_index]
        question_char = self.dataset['questions_char'][question_index]
        
        supporting_facts = self.dataset['supporting_facts'][index]
        
        out_dict =  {'question_word':np.array(question_word), 'question_char':np.array(question_char), 'supporting_facts':supporting_facts}
        
        # include "unpadded_passage_lengths"
        out_dict["unpadded_passage_lengths"] = self.dataset["unpadded_passage_lengths"][passage_index]
        
        for i in range(len(self.dataset['passages_word'])):
            out_dict["sent_word_{}".format(i)] = self.dataset['passages_word'][i][passage_index]
            out_dict["sent_char_{}".format(i)] = self.dataset['passages_char'][i][passage_index]
            
            
        for i,idx in enumerate(prev_indices):
            qa_word = self.dataset['questions_word'][idx] + self.dataset['answers_word'][idx]
            qa_char = self.dataset['questions_char'][idx] + self.dataset['answers_char'][idx]
            out_dict["history_word_{}".format(i)] = qa_word
            out_dict["history_char_{}".format(i)] = qa_char
            
        return out_dict
    
    
data = unpickler(in_pkl_path, in_pkl_name)

dataset = Coqa(data)


mega_dataset = []

for i in range(len(dataset)):
    mega_dataset.append(dataset[i])
    
mega_aligned_dataset = {}

keys = list(mega_dataset[0].keys())

for key in keys:
    mega_aligned_dataset[key] = []

for item in tqdm(mega_dataset):
    for key in keys:
        mega_aligned_dataset[key].append(item[key])

# pdb.set_trace()

print("writing pkl")
pickler(out_pkl_path, out_pkl_name, mega_aligned_dataset)

print("Done")