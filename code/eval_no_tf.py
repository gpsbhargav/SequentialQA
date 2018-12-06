# TODO:
# Copy that dataset class. make it return unpadded question, turn_id
# Add a batch dimention in the data. Do this in the dataset class
# Load model weights (code on my bitbucket)
# cache previous two answer sentences
# cache previous two unpadded questions
# create history depending on turn_id
# feed data to model
# store predictions(no change)



import utils
from model_fixed_size_query import SentenceSelector
import options

import torch
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

from sklearn.metrics import accuracy_score

from tqdm import tqdm

import os
import time
import glob
from collections import deque

import pdb


class Coqa(Dataset):
    
    def __init__(self,dataset,device):
        self.dataset = dataset
        self.separator = self.dataset['separator_id']
    
    def __len__(self):
        assert(len(self.dataset["data_points"]) == len(self.dataset['supporting_facts']))
        return len(self.dataset["data_points"])
    
    def __getitem__(self,index):
        passage_index, question_index, prev_indices = self.dataset["data_points"][index]
        
        turn_id = self.dataset["turn_ids"][index] - 1 # turn_ids start from 1 
        
        question_word = self.dataset['questions_word'][question_index]
        question_char = self.dataset['questions_char'][question_index]
        
        question_word_unpadded = self.dataset['questions_word_unpadded'][question_index] + [self.separator]
#         question_char_unpadded = self.dataset['questions_char_unpadded'][question_index]
        
        
        supporting_facts = self.dataset['supporting_facts'][index]
        
        data_point =  {'question_word':np.array(question_word), 'question_char':np.array(question_char), 'supporting_facts':supporting_facts, 'question_word_unpadded':np.array(question_word_unpadded),
                    'turn_id':turn_id}
        
        # include "unpadded_passage_lengths"
        data_point["unpadded_passage_lengths"] = self.dataset["unpadded_passage_lengths"][passage_index]
        
        for i in range(len(self.dataset['passages_word'])):
            data_point["sent_word_{}".format(i)] = self.dataset['passages_word'][i][passage_index]
            data_point["sent_char_{}".format(i)] = self.dataset['passages_char'][i][passage_index]
            
        # empty histories. Has to be replaced later according to turn_id
        for i,idx in enumerate(prev_indices):
            data_point["history_word_{}".format(i)] = self.dataset['questions_word'][0]
            data_point["history_char_{}".format(i)] = self.dataset['questions_char'][0]
            
        out_dict = {}
        for key in data_point.keys():
            if(key in ['turn_id']):
                out_dict[key] = data_point[key]
            elif(key == "supporting_facts"):
                out_dict[key] = torch.tensor(data_point[key], dtype=torch.float32,device=device).unsqueeze(0)
            else:
                out_dict[key] = torch.tensor(data_point[key], device=device).unsqueeze(0)
            
        return out_dict

    
options = options.CoqaOptions()    
torch.cuda.set_device(0)
device = torch.device('cuda:{}'.format(options.gpu))

model_file = options.save_path + "best_snapshot_dev_EM_0.49555305023174245_iter_9339_model.pt"
in_pkl_name = "preprocessed_dev_no_tf.pkl"


data_raw = utils.unpickler(options.data_pkl_path, in_pkl_name)

dev_data = Coqa(data_raw,device)


glove = utils.load_glove(options.data_pkl_path,options.glove_store)

print("Loading model")


model = SentenceSelector(options, glove, device)
model.to(device)
model.load_state_dict(torch.load(model_file))



print("===============================")
print("Model:")
print(model)
print("===============================")


dev_data_len = len(dev_data)

print("Evaluating")
# switch model to evaluation mode
model.eval();

answers_for_whole_dev_set = []

if(options.history_size > 0):
    question_history = deque(maxlen=options.history_size)
    answer_history = deque(maxlen=options.history_size)
    

with torch.no_grad():
    for i in tqdm(range(dev_data_len)):
        dev_batch = dev_data[i]
        dev_rnn_first_hidden = model.get_first_hidden(len(dev_batch["question_word"]), device=device)
        if(dev_batch['turn_id'] > 0):
            for i in range(min(options.history_size, dev_batch['turn_id'])):
                h = torch.cat([question_history[i], answer_history[i]],dim=1).to(device)
                dev_batch["history_word_{}".format(i)] = h
        elif(options.history_size > 0):
            question_history.clear()
            answer_history.clear()
        dev_answer = model(dev_batch,dev_rnn_first_hidden)
        answers_for_whole_dev_set.append(dev_answer.cpu().numpy())
        if(options.history_size > 0):
            answer_sentence_index = dev_answer.cpu().numpy().argmax(axis=1)[0]
            answer_sentence = dev_batch["sent_word_{}".format(answer_sentence_index)]
            answer_history.appendleft(answer_sentence)
            question_history.appendleft(dev_batch["question_word_unpadded"])


answers_for_whole_dev_set = np.concatenate(answers_for_whole_dev_set, axis = 0)

dev_answer_labels = answers_for_whole_dev_set.argmax(axis=1)

dev_exact_match = accuracy_score(np.array(data_raw["supporting_facts"]).argmax(axis = 1), dev_answer_labels)

print("Exact match = {}".format(dev_exact_match))


    
        

# save predictions
utils.pickler(options.save_path, "predictions_no_tf.pkl", answers_for_whole_dev_set)
print("Saved predictions")
        
            


            

        
        
        
        
        
        