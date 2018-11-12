import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class SentenceSelector(nn.Module):
    def __init__(self, options, weight_matrix, device):
        super(SentenceSelector, self).__init__()
        
        self.device = device
        self.word_embedding = nn.Embedding(options.word_vocab_size, options.word_embedding_size, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
        self.word_embedding.weight.requires_grad = options.update_word_embeddings
        
        self.char_embedding = nn.Embedding(options.char_vocab_size, options.char_embedding_size, padding_idx=options.char_pad_index, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
        
        self.answer_sentence_encoder = SequenceEncoder(options, query_size_multiplier = 3)
        self.conditional_question_sentence_encoder =  SequenceEncoder(options, query_size_multiplier = 2)
        
        self.char_cnn = nn.Conv1d(in_channels=options.char_embedding_size, 
                                  out_channels=options.char_embedding_size, 
                                  kernel_size=options.max_word_len, 
                                  stride=options.max_word_len, 
                                  padding=0, dilation=1, groups=1, bias=True)
        
        self.bigru = nn.GRU(input_size= options.bi_rnn_hidden_state * 2, hidden_size=options.bi_rnn_hidden_state, num_layers = options.num_rnn_layers, batch_first = True, dropout = options.recurrent_dropout, bidirectional=True)
        
        self.options = options
        
        self.attn_model = Attn(options,query_size_multiplier=3)
        
#         self.h0 = nn.Parameter(torch.randn(2, options.batch_size, options.bi_rnn_hidden_state), requires_grad = False)

        

    def get_first_hidden(self,batch_size, device):
        return torch.randn(2, batch_size, self.options.bi_rnn_hidden_state, device = device)
    
    
    def build_embedding(self,word_sequences_in, char_sequences_in):
        char_embeddings = self.char_embedding(char_sequences_in)
        char_embedding_through_cnn = self.char_cnn(char_embeddings.transpose(1,2))
        char_level_word_embeddings = char_embedding_through_cnn.transpose(1,2)
        words_embedded = self.word_embedding(word_sequences_in)
        combined = torch.cat([words_embedded,char_level_word_embeddings],dim=-1)
        return combined
    
    def forward(self,data,rnn_hidden):
    # Embed everything at word level. embedded_words={}
    # Embed everything at char level.Run through embedding layer and then CNN embedded_chars={}
    # Concatenate both. combined_embeddings_input={}
    # Encode question1 to get q1
    # Encode history1 based on question1 to get h1
    # Concatenate q1 and h1
    # Encode history2 based on [q1;h1] to get h2
    # concatenate all to get [q1;h1;h2]. call it history_aware_question
    # Encode all sentences in "combined_embeddings_input" based on history_aware_question.
    # Run the above through a biGru to get final_sentence_representations
    # Return un-normalized attention scores between history_aware_question and final_sentence_representations
    
        batch_size = len(data["question_word"])
        
        combined_embeddings_input = {}

        combined_embeddings_input["question"] = self.build_embedding(data["question_word"], data["question_char"])
        
        combined_embeddings_input["history_0"] = self.build_embedding(data["history_word_0"], data["history_char_0"])
        
        combined_embeddings_input["history_1"] = self.build_embedding(data["history_word_1"], data["history_char_1"])
        
        for i in range(self.options.max_para_len):
            combined_embeddings_input["sent_{}".format(i)] = self.build_embedding(data["sent_word_{}".format(i)], data["sent_char_{}".format(i)])
                
        question1 = self.conditional_question_sentence_encoder(input=combined_embeddings_input["question"], bi_rnn_h0=rnn_hidden, query = None)
        
        question1_extended = torch.cat([question1, 
           torch.randn(batch_size,self.options.bi_rnn_hidden_state * 2, device=self.device )],dim=-1)
                
        history0_encoding = self.conditional_question_sentence_encoder(input=combined_embeddings_input["history_1"], bi_rnn_h0=rnn_hidden, query = question1_extended.unsqueeze(1))
        
        q1_h0 = torch.cat([question1,history0_encoding],dim=-1)
        
        history1_encoding = self.conditional_question_sentence_encoder(input=combined_embeddings_input["history_1"], bi_rnn_h0=rnn_hidden, query = q1_h0.unsqueeze(1))
        
        
        history_aware_question = torch.cat([q1_h0, history1_encoding],dim=-1)
        
        
        sent_reps = []
        for i in range(self.options.max_para_len):
            sent_rep = self.answer_sentence_encoder(input=combined_embeddings_input["sent_{}".format(i)],
                            bi_rnn_h0=rnn_hidden, query = history_aware_question.unsqueeze(1))
            sent_reps.append(sent_rep)
        
        
        paragraph_rep = torch.stack(sent_reps,dim=1)
                
        paragraph_rep, _ = self.bigru(paragraph_rep, rnn_hidden)
        
        raw_scores = self.attn_model(history_aware_question.unsqueeze(1), paragraph_rep,normalize_scores=False)
        
        return raw_scores
        
    
        


class SequenceEncoder(nn.Module):
    def __init__(self, options, query_size_multiplier = 1):
        super(SequenceEncoder, self).__init__()
        
        output_size = options.bi_rnn_hidden_state * 2
        
        input_size = options.total_word_embedding_size

        self.bigru = nn.GRU(input_size= input_size, hidden_size=options.bi_rnn_hidden_state, num_layers = options.num_rnn_layers, batch_first = True, dropout = options.recurrent_dropout, bidirectional=True)

        self.linear = nn.Linear(output_size, output_size, bias=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        
        self.fixed_query = nn.Parameter(torch.randn(1, 1, output_size*query_size_multiplier))
        
        self.attn_model = Attn(options,query_size_multiplier=query_size_multiplier)
    
    
    def forward(self, input, bi_rnn_h0, query = None):
        if(query is None):
            query = self.fixed_query
        seq_rep, _ = self.bigru(input, bi_rnn_h0)
        seq_rep_translated = self.linear(seq_rep)
        seq_rep_translated = self.activation(seq_rep_translated)
        attn_weights = self.attn_model(query, seq_rep_translated)
        final_seq_rep = attn_weights.bmm(seq_rep_translated)
        final_seq_rep = final_seq_rep.flatten(1)
        return final_seq_rep
        
    

    
# This class is adapted from the pytorch tutorial https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
class Attn(torch.nn.Module):
    def __init__(self, options, query_size_multiplier = 1):
# query_size_multiplier is hacky. It is used to caclulate the size of the weight
# matrix for attention. Assumption:  the encoder states are of size 
# options.bi_rnn_hidden_state * 2 and the query is of size 
# options.bi_rnn_hidden_state + options.bi_rnn_hidden_state * query_size_multiplier. The last quantity gives 
# the size of the concatenation of the encoder state and query
        
# Attention methods other than 'concat' may not work for query_size_multiplier != 1

        super(Attn, self).__init__()
        method = options.attention_type
        hidden_size = options.bi_rnn_hidden_state * 2
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size + self.hidden_size * query_size_multiplier, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=-1)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # this if-else block doesn't generalize. Careful while using this in another situation
#         if(len(hidden_in.shape) == 2):
#             hidden = hidden_in.unsqueeze(1)
#         else:
#             hidden = hidden
        energy = self.attn(torch.cat([hidden.expand(encoder_output.size(0), encoder_output.size(1), hidden.size(-1)), encoder_output],dim=-1)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs,normalize_scores=True):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()
        
        if(normalize_scores):
            # Return the softmax normalized probability scores (with added dimension)
            return_value =  F.softmax(attn_energies, dim=1).unsqueeze(1)
        else:
            return_value =  attn_energies
        
        return return_value
        
        
            
            