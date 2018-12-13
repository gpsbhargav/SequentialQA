import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class SentenceSelector(nn.Module):
    def __init__(self, options, weight_matrix, device):
        super(SentenceSelector, self).__init__()
        
        self.options = options
        
        self.device = device
        self.word_embedding = nn.Embedding(options.word_vocab_size, options.word_embedding_size, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
        self.word_embedding.weight.requires_grad = options.update_word_embeddings
        
        self.answer_sentence_encoder = SequenceEncoder(options, options.final_question_representation_size, options.total_word_embedding_size, attention_type='dot')
        
        self.question_encoder =  SequenceEncoder(options, options.total_word_embedding_size, options.total_word_embedding_size, attention_type='dot')
        
        self.history_item_encoder =  SequenceEncoder(options, options.total_word_embedding_size, options.total_word_embedding_size, attention_type='concat')
        
        
        if(self.options.embedding_type == 'word_plus_char'):
            self.char_embedding = nn.Embedding(options.char_vocab_size, options.trainable_embedding_size, padding_idx=options.char_pad_index, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)

            self.char_embedding.weight.requires_grad = True

            self.char_cnn = nn.Conv1d(in_channels=options.trainable_embedding_size, 
                                  out_channels=options.trainable_embedding_size, 
                                  kernel_size=options.max_word_len, 
                                  stride=options.max_word_len, 
                                  padding=0, dilation=1, groups=1, bias=True)
        elif(self.options.embedding_type == 'two_channel_word'):
            self.word_embedding_channel_2 = nn.Embedding(options.word_vocab_size, options.trainable_embedding_size, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
        
        self.rnn = nn.LSTM(input_size= options.bi_rnn_hidden_state * 2, hidden_size=options.bi_rnn_hidden_state, num_layers = options.num_rnn_layers, batch_first = True, dropout = options.recurrent_dropout, bidirectional=True)
        
        
        self.question_linear_layer = nn.Linear(options.bi_rnn_hidden_state * 2 * 2, options.bi_rnn_hidden_state * 2, bias=True)
        
        if(self.options.history_size != 0):
            self.history_attn_model = Attn(options, options.bi_rnn_hidden_state * 2, options.bi_rnn_hidden_state * 2, method='concat')
        
        self.sentence_selection_attn_model = Attn(options, options.bi_rnn_hidden_state * 2, options.final_question_representation_size, method='dot')
        
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        
        

        

    def get_first_hidden(self, batch_size, device):
        return (torch.randn(self.options.num_rnn_layers * 2, batch_size, self.options.bi_rnn_hidden_state, device = device),torch.randn(self.options.num_rnn_layers * 2, batch_size, self.options.bi_rnn_hidden_state, device = device))
    
    
    def build_embedding(self,word_sequences_in, char_sequences_in):
        if(self.options.embedding_type == 'word_plus_char'):
            char_embeddings = self.char_embedding(char_sequences_in)
            char_embedding_through_cnn = self.char_cnn(char_embeddings.transpose(1,2))
            char_level_word_embeddings = char_embedding_through_cnn.transpose(1,2)
            words_embedded = self.word_embedding(word_sequences_in)
            combined = torch.cat([words_embedded,char_level_word_embeddings],dim=-1)
            return combined
        elif(self.options.embedding_type == 'two_channel_word'):
            words_embedded = self.word_embedding(word_sequences_in)
            words_embedded_channel_2 = self.word_embedding_channel_2(word_sequences_in)
            combined = torch.cat([words_embedded,words_embedded_channel_2],dim=-1)
            return combined
        else:
            words_embedded = self.word_embedding(word_sequences_in)
            return words_embedded
        
     
    def construct_mask(self, tensor_in, padding_index = 0):
        mask = tensor_in == padding_index
        float_mask = mask.type(dtype=torch.float32)
        float_mask =  float_mask.masked_fill(mask=mask, value=-1e-20)
        return float_mask
    
    def forward(self, data, rnn_hidden):

        batch_size = len(data["question_word"])
        
        combined_embeddings_input = {}

        combined_embeddings_input["question"] = self.build_embedding(data["question_word"], data["question_char"])
        
        mask = self.construct_mask(tensor_in=data["question_word"], padding_index = self.options.word_pad_index)        
        question = self.question_encoder(input=combined_embeddings_input["question"], bi_rnn_h0=rnn_hidden, query = None, mask=mask)
        

        if(self.options.history_size != 0):
            history_reps_list = []
            
            for i in range(self.options.history_size):
                history_embedding = self.build_embedding(data["history_word_{}".format(i)], data["history_char_{}".format(i)])
                mask = self.construct_mask(tensor_in=data["history_word_{}".format(i)], padding_index = self.options.word_pad_index)
                history_encoding = self.history_item_encoder(input=history_embedding, bi_rnn_h0= rnn_hidden, query = question.unsqueeze(1), mask=mask)
                history_reps_list.append(history_encoding)
             
            if(self.options.history_size > 1):
                history_reps = torch.stack(history_reps_list,dim=1)   
                attn_weights = self.history_attn_model(question.unsqueeze(1), history_reps)
                final_history_rep = attn_weights.bmm(history_reps)
                final_history_rep = final_history_rep.flatten(1)
                final_history_rep = torch.tanh(final_history_rep)
            else:
                final_history_rep = history_reps_list[0]
            
            question_history_concat = torch.cat([question,final_history_rep],dim=-1)
            history_gate = torch.sigmoid(self.question_linear_layer(question_history_concat))
            gated_history = history_gate * final_history_rep
            history_aware_question = question + gated_history
        else:
            history_aware_question = question
        
        sent_reps = []
        for i in range(self.options.max_para_len):
            combined_embeddings_input = self.build_embedding(data["sent_word_{}".format(i)], data["sent_char_{}".format(i)])
            mask = self.construct_mask(tensor_in=data["sent_word_{}".format(i)], padding_index = self.options.word_pad_index)
            sent_rep = self.answer_sentence_encoder(input=combined_embeddings_input,
                            bi_rnn_h0=rnn_hidden, query = history_aware_question.unsqueeze(1), mask=mask)
            sent_reps.append(sent_rep)
        
        
        paragraph_rep = torch.stack(sent_reps,dim=1)
        
#         paragraph_rep = self.dropout(paragraph_rep)
        
#         paragraph_rep, _ = self.rnn(paragraph_rep, rnn_hidden)
        
        raw_scores = self.sentence_selection_attn_model(history_aware_question.unsqueeze(1), paragraph_rep, normalize_scores=False)
        
        unpadded_paragraph_lengths = data["unpadded_passage_lengths"]
        mask = self.construct_mask(unpadded_paragraph_lengths, padding_index = 0)
        
        raw_scores = raw_scores + mask
        
        return raw_scores
        
    
        


class SequenceEncoder(nn.Module):
    def __init__(self, options, query_size, input_size, attention_type):
        super(SequenceEncoder, self).__init__()
        
        self.rnn = nn.LSTM(input_size= input_size, hidden_size=options.bi_rnn_hidden_state, num_layers = options.num_rnn_layers, batch_first = True, dropout = options.recurrent_dropout, bidirectional=True)
        
        output_size = options.bi_rnn_hidden_state*2

        self.linear = nn.Linear(output_size, output_size, bias=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        
        self.fixed_query = nn.Parameter(torch.randn(1, 1, query_size, requires_grad=True))
        
        self.attn_model = Attn(options, output_size, query_size, method=attention_type)
    
    
    def forward(self, input, bi_rnn_h0, mask = None, query = None):
        if(query is None):
            query = self.fixed_query
        seq_rep, _ = self.rnn(input, bi_rnn_h0)
        seq_rep = self.dropout(seq_rep)
        seq_rep_translated = self.linear(seq_rep)
        seq_rep_translated = self.activation(seq_rep_translated)
        attn_weights = self.attn_model(query, seq_rep_translated, mask=mask)
        final_seq_rep = attn_weights.bmm(seq_rep_translated)
        final_seq_rep = final_seq_rep.flatten(1)
        return final_seq_rep
        
    

    
# This class is adapted from the pytorch tutorial https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
class Attn(torch.nn.Module):
    def __init__(self, options, size_of_things_to_attend_to, size_of_query, method):

        super(Attn, self).__init__()
        hidden_size = size_of_things_to_attend_to
        self.method = method
        if self.method not in ['dot', 'general', 'concat', 'multiplicative']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(hidden_size, hidden_size)
        elif self.method == 'multiplicative':
            self.attn = torch.nn.Linear(size_of_query , size_of_things_to_attend_to ,bias=False)
            self.v = torch.nn.Parameter(torch.randn(size_of_things_to_attend_to, requires_grad=True))
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(size_of_things_to_attend_to + size_of_query , options.attention_linear_layer_out_dim)
            self.v = torch.nn.Parameter(torch.randn(options.attention_linear_layer_out_dim, requires_grad=True))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=-1)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat([hidden.expand(encoder_output.size(0), encoder_output.size(1), hidden.size(-1)), encoder_output],dim=-1)).tanh()
        return torch.sum(self.v * energy, dim=2)
    
    def multiplicative_score(self, hidden, encoder_output):
        transformed_encoder_output = self.attn(encoder_output)
        attention_scores = self.dot_score(hidden, transformed_encoder_output)
        return attention_scores

    def forward(self, hidden, encoder_outputs, mask = None, normalize_scores=True):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'multiplicative':
            attn_energies = self.multiplicative_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()
        
        if(mask is not None):
            attn_energies = attn_energies + mask
        
        if(normalize_scores):
            # Return the softmax normalized probability scores (with added dimension)
            return_value =  F.softmax(attn_energies, dim=1).unsqueeze(1)
        else:
            return_value =  attn_energies
        
        return return_value
        
        
            
            