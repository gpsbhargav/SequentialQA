class CoqaOptions:
    def __init__(self):
        self.epochs = 20
        self.batch_size = 128
        self.bi_rnn_hidden_state = 150   # 150 forward and 150 backward =  300D hidden state
        self.num_rnn_layers = 1
        self.linear_layer_in_size = self.bi_rnn_hidden_state * 2
        self.word_embedding_size = 300
        self.char_embedding_size = 50
        self.use_char_embeddings = False
        self.history_size = 0
        self.max_sent_len = 40
        self.max_para_len = 25
        self.max_word_len = 10
        self.word_vocab_size = 37212
        self.char_vocab_size = 592
        self.char_pad_index = 0
        self.word_pad_index = 0
        self.dropout = 0.00001
        self.recurrent_dropout = 0.0001
        self.lr = 0.001
        self.max_gradient_norm = 1
        self.update_word_embeddings = False
        self.attention_type = 'dot'  # one of ['dot', 'general', 'concat']
        self.data_pkl_path = "../data/coqa/"
        self.glove_store = "precomputed_glove.npy"
        self.word_vocab_pkl_name = "word_vocabulary.pkl"
        self.char_vocab_pkl_name = "char_vocabulary.pkl"
        self.train_pkl_name = "preprocessed_standard_train.pkl"
        self.dev_pkl_name = "preprocessed_standard_dev.pkl"
        self.save_path = "../saved_models_coqa/"
        self.predictions_pkl_name = "predictions.pkl"
        self.log_every = 1
        self.save_every = self.log_every * 500
#         self.dev_every = 500
        self.early_stopping_patience = 4
        self.gpu = 0
        
        assert(self.log_every < self.save_every)
        
        if(self.use_char_embeddings):
            self.total_word_embedding_size = self.word_embedding_size + self.char_embedding_size
        else:
            self.total_word_embedding_size = self.word_embedding_size
        
        if(self.history_size != 0):
            self.final_question_representation_size = (self.bi_rnn_hidden_state * 2) * 2
        else:
            self.final_question_representation_size = self.bi_rnn_hidden_state * 2