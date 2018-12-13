class CoqaOptions:
    def __init__(self):
        self.epochs = 30
        self.batch_size = 64
        self.num_rnn_layers = 1
        self.word_embedding_size = 300
        self.trainable_embedding_size = 50  #channel 2 of embeddings. can be char level or word
        self.embedding_type = 'word_only'  # one of 'two_channel_word', 'word_plus_char', 'word_only'
        self.history_size = 2
        self.max_para_len = 25
        self.word_vocab_size = 36238
        self.char_vocab_size = 598
        self.char_pad_index = 0
        self.word_pad_index = 0
        self.dropout = 0.3
        self.recurrent_dropout = 0.2
        self.lr = 0.001
        self.weight_decay = 0
        self.max_gradient_norm = 1
        self.update_word_embeddings = False
        self.data_pkl_path = "../data/coqa/"
        self.glove_store = "precomputed_glove.npy"
        self.word_vocab_pkl_name = "word_vocabulary.pkl"
        self.char_vocab_pkl_name = "char_vocabulary.pkl"
        self.train_pkl_name = "preprocessed_standard_train.pkl"
        self.dev_pkl_name = "preprocessed_standard_dev.pkl"
        self.save_path = "../saved_models_coqa/"
        self.predictions_pkl_name = "predictions.pkl"
        self.log_every = 100
        self.save_every = self.log_every * 500
        self.early_stopping_patience = 4
        self.gpu = 0
        
        assert(self.log_every < self.save_every)
        
        if(self.embedding_type != 'word_only'):
            self.total_word_embedding_size = self.word_embedding_size + self.trainable_embedding_size
        else:
            self.total_word_embedding_size = self.word_embedding_size
        
        self.bi_rnn_hidden_state = int(self.total_word_embedding_size / 2)
        self.linear_layer_in_size = self.bi_rnn_hidden_state * 2
        self.attention_linear_layer_out_dim = self.bi_rnn_hidden_state
        
        if(self.history_size != 0):
            self.final_question_representation_size = self.bi_rnn_hidden_state * 2 
        else:
            self.final_question_representation_size = self.bi_rnn_hidden_state * 2
