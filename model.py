import numpy as np
import h5py

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, GRU, CuDNNLSTM, CuDNNGRU
from keras.optimizers import RMSprop
from keras.initializers import Orthogonal
from keras.callbacks import EarlyStopping
from keras import metrics
from keras.utils import plot_model

import utils
from preprocessing import preprocess_sentence, preprocess_sparql


class MyVector:

    def get_vocabulary(self):
        raise NotImplementedError

    def get_vocabulary_size(self):
        raise NotImplementedError

    def get_dimension(self):
        raise NotImplementedError

    def get_word_vector(self, word):
        raise NotImplementedError

    def get_word_id(self, word):
        raise NotImplementedError


class NaiveVector(MyVector):

    def __init__(self, file_path):
        self.size, self.dimension, self.vectors = self.__load_vectors(
            file_path)
        self.vocabulary = list(self.vectors.keys())
        self.word2index = {w: i for i, w in enumerate(self.vocabulary)}

    def get_vocabulary(self):
        return self.vocabulary

    def get_vocabulary_size(self):
        return self.size

    def get_dimension(self):
        return self.dimension

    def get_word_vector(self, word):
        return self.vectors[word]

    def get_word_id(self, word):
        return self.word2index[word]

    def __load_vectors(self, fname):
        """
        Load word embeddings from a pre-trained vector file
        """
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # number of words and dimensions
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return n, d, data


class MyModel:
    """MyModel"""

    # Default parameters
    batch_size = 64
    epochs = 100
    dropout = 0.2
    hidden_units = 128
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    num_layers = 1
    use_attention = False
    beam_size = 1

    sos_symbol = '<s>'
    eos_symbol = '</s>'
    unk_symbol = '<unk>'

    def __init__(self,
                 input_word_vector,
                 target_word_vector=None,
                 hidden_units=128,
                 num_layers=1,
                 dropout=0.2,
                 use_attention=False,
                 beam_size=1):
        """Creates a model"""
        self.input_word_vector = input_word_vector
        self.target_word_vector = target_word_vector
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.beam_size = beam_size

    def __build_vocabulary(self, input_texts, target_texts):
        if self.input_word_vector is not None:
            self.input_vocabulary = self.input_word_vector.get_vocabulary()
            self.encoder_embedding_size = self.input_word_vector.get_dimension()
        else:
            self.input_vocabulary = utils.build_vocabulary_from_texts(
                input_texts)
        self.encoder_vocab_size = len(self.input_vocabulary)

        if self.target_word_vector is not None:
            self.output_vocabulary = self.target_word_vector.get_vocabulary()
            self.decoder_embedding_size = self.target_word_vector.get_dimension()
        else:
            self.output_vocabulary = utils.build_vocabulary_from_texts(
                target_texts)
        self.decoder_vocab_size = len(self.output_vocabulary)

    def __build_model(self):

        self.encoder_states_layered = []

        # Encoder
        self.encoder_inputs = Input(shape=(None, self.encoder_embedding_size))

        return_sequences = True

        for i in range(self.num_layers):
            if i == self.num_layers-1:
                return_sequences = False
            # encoder_embedding = Embedding(num_input_words+1, embedding_size, weights=[embedding_matrix], trainable=False)(encoder_inputs)
            if tf.test.is_gpu_available():
                encoder_lstm = CuDNNLSTM(self.hidden_units, return_sequences=return_sequences, return_state=True, dropout=self.dropout, recurrent_dropout=self.dropout,
                                kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
            else:
                encoder_lstm = LSTM(self.hidden_units, return_sequences=return_sequences, return_state=True, dropout=self.dropout, recurrent_dropout=self.dropout,
                                kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
            if i == 0:
                self.encoder_outputs, state_hidden, state_cell = encoder_lstm(self.encoder_inputs)
            else:
                self.encoder_outputs, state_hidden, state_cell = encoder_lstm(self.encoder_outputs)
            encoder_states = [state_hidden, state_cell]
            self.encoder_states_layered.extend(encoder_states)

        # Decoder
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_embedding = Embedding(
            self.decoder_vocab_size, self.decoder_embedding_size)(self.decoder_inputs)
        self.decoder_layers = []

        for i in range(self.num_layers):
            
            if tf.test.is_gpu_available():
                decoder_lstm = CuDNNLSTM(self.hidden_units, return_sequences=True, return_state=True, dropout=self.dropout,
                                    recurrent_dropout=self.dropout, kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
            else:
                decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True, dropout=self.dropout,
                                    recurrent_dropout=self.dropout, kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
            
            initial_state = [self.encoder_states_layered[2*i], self.encoder_states_layered[2*i+1]]

            if i == 0:
                self.decoder_outputs, _, _ = decoder_lstm(
                    self.decoder_embedding, initial_state=initial_state)
            else:
                self.decoder_outputs, _, _ = decoder_lstm(
                    self.decoder_outputs, initial_state=initial_state)
            self.decoder_layers.append(decoder_lstm)
        
        self.decoder_dense = Dense(
                self.decoder_vocab_size, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.model = Model(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

    def __preprocess_data(self, input_texts, target_texts, reversed_input=False):
        preprocessed_inputs = []
        for input_text in input_texts:
            if reversed_input:
                reversed_input_text = " ".join(input_text.split()[::-1])
                preprocessed_inputs.append(preprocess_sentence(reversed_input_text))
            else:
                preprocessed_inputs.append(preprocess_sentence(input_text))
        
        preprocessed_targets = []
        for target_text in target_texts:
            preprocessed_targets.append(preprocess_sparql(target_text))

        return preprocessed_inputs, preprocessed_targets

    def __prepare_data(self, input_texts, target_texts):
        num_of_samples = len(input_texts)

        # Data declaration
        self._max_encoder_seq_length = max(
            [len(text.split()) for text in input_texts])
        self._max_decoder_seq_length = max(
            [len(text.split()) for text in target_texts])

        print("Number of samples:", num_of_samples)
        print("Number of unique input words:", self.encoder_vocab_size)
        print("Number of unique output words:", self.decoder_vocab_size)
        print("Max sequenc length for inputs:", self._max_encoder_seq_length)
        print("Max sequenc length for outputs:", self._max_decoder_seq_length)

        self.input_word_index = dict(
            [(word, i) for i, word in enumerate(self.input_vocabulary)])
        self.target_word_index = dict(
            [(word, i) for i, word in enumerate(self.output_vocabulary)])
        self.reverse_input_word_index = dict(
            (i, char) for char, i in self.input_word_index.items())
        self.reverse_target_word_index = dict(
            (i, char) for char, i in self.target_word_index.items())

        # Setting up encoder input data
        encoder_input_data = np.zeros(
            (num_of_samples, self._max_encoder_seq_length, self.encoder_embedding_size), dtype='float32')

        # len(input_texts) == len(target_texts) because they exist as feature-label pairs
        decoder_input_data = np.zeros(
            (num_of_samples, self._max_decoder_seq_length), dtype='float32')

        decoder_target_data = np.zeros(
            (num_of_samples, self._max_decoder_seq_length, self.decoder_vocab_size), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            # Feed word embeddings into encoder as input
            for t, word in enumerate(input_text.split()):
                word_vector = self.input_word_vector.get_word_vector(
                    word.lower())
                for j in range(self.encoder_embedding_size):
                    encoder_input_data[i, t, j] = word_vector[j]
            # Feed word indexes into decoder as input,
            # one-hot vectors as decoder target
            for t, word in enumerate(target_text.split()):
                decoder_input_data[i, t] = self.target_word_index[word]
                if t > 0:
                    decoder_target_data[i, t-1,
                                        self.target_word_index[word]] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

    def train(self, input_texts, target_texts, early_stopping=False, reversed_input=False, epochs=100, batch_size=64):

        self.epochs = epochs
        self.batch_size = batch_size
        input_texts, target_texts = self.__preprocess_data(input_texts, target_texts, reversed_input=reversed_input)
        self.__build_vocabulary(input_texts, target_texts)
        self.__build_model()
        encoder_input_data, decoder_input_data, decoder_target_data = self.__prepare_data(
            input_texts, target_texts)

        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=3))
        my_optimizer = RMSprop(lr=0.001)
        self.model.compile(optimizer=my_optimizer,
                           loss=keras.losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
        self.history = self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                      batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2, shuffle=True, callbacks=callbacks)

        return self.history

    def save(self, file_location):
        if self.model is not None:
            self.model.save(file_location)

    def plot_model(self, img_location):
        plot_model(self.model, to_file=img_location)

    def load_model(self, model_location):
        self.model = keras.models.load_model(model_location)

    def inference(self, input_sequence):

        input_sequence = preprocess_sentence(input_sequence)
        input_sequence_parts = input_sequence.split()
        _encoder_input_data = np.zeros((1, len(input_sequence_parts), self.encoder_embedding_size), dtype='float32')
        for t, word in enumerate(input_sequence_parts):
            _word_vector = self.input_word_vector.get_word_vector(word)
            for j in range(self.encoder_embedding_size):
                _encoder_input_data[0, t, j] = _word_vector[j]

        encoder_model = Model(self.encoder_inputs, self.encoder_states_layered)
        decoder_states_inputs_layered = []
        for _ in range(self.num_layers):
            decoder_states_inputs_layered.extend([Input(shape=(self.hidden_units,)), Input(shape=(self.hidden_units,))])
        decoder_states_layered = []

        for i in range(self.num_layers):

            decoder_lstm = self.decoder_layers[i]
            initial_state = [decoder_states_inputs_layered[2*i], decoder_states_inputs_layered[2*i+1]]
            if i == 0:
                decoder_outputs, state_hidden, state_cell = decoder_lstm(self.decoder_embedding, initial_state=initial_state)
            else:
                decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_outputs, initial_state=initial_state)
            decoder_states = [state_hidden, state_cell]
            decoder_states_layered.extend(decoder_states)
            
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs_layered,
            [decoder_outputs] + decoder_states_layered
        )

        states_value_layered = encoder_model.predict(_encoder_input_data)
        target_seq = np.zeros((1,))
        target_seq[0] = self.target_word_index[self.sos_symbol]

        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            decoder_output, *decoder_state_layered = decoder_model.predict(
                [target_seq] + states_value_layered)

            sampled_word_index = np.argmax(decoder_output[0, -1, :])
            sampled_word = self.reverse_target_word_index[sampled_word_index]

            if sampled_word == self.eos_symbol or len(decoded_sentence) > self._max_decoder_seq_length:
                stop_condition = True
            else:
                decoded_sentence.append(sampled_word)

            target_seq = np.zeros((1,))
            target_seq[0] = sampled_word_index

            states_value_layered = decoder_state_layered

        return decoded_sentence
