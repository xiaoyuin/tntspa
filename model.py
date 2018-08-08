import numpy as np
import h5py
from matplotlib import pyplot

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.optimizers import RMSprop
from keras.initializers import Orthogonal
from keras.callbacks import EarlyStopping
from keras import metrics

import fastText
from preprocessing import preprocess_sentence
from gensim.models import KeyedVectors
import utils


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


class FastTextVector(MyVector):

    def __init__(self, file_path):
        # Use fastText to load fastText vector models
        self.vectors = fastText.load_model(file_path)

    def get_vocabulary(self):
        return self.vectors.get_words()

    def get_vocabulary_size(self):
        return len(self.vectors.get_words())

    def get_dimension(self):
        return self.vectors.get_dimension()

    def get_word_vector(self, word):
        return self.vectors.get_word_vector(word)

    def get_word_id(self, word):
        return self.vectors.get_word_id(word)


class GensimKeyedVector(MyVector):

    def __init__(self, file_path):
        # Use fastText to load fastText vector models
        self.vectors = KeyedVectors.load(file_path)

    def get_vocabulary(self):
        return self.vectors.get_words()

    def get_vocabulary_size(self):
        return len(self.vectors.get_words())

    def get_dimension(self):
        return self.vectors.get_dimension()

    def get_word_vector(self, word):
        return self.vectors.get_vector(word)

    def get_word_id(self, word):
        return self.vectors.get_word_id(word)


class MyModel:
    """MyModel"""

    # Default parameters
    batch_size = 64
    epochs = 100
    dropout = 0.2
    hidden_units = 128
    encoder_embedding_size = 300
    decoder_embedding_size = 300

    sos_symbol = '<s>'
    eos_symbol = '</s>'
    unk_symbol = '<unk>'

    def __init__(self, input_word_vector, target_word_vector=None):
        """Creates a model"""
        self.input_word_vector = input_word_vector
        self.target_word_vector = target_word_vector

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

        # Encoder
        self.encoder_inputs = Input(shape=(None, self.encoder_embedding_size))
        # encoder_embedding = Embedding(num_input_words+1, embedding_size, weights=[embedding_matrix], trainable=False)(encoder_inputs)
        self.encoder = LSTM(self.hidden_units, return_state=True, dropout=self.dropout, recurrent_dropout=self.dropout,
                            kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
        _, self.state_hidden, self.state_cell = self.encoder(
            self.encoder_inputs)
        self.encoder_states = [self.state_hidden, self.state_cell]

        # Decoder
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_embedding = Embedding(
            self.decoder_vocab_size, self.decoder_embedding_size)(self.decoder_inputs)
        self.decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True, dropout=self.dropout,
                                 recurrent_dropout=self.dropout, kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
        self.decoder_outputs, _, _ = self.decoder_lstm(
            self.decoder_embedding, initial_state=self.encoder_states)
        self.decoder_dense = Dense(
            self.decoder_vocab_size, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.model = Model(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

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

    def train(self, input_texts, target_texts, early_stopping=False):

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

    def visualize(self):

        pyplot.plot(self.history.history['loss'])
        pyplot.plot(self.history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()

    def save(self, file_location):
        if self.model is not None:
            self.model.save(file_location)

    def load_model(self, model_location):
        self.model = keras.models.load_model(model_location)

    def inference(self, input_sequence):

        encoder_model = Model(self.encoder_inputs, self.encoder_states)
        decoder_state_input_hidden = Input(shape=(self.hidden_units,))
        decoder_state_input_cell = Input(shape=(self.hidden_units,))
        decoder_states_inputs = [
            decoder_state_input_hidden, decoder_state_input_cell]

        decoder_outputs, state_hidden, state_cell = self.decoder_lstm(
            self.decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_hidden, state_cell]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        states_value = encoder_model.predict(input_sequence)
        target_seq = np.zeros((1,))
        target_seq[0] = self.target_word_index[self.sos_symbol]

        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            decoder_output, h, c = decoder_model.predict(
                [target_seq] + states_value)

            sampled_word_index = np.argmax(decoder_output[0, -1, :])
            sampled_word = self.reverse_target_word_index[sampled_word_index]

            if sampled_word == self.eos_symbol or len(decoded_sentence) > self._max_decoder_seq_length:
                stop_condition = True
            else:
                decoded_sentence.append(sampled_word)

            target_seq = np.zeros((1,))
            target_seq[0] = sampled_word_index

            states_value = [h, c]

        return decoded_sentence


input_texts = utils.read_lines("data/monument_600/train.en")
target_texts = utils.read_lines("data/monument_600/train.sparql")

model = MyModel(input_word_vector=FastTextVector("data/wiki.en.bin"))
model.train(input_texts, target_texts)
model.visualize()
model.save('output/my_model.h5')
print(model.inference("I have a dream"))
