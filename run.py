
# coding: utf-8

# # Necessary imports

# In[1]:


import keras
import numpy as np
import random
from preprocessing import load_vectors, preprocess_sentence, preprocess_sparql
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.initializers import Orthogonal
import fastText


# # Read English and SPARQL files
# 
# - input English sentences
# - pre-trained English word vectors
# - target SPARQL queries
# 

# In[2]:


# Files Reading
file_input_en = "data/qald-7-train-largescale.en"
file_input_en_vectors = "data/wiki.en.bin"
file_target_sparql = "data/qald-7-train-largescale.sparql"

sos_symbol = '<s>'
eos_symbol = '</s>'


# # Preprocess the English sentences and load the vectors for both input and output(target)
# - deal with word boundaries
# - deal with starting and ending symbols (*unsettled*)

# In[3]:


input_texts = []
with open(file_input_en) as file:
    for line in file:
        input_texts.append(preprocess_sentence(line))
    
target_texts = []
with open(file_target_sparql) as file:
    for line in file:
        target_texts.append(preprocess_sparql(line))

input_vectors = fastText.load_model(file_input_en_vectors) # Use fastText to load fastText vector models



# In[4]:


input_vocabulary = input_vectors.get_words()
output_vocabulary = set()

for target_text in target_texts:
    output_vocabulary.update(target_text.split())

encoder_vocab_size = len(input_vocabulary)
decoder_vocab_size = len(output_vocabulary)


# # Declare Parameters for the model

# In[5]:


# Parameters
batch_size = 1
epochs = 100
dropout = 0.2
num_samples = len(input_texts)
encoder_embedding_size = input_vectors.get_dimension()
decoder_embedding_size = 300
hidden_units = 128


# # Define the model

# In[6]:


# Model definition
# Encoder
encoder_inputs = Input(shape=(None, encoder_embedding_size))
# encoder_embedding = Embedding(num_input_words+1, embedding_size, weights=[embedding_matrix], trainable=False)(encoder_inputs)
encoder = LSTM(hidden_units, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
_, state_hidden, state_cell = encoder(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(decoder_vocab_size, decoder_embedding_size)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=Orthogonal(), recurrent_regularizer=keras.regularizers.l2())
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(decoder_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# # Initializing input data and target data

# In[7]:



# Data declaration
max_encoder_seq_length = max([len(text.split()) for text in input_texts])
max_decoder_seq_length = max([len(text.split()) for text in target_texts])

print("Number of samples:", num_samples)
print("Number of unique input words:", encoder_vocab_size)
print("Number of unique output words:", decoder_vocab_size)
print("Max sequenc length for inputs:", max_encoder_seq_length)
print("Max sequenc length for outputs:", max_decoder_seq_length)

input_word_index = dict([(word, i) for i, word in enumerate(input_vocabulary)])
target_word_index = dict([(word, i) for i, word in enumerate(output_vocabulary)])

# Setting up encoder input data 
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, encoder_embedding_size), dtype='float32')

# len(input_texts) == len(target_texts) because they exist as feature-label pairs
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='float32')

decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, decoder_vocab_size), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # Feed word embeddings into encoder as input
    for t, word in enumerate(input_text.split()):
        word_vector = input_vectors.get_word_vector(word.lower())
        for j in range(encoder_embedding_size):
            encoder_input_data[i, t, j] = word_vector[j]
    # Feed word indexes into decoder as input,
    # one-hot vectors as decoder target
    for t, word in enumerate(target_text.split()):
        decoder_input_data[i, t] = target_word_index[word]
        if t > 0:
            decoder_target_data[i, t-1, target_word_index[word]] = 1.


# # Configure the model and train the model

# tb_callback = keras.callbacks.TensorBoard()
# 
# 
# - configure the model with optimizer and loss function

# In[8]:


my_optimizer = RMSprop(lr=0.001)

model.compile(optimizer=my_optimizer, loss=keras.losses.categorical_crossentropy)


# - train the model:

# In[9]:



history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 
    batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True
)



# # Visualize the training loss and save the model
# 
# - plot the training loss and validation loss along with epochs

# In[ ]:


from matplotlib import pyplot

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# - plot the model into an image
# - save the model

# In[ ]:


from keras.utils import plot_model

plot_model(model, to_file='output/model.png')

model.save('output/seq2seq.h5')


# # Inference mode
# Input some sentence into the encoder, and decode the output sequence

# In[10]:


# Inference

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_hidden = Input(shape=(hidden_units,))
decoder_state_input_cell = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

reverse_input_word_index = dict((i, char) for char, i in input_word_index.items())
reverse_target_word_index = dict((i, char) for char, i in target_word_index.items())


# In[11]:


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,))
    target_seq[0] = target_word_index[sos_symbol]

    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        decoder_output, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_word_index = np.argmax(decoder_output[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_word_index]

        if sampled_word == eos_symbol or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1,))
        target_seq[0] = sampled_word_index

        states_value = [h, c]

    return decoded_sentence


# In[12]:



# Try out decoding sentences from training set
# because we train on training set, the result should be good
for seq_index in random.sample(range(num_samples), 10):
    input_seq = encoder_input_data[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-', str(seq_index)+'th:' )
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', *decoded_sentence)
    print('Decoded sentence length:', len(decoded_sentence))
    print('Target sentence:', target_texts[seq_index])
    print('Target sentence length:', len(target_texts[seq_index].split()))

