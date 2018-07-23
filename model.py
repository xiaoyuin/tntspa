from tensorflow import keras
import tensorflow as tf
import numpy as np
from preprocessing import load_vectors, preprocess_sentence
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import fastText
