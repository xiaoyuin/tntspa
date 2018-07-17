from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

print(tf.__version__)
