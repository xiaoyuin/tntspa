import utils
from model import MyModel
# from fasttext_vector import FastTextVector
from gensim_vector import GensimFastTextVector
import time

input_texts = utils.read_lines("data/monument_600/train.en")
target_texts = utils.read_lines("data/monument_600/train.sparql")

start_time = time.time()
print("fastText model loading...")
# fasttext_model = FastTextVector("data/wiki.en.bin")
fasttext_model = GensimFastTextVector("data/wiki.en.bin")
e = int(time.time() - start_time)
print("fastText model loaded! Elapsed time: ", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

model = MyModel(input_word_vector=fasttext_model, num_layers=2)
model.train(input_texts, target_texts, epochs=100, early_stopping=True)

model.save('output/my_model.h5')