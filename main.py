import utils
from model import MyModel
from fasttext_vector import FastTextVector


input_texts = utils.read_lines("data/monument_600/train.en")
target_texts = utils.read_lines("data/monument_600/train.sparql")

model = MyModel(input_word_vector=FastTextVector("data/wiki.en.bin"), num_layers=2)
model.train(input_texts, target_texts)
model.visualize()
model.save('output/my_model.h5')
print(model.inference("I have a dream"))