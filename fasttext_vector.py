
import fastText
from model import MyVector
import time

class FastTextVector(MyVector):

    def __init__(self, file_path):
        # Use fastText to load fastText vector models
        start_time = time.time()
        print("fastText model loading...")
        self.vectors = fastText.load_model(file_path)
        self.vocabualry = self.vectors.get_words()
        e = int(time.time() - start_time)
        print("fastText model loaded! Elapsed time: ", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))


    def get_vocabulary(self):
        return self.vocabualry

    def get_vocabulary_size(self):
        return len(self.vectors.get_words())

    def get_dimension(self):
        return self.vectors.get_dimension()

    def get_word_vector(self, word):
        return self.vectors.get_word_vector(word)

    def get_word_id(self, word):
        return self.vectors.get_word_id(word)