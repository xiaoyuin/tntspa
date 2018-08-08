
import fastText
from model import MyVector


class FastTextVector(MyVector):

    def __init__(self, file_path):
        # Use fastText to load fastText vector models
        self.vectors = fastText.load_model(file_path)
        self.vocabualry = self.vectors.get_words()

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