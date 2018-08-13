from gensim.models import KeyedVectors, FastText
from model import MyVector

class GensimKeyedVector(MyVector):

    def __init__(self, file_path):
        self.vectors = KeyedVectors.load_word2vec_format(file_path)
        self.vocabulary = self.vectors.vocab

    def get_vocabulary(self):
        return self.vectors.index2word

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def get_dimension(self):
        return self.vectors.vector_size

    def get_word_vector(self, word):
        return self.vectors.get_vector(word)

    def get_word_id(self, word):
        return self.vocabulary[word].index

class GensimFastTextVector(MyVector):

    def __init__(self, file_path):
        # Use fastText to load fastText vector models
        self.model = FastText.load_fasttext_format(file_path)
        self.vocabulary = self.model.wv.index2word

    def get_vocabulary(self):
        return self.vocabulary

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def get_dimension(self):
        return self.model.vector_size

    def get_word_vector(self, word):
        return self.model.wv.get_vector(word)

    def get_word_id(self, word):
        return self.model.wv.vocab[word].index