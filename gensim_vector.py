from gensim.models import KeyedVectors, FastText
from model import MyVector
import time

class GensimKeyedVector(MyVector):

    def __init__(self, file_path):
        start_time = time.time()
        print("Gensim KeyedVector model loading...")
        self.vectors = KeyedVectors.load_word2vec_format(file_path)
        self.vocabulary = self.vectors.vocab
        e = int(time.time() - start_time)
        print("Gensim KeyedVector model loaded! Elapsed time: ", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

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
        start_time = time.time()
        print("Gensim FastText model loading...")
        self.model = FastText.load_fasttext_format(file_path)
        self.vocabulary = self.model.wv.index2word
        e = int(time.time() - start_time)
        print("Gensim FastText model loaded! Elapsed time: ", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

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