import numpy as np
from gensim.models import KeyedVectors, FastText


class WordModels:
    """
    Class to load, train or save word embeddings.

    Attributes:
        __embedding (gensim.models.keyedvectors.*): word embedding
        __model (type): word embedding model

    Methods:
        get_embedding(): Retrieves the embedding from the attribute '__embedding'.
        get_model(): Retrieves the embedding from the attribute '__model'.
        load(path, model_type='word2vec'): loads word embedding model saved at 'path' and updates parameters
            '__embedding' and '__model'. Argument 'model_type' describes which library we want to use.
        train(documents, size=300, window=3, min_count=4, epochs=50): trains word embedding model on corpus 'documents'
            using fastText model (see `documentation for FastText <https://radimrehurek.com/gensim/models/fasttext.html#module-gensim.models.fasttext>`_).
        save_to_file(file_name): saves the model to a file with path 'file_name'.
    """

    def __init__(self):
        """
        Takes no arguments.
        """
        self.__embedding = None
        self.__model = None

    def get_embedding(self):
        """
        Retrieves the embedding.

        Returns:
            obj: The embedding stored in the instance
        """

        return self.__embedding

    def get_model(self):
        """
        Retrieves the model.

        Returns:
            obj: The model stored in the instance
        """

        return self.__model

    def load(self, path, model_type='word2vec'):
        """
        Load pre-trained word embedding model and save it into embed_words.__embedding.

        Args:
            path (str): relative path to the file containing pre-trained model.
            model_type (str): type of the model - must be one of the following:'word2vec' or 'fasttext'.
                (Default = 'word2vec')
                What sets these two models apart is what they consider to be an atomic embedding element: word2vec
                considers a word to be the smallest part of language to embed, while fasttext uses character n-grams as
                well. Because of this we can extract embeddings for out-of-vocabulary terms, providing embeddings of
                rare and previously unseen words.
        """

        # Code for loading Word2vec model:
        if model_type == 'word2vec':
            self.__model = KeyedVectors.load_word2vec_format(path)
            self.__embedding = self.__model.wv

        # Code for loading fastText model:
        elif model_type == 'fasttext':
            self.__model = FastText.load_fasttext_format(path)
            self.__embedding = self.__model.wv

        # In case we're trying to load an unsupported model type:
        else:
            raise Exception("Model '{}' not supported (must be 'word2vec' or 'fasttext').".format(model_type) +
                            " Cannot load word embedding model.")

    def train(self, documents, size=300, window=3, min_count=4, epochs=50):
        """
        Train a fastText word embedding model (see `documentation for fastText <https://radimrehurek.com/gensim/models/fasttext.html#module-gensim.models.fasttext>`_.) on the sentences provided
        as an argument.

        Args:
            documents (list(str)): list of documents represented as a stripped lowercase string
            size (int): dimension of the embedding space. (Default = 300)
            window (int): The maximum distance between the current and predicted word within a sentence. (Default = 3)
            min_count (int): The model ignores all words with total frequency lower than min_count. (Default = 4)
            epochs (int): Number of iterations over the corpus. (Default = 50)

        Returns:
            bool: True, if the training was successful, False if it was not.
        """

        # Code for training of fastText models:
        # if model_type == 'fasttext':
        try:
            tm = FastText(size=size, window=window, min_count=min_count)
            tm.build_vocab(sentences=documents)
            tm.train(sentences=documents, total_examples=len(documents), epochs=epochs)

            # # Code for training of Word2vec models:
            # elif model_type == 'word2vec':
            #     tm = KeyedVectors(documents, size=size, window=window, min_count = min_count, iter=epochs)

            # # In case we're trying to train an unsupported model type:
            # else:
            #     raise Exception("Model '{}' not supported (must be 'word2vec' or 'fasttext').".format(model_type))

            # Assign values for self.__embedding and self.__model:
            self.__model = tm
            self.__embedding = tm.wv
            return True
        except:
            return False

    def save_to_file(self, file_name):
        """
        Save the (pre-trained) word embedding model in a file.

        Args:
            file_name (str): relative path to the file, in which we want to save the model.

        Returns:
            bool: True if the saving process ended successfully, False otherwise.
        """

        try:
            self.__model.save(file_name)
            return True
        except OSError:
            return False
