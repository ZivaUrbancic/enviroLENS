import numpy as np
import operator
import string
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation
from nltk.corpus import stopwords as sw
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def customized_strip(s):
    """
    Static function that strips a given text of most unwanted characters.

    Args:
        s (string): text we want to strip

    Returns:
        string: stripped text
    """

    # strip the comments
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.replace('“', '')
    s = s.replace('”', '')

    s = re.sub('https?:\/\/[^\s]+', ' ', s)  # strip urls
    s = re.sub('[\d]+', ' ', s)              # strip numbers

    # strip whitespace
    s = s.replace("\r", ' ').replace("\xa0", ' ')
    s = re.sub('[\s]+', ' ', s)

    s = strip_punctuation(s)
    s = s.lower()
    return s


class DocumentModels:
    """
    A class to represent documents as document embeddings.

    Attributes:
        __word_vectors (gensim.models.keyedvectors.*): word embedding we want to use to construct document embeddings
        __documents (list(string)): list of documents we want to embed
        __stopwords (list(string)): list of words we want to remove from documents' texts.
        __embedding (numpy.ndarray): a matrix where i-th line is the document embedding of i-th document in the list
            '__documents'.
        __idf (dict{string: numpy.float64}): a dictionary of words and their idf coefficients in the corpus

    Methods:
        get_embedding(): Retrieves the embedding from attribute '__embedding'.
        get_idf(): Retrieves the dictionary of idf coefficients from attribute '__idf'.
        tokenize(text): Tokenizes and removes stopwords from the provided text.
        compute_idf(): Computes idf coefficients of words appearing in the corpus and saves them in '__idf'.
        average_word_embedding(text, use_tfidf_weighing): Creates a document embedding as the average of word embeddings
            of words that appear in given text.
        embed_list_of_documents(docs, use_tfidf_weighing): Returns a document embedding of documents given as an
            argument.
        embed_documents(use_tfidf_weighing): Creates the embedding for documents given at initialization and saves it in
            the attribute '__embedding'.
        add_documents(docs, use_tfidf_weighing): Appends given documents to the model's attribute '__documents' and adds
            lines for document embeddings of those documents at the end of the matrix '__embedding'.
        remove_documents(docs): Removes given documents from the model's attribute '__documents' and removes lines for
            their embeddings from the matrix '__embedding'.
    """

    def __init__(self, word_embedding, documents, stopwords=None):
        """
        Args:
             word_embedding (gensim.models.keyedvectors.*): word embedding we want to use to construct document
                embeddings
             documents (list(string)): list of documents we want to embed
             stopwords (list(string)): list of words we want to remove from documents' texts. (Default = customized list
                of english stopwords from module nltk)
        """

        self.__word_vectors = word_embedding
        self.__documents = documents

        # initialize stopwords
        if stopwords is None:
            self.__stopwords = sw.words('english') + list(string.punctuation)
        else:
            try:
                self.__stopwords = stopwords
            except ImportError:
                print("There was an error loading stopwords from module nltk.")

        self.__embedding = None
        self.__idf = None

    def get_embedding(self):
        """
        Retrieves the embedding.

        Returns:
            obj: The embedding stored in the instance
        """

        return self.__embedding

    def get_idf(self):
        """
        Retrieves the dictionary of idf coefficients.

        Returns:
            obj: The idf coefficients stored in the instance
        """

        return self.__idf

    def tokenize(self, text):
        """
        Tokenizes and removes stopwords from the provided text.

        Args:
            text (str): text that we want to tokenize.

        Returns:
            word_sorted (list(tuple(str,int))): A list of tuples ('word', n), where n is the number of times word
                appears in text. The list is sorted by occurrences decreasingly.
        """

        # Strip the text of comments, urls, numbers and newline
        stripped_text = customized_strip(text)

        # Strip punctuation and make everything lowercase
        # custom_filters = [lambda x: x.lower(), strip_punctuation]
        tokens = preprocess_string(stripped_text) #, custom_filters)

        # Filter through tokens and remove stopwords
        filtered = [w for w in tokens if not w in self.__stopwords]

        # get the most frequent words in the document
        count = {}
        for word in filtered:
            if word not in count:
                count[word] = 0
            count[word] += 1

        word_sorted = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        return word_sorted

    def compute_idf(self):
        """
        Computes idf coefficients of words appearing in the corpus and saves them in '__idf'.
        """

        # initialize TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform([customized_strip(d) for d in self.__documents])

        # make it into a dictionary
        self.__idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    def average_word_embedding(self, text, use_tfidf_weighing=True):
        """
        Creates a document embedding as an average of word embeddings of words that appear in given text.

        Args:
            text (str): text, from which we take words and compute their average embedding
            use_tfidf_weighing (bool): True, if we want to use weighed average of words with tfidf coefficients of words
                as weights. (Default = True)

        Returns:
            (numpy.ndarray): an average of word embeddings of words that appear in the given text.
        """

        embedding = np.zeros(self.__word_vectors.vector_size, dtype=np.float32)
        if text is None:
            return embedding

        word_sorted = self.tokenize(text)
        norm = 0
        for token, number_of_appearances in word_sorted:
            # sum all tokens embeddings of the vector
            if token in self.__word_vectors.vocab.keys():

                # Set the coefficient with which to multiply the embedding of the word
                coef = number_of_appearances
                if use_tfidf_weighing and token in self.__idf:
                    coef = coef * self.__idf[token]
                elif use_tfidf_weighing:
                    # Skip the word if it has no idf coef
                    coef = 0
                    number_of_appearances = 0

                embedding += coef * self.__word_vectors[token]
                norm += number_of_appearances

        # return the normalized embedding; if not zero
        return embedding if norm == 0 else embedding / norm

    def embed_list_of_documents(self, docs, use_tfidf_weighing=True):
        """
        Returns a document embedding of documents given as an argument.

        Args:
            docs (list(string)): list of documents we want to embed
            use_tfidf_weighing (bool): True, if we want to use weighed average of words with tfidf coefficients of words
                as weights. (Default = True)

        Returns:
             numpy.ndarray: matrix of document embeddings, where i-th line is the embedding of i-th document of the
                 list 'docs'.
        """

        embedding = np.zeros((len(docs), self.__word_vectors.vector_size), dtype=np.float32)
        # embed individual documents:
        for id, document in enumerate(docs):
            embedding[id,:] = self.average_word_embedding(document, use_tfidf_weighing)
        return embedding

    def embed_documents(self, use_tfidf_weighing=True):
        """
        Creates document embeddings for documents given at initialization and saves it in the attribute '__embedding'.

        Args:
            use_tfidf_weighing (bool): True, if we want to use weighed average of words with tfidf coefficients of words
                as weights. (Default = True)
        """

        self.compute_idf()
        self.__embedding = self.embed_list_of_documents(self.__documents, use_tfidf_weighing)

    def embed_with_common_component_removal(self, use_tfidf_weighing=True):
        """
        Creates document embeddings using the method of common component removal (subtracts the projection of the
        embedding on the first singular vector of the embedding matrix from every document embedding). Method returns
        the new embedding.

        Arg:
            use_tfidf_weighing (bool): True, if we want to use weighed average of words with tfidf coefficients of words
                as weights. (Default = True)
        """

        # Embed document, if it hasn't been done before.
        if self.__embedding is None:
            self.embed_documents(use_tfidf_weighing)

        # Initialize the new embedding:
        subtracted_embedding = np.zeros(self.__embedding.shape)

        # Compute the first singular vector
        u, s, vh = np.linalg.svd(self.__embedding)
        first_singular_vector = vh[0, :]

        # For every documents subtract the projection of its embedding on the first singular vector from the embedding
        for i in range(len(self.__documents)):
            temp = np.matmul(self.__embedding[i,:], first_singular_vector.transpose())
            subtracted_embedding[i, :] = self.__embedding[i, :] - temp*first_singular_vector

        return subtracted_embedding

    def add_documents(self, docs, use_tfidf_weighing=True):
        """
        Appends given documents to the model's attribute '__documents' and adds lines for document embeddings of those
        documents at the end of the matrix '__embedding'.

        Args:
            docs (list(str)): List of documents we want to add to our model
            use_tfidf_weighing (bool): True, if we want to use weighed average of words with tfidf coefficients of words
                as weights. (Default = True)
        """

        # From list docs remove the documents that we already have in the corpus.
        docs = [d for d in docs if d not in self.__documents]

        # Update the list of documents and recompute idf coefficients.
        self.__documents = self.__documents + docs
        self.compute_idf()

        # Embed new documents.
        new_embedding = self.embed_list_of_documents(docs, use_tfidf_weighing)
        self.__embedding = np.concatenate((self.__embedding, new_embedding))

    def remove_documents(self, docs):
        """
        Removes given documents from the model's attribute '__documents' and removes lines for their embeddings from the
        matrix '__embedding'.

        Args:
            docs (list(str)): List of documents we want to remove from our model

        Returns:
            int: number of documents removed.
        """

        number_removed = 0
        for doc in docs:
            if doc in self.__documents:
                # Figure out in which line the embedding of this document lies.
                index = self.__documents.index(doc)

                # Update the attributes.
                self.__documents.remove(doc)
                self.__embedding = np.concatenate((self.__embedding[0:index, :], self.__embedding[index+1:, :]))

                number_removed += 1
        self.compute_idf()
        return number_removed
