import numpy as np
import matplotlib.pyplot as plt


class DocumentSimilarity:
    """
    Given a document embedding, provides tools for analysis of document similarity.

    Args:
        __embedding (numpy.ndarray): matrix of document embeddings
        __embedding2D (numpy.ndarray): matrix of document embeddings with the dimension reduced to 2.

    Methods:
        get_embedding(): Retrieves the embedding from attribute '__embedding'.
        get_embedding2D(): Retrieves the embedding from attribute '__embedding2D'.
        euclid_similarity(emb1, emb2): Calculate the Euclid similarity between two embeddings.
        k_nearest_neighbors(index, k=10, similarity=self.euclid_similarity): Get the k documents with embeddings nearest
            to the document embedding we chose.
        reduce_dimension(dimension=2): Reduces the dimension of the embedding to 'dimension' using t-SNE algorithm.
    """

    def __init__(self, embedding):
        """
        Args:
            embedding(numpy.ndarray): matrix of document embeddings
        """

        self.__embedding = embedding
        self.__embedding2D = None

    def get_embedding(self):
        """
        Retrieves the embedding.

        Returns:
            obj: The embedding stored in the instance
        """

        return self.__embedding

    def get_embedding2d(self):
        """
        Retrieves the embedding with reduced dimension.

        Returns:
            obj: The 2-dimensional embedding stored in the instance
        """

        return self.__embedding2D

    def euclid_similarity(self, emb1, emb2):
        """
        Calculate the Euclid similarity between two embeddings.

        Args:
            emb1 (numpy.ndarray): the first embedding we want to compare
            emb2 (numpy.ndarray): the second embedding we want to compare

        Returns:
            numpy.float32: Euclid distance between embeddings of documents with given indices
        """

        return np.linalg.norm(emb1 - emb2)

    def k_nearest_neighbors(self, emb, k=10, similarity=None):
        """
        Get the k documents with embeddings nearest to the document embedding we chose.

        Args:
            emb (numpy.ndarray): embedding of the chosen document, whose neighbors we are trying to find
            k (int): number of neighbors we want to find. (Default = 10)
            similarity (function): metric function that we want to use for computing the distance between documents
                (Default = self.euclid_similarity)

        Returns:
            list: a list of k indices of the k documents whose embeddings are closest to the chosen embedding in the
            specified metric
        """

        # Set 'similarity' to self.euclid_similarity if nothing is set.
        if similarity is None:
            similarity = self.euclid_similarity

        # calculate the similarities and revert it
        sims = [similarity(emb, d) for d in self.__embedding]

        # sort and get the corresponding indices
        indices = []
        for c, i in enumerate(np.argsort(sims)):
            if c == k:
                break
            indices.append(i)

        # return indices of the neighbors
        return indices

    def reduce_dimension(self, dimension=2, update=False):
        """
        Reduces the dimension of the embedding to 'dimension'. If 'update' is set to True and 'dimension' is set to 2
        it updates the value of the attribute '__embedding2D'.

        Args:
            dimension (int): the dimension we want to reduce the embedding to. (Default = 2)
            update (bool): indicating if we want to update attribute '__embedding2D' or not. Only performs the update
                when the parameter 'dimension' is set to 2. (Default = False)

        Returns:
            numpy.ndarray: embedding matrix with reduced dimension.
        """

        embedding = TSNE(n_components=dimension).fit_transform(self.__embedding)
        if update and dimension == 2:
            self.__embedding2D = embedding
        elif update:
            raise Exception('Cannot update the parameter "__embedding2D" for "dimension" other than 2.')
        return embedding

    def plot(self, indices, color='b'):
        """
        Given a list 'indices', the method plots the embeddings of documents with indices in 'indices'. It also
        reduces the dimension of the embedding and saves it in the parameter '__embedding2D'.

        Args:
            indices (list(int)): list of indices of the documents we want to plot.
            color (string): string describing the color in which to plot the embeddings (see
            `list of named colors in matplotlib library<https://matplotlib.org/3.1.0/gallery/color/named_colors.html>`_). (Default = 'b')

        """

        # Reduce the dimension to 2 if that has not been done yet.
        if self.__embedding2D is None:
            self.__embedding2D = self.reduce_dimension()

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=160, facecolor='w')

        # Plot embeddings
        temp = self.__embedding2D[indices,:]
        x, y = temp.transpose()[0], temp.transpose()[1]
        ax.scatter(x, y, 2, marker='x', c='b')

        plt.axis('off')
        plt.show()
        return None

