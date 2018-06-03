
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""zadanie6.

Module implementing following functionalities:

    - Extension for CountVectorizer using stemming or lemmatizing (using nltk
    library)
    - TextMultinomialNaiveBayes - Naive Bayes for Text classification,
    implementation focused on sparse data
    - MultinomialNaiveBayes - Naive Bayes for other tasks, implementation
    focused on sparse data

"""

from timeit import default_timer as timer

import nltk
import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer

###############################################################################
#                               VECTORIZERS
###############################################################################


class StemmingCountVectorizer(CountVectorizer):
    """StemmingCountVectorizer.

    Extension of sklearn's CountVectorizer (uses additional stemming)
    All functionalities of normal CountVectorizer are otherwise the
    same

    """

    def __init__(self, *, stemmer=nltk.stem.PorterStemmer(), **kwargs):
        """__init__.

        :param stemmer: Stemming object used for word stemming (default:
        nltk.stem.PorterStemmer())
        :param **kwargs: keyword arguments for base sklearn's CountVectorizer

        """
        super().__init__(**kwargs)
        self._stemmer = stemmer

    def build_analyzer(self):
        """build_analyzer.

        Overloaded function for build_analyzer (additionally stems the
        words)

        """

        analyzer = super().build_analyzer()
        return lambda doc: ([self._stemmer.stem(word)
                             for word in analyzer(doc)])


class LemmatizingCountVectorizer(CountVectorizer):
    """LemmatizingCountVectorizer.

    Extension of sklearn's CountVectorizer (uses additional
    lemmatizing) All functionalities of normal CountVectorizer are
    otherwise the same.

    """

    def __init__(self,
                 *,
                 lemmatizer=nltk.stem.wordnet.WordNetLemmatizer(),
                 **kwargs):
        """__init__"""
        """__init__

        :param lemmatizer: Lemmatizing object used for word lemmatizing
        (default: nltk.stem.wordnet.WordNetLemmatizer())
        :param **kwargs: keyword arguments for base sklearn's CountVectorizer

        """
        super().__init__(**kwargs)
        self._lemmatizer = lemmatizer

    def build_analyzer(self):
        """build_analyzer.

        Overloaded function for build_analyzer (additionally lemmatizes
        the words)


        """
        analyzer = super().build_analyzer()
        return lambda doc: ([self._lemmatizer.lemmatize(word)
                             for word in analyzer(doc)])



###############################################################################
#           FUNCTION TRANSFORMING LOGPOSTERIORS TO LOGITS
###############################################################################


def unnormalized_logposteriors_to_logits(ulp):
    """unnormalized_logposteriors_to_logits.
    Normalizes logposteriors and transforms them into logits

    See https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
    for informations

    :param ulp: numpy array containing unnormalized log posterios (matrix type)
    :returns: numpy array of logits of shape ulp.shape[0], 1

    """
    assert len(ulp.shape) == 2
    ulp -= np.max(ulp, axis=1).reshape(-1, 1)
    ulp -= np.log(np.sum(np.exp(ulp), axis=1)).reshape(-1, 1)
    return ulp


###############################################################################
#           NAIVE BAYES IMPLEMENTATIONS FOCUSED ON SPARSE MATRICES
###############################################################################


class SparseNaiveBayesBase:
    """SparseNaiveBayesBase.
    Base class for other Naive Bayes implementations, simplifying logic.
    Should not be touched or modified, only derived
    """

    def __init__(self, X, y, *, alpha):
        """__init__.

        Fits the model to the training data and displays efficienct measurements
        for calculating prior and likelihoods

        :param X: array-like matrix with sparse features data (e.g. BOW)
        :param y: array of shape (X.shape[0], ) containing labels
        :param *: placeholder to enforce kwargs passing for alpha
        :param alpha: Smoothing parameter (default: 1, Laplace smoothing)
        """

        #######################################################################
        # CALCULATE PRIOR
        #######################################################################

        self.alpha = alpha

        start = timer()

        unique, self.priors = np.unique(y, return_counts=True)
        self.priors = self.priors / X.shape[0]
        self.log_priors = np.log(self.priors)

        end = timer()
        time_prior = end - start
        print(
            "Time taken to calculate prior in seconds: {}".format(time_prior))

        #######################################################################
        # CALCULATE LIKELIHOODS, LOG LIKELIHOODS ETC.
        #######################################################################

        start = timer()

        class_indices = np.array(
            np.ma.make_mask([y == current for current in unique]))
        class_datasets = np.array([X[indices] for indices in class_indices])

        classes_metrics = np.array([
            dataset.sum(axis=0) for dataset in class_datasets
        ]).reshape(len(unique), -1) + self.alpha

        self.likelihoods = classes_metrics / \
            classes_metrics.sum(axis=1)[:, np.newaxis]
        self.log_likelihoods = np.log(self.likelihoods)

        end = timer()
        time_likelihood = end - start
        print("Time taken to calculate likelihoods in seconds: {}".format(
            time_likelihood))
        print("Time for training overally: {}".format(
            time_likelihood + time_prior))

    def predict_logits(self, X):
        """predict_logits

        Predicts logits for given dataset X

        :param X: array-like matrix with sparse features data (e.g. BOW)
        :returns: array-like containing logits with observations probability for
        each class, shaped (X.shape[0], classes_count)

        """
        logposteriors = scipy.sparse.csr_matrix.dot(
            X, self.log_likelihoods.T) + self.log_priors
        return unnormalized_logposteriors_to_logits(logposteriors)

    def predict(self, X):
        """predict

        Predicts the most probable label for X

        :param X: array-like matrix with sparse features data (e.g. BOW)
        :returns: array-like list of predicted classes, shaped (X.shape[0], )
        """
        return np.argmax(self.predict_logits(X), axis=1)


class TextMultinomialNaiveBayes(SparseNaiveBayesBase):
    """TextMultinomialNaiveBayes.

    Concrete class implementing Multinomial Naive Bayes for text classification
    """

    def __init__(self,
                 X,
                 y,
                 *,
                 alpha=1,
                 vectorizer=CountVectorizer(
                     min_df=5, stop_words='english', decode_error='ignore')):
        """__init__.

        Trains TextMultinomialBayes w.r.t. given data
        (document list, see :param X:).

        For more information, see base class SparseNaiveBayesBase

        :param X: list containing text documents
        :param y: array of shape (X.shape[0], ) containing labels for each
        document
        :param *: placeholder to enforce kwargs passing for alpha
        :param alpha: Smoothing parameter (default: 1, Laplace smoothing)
        :param vectorizer: Vectorizing object used for transforming X into it's
        discrete counterparts (default: sklearn's CountVectorizer(min_df=5,
        stop_words='english', decode_error='ignore'))

        """

        self.vectorizer = vectorizer

        #######################################################################
        # VECTORIZE FEATURES
        #######################################################################
        start = timer()

        X_train = vectorizer.fit_transform(X)

        end = timer()
        print("Time taken to vectorize text in seconds using vectorizer: \
              \n\n{}: \n\n{}\n".format(str(vectorizer), end - start))

        # DELEGATE REST OF THE WORK TO BASE CLASS
        super().__init__(X_train, y, alpha=alpha)

    @property
    def features(self):
        """features.

        Strings learned by @vectorizer in the exactly same order

        :returns: list of strings, words learned by @vectorizer (see __init__)
        as a list and in the same order
        """
        return np.array(list(self.vectorizer.vocabulary_.keys()))

    # easter egg: jego najlepszy fragment zdania do tej pory to Allah unethical kindness
    # dla ateizmu, w sumie z dwoma pierwszymi słowami ciężko się nie zgodzić...
    def generate_sentence(self, nb_class, length):
        """generate_sentence.

        Generates sentence for given class of given length (may return shorter,
        though highly unlikely for big enough document and relative small length
        to document's length)

        :param nb_class: [int] Class for which we want to generate sentence
        :param length: [int] Length of the sentence
        """
        indices = np.random.multinomial(length,
                                        self.likelihoods[nb_class]).nonzero()
        return " ".join(self.features[indices])

    def predict_logits(self, X):
        """predict_logits

        Predicts logits for given dataset X

        :param X: list containing text documents
        :returns: array-like containing logits with observations probability for
        each class, shaped (X.shape[0], classes_count)

        """
        X_predict = self.vectorizer.transform(X)
        return super().predict_logits(X_predict)


class MultinomialNaiveBayes:
    """MultinomialNaiveBayes"""

    def __init__(self, X, y):
        """__init__

        Trains MultinomialNaiveBayes w.r.t. given data. If you have list of
        documents, you should use TextBernoulliNaiveBayes instead, or vectorize
        them beforehand.

        :param X: array-like matrix of features and examples (base of training)
        :param y: array-like vector of corresponding labels
        """
        super().__init__(X, y, alpha=0)

    def generate(self, nb_class, length):
        """generate

        Generates array of most likely features as a list of indices

        :param nb_class: [int] Class for which we want to generate sentence
        :param length: [int] Length of the sentence
        :returns: np.array list of indices into generated features
        """
        return np.random.multinomial(length,
                                     self.likelihoods[nb_class]).nonzero()


###############################################################################
#               COMPATIBILITY CLASSES FOR ASSIGNMENT
###############################################################################


class TextBernoulliNaiveBayes:
    """TextBernoulliNaiveBayes
    Empty compatibility class for notebook assignment
    """
    pass


class BernoulliNaiveBayes:
    """BernoulliNaiveBayes
    Empty compatibility class for notebook assignment
    """
    pass
