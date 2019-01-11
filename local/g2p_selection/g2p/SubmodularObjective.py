from __future__ import print_function
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Base method
class FeatureObjective(object):
    def __init__(self, wordlist, test_wordlist=None, n_order=4,
                append_ngrams=True, binarize_counts=False):
        self.wordlist = wordlist
        if not test_wordlist:
            self.test_wordlist = wordlist
        else:
            self.test_wordlist = test_wordlist

        self._ngram_vectorizer = None
        
        self.n_order = n_order
        self.append_ngrams = append_ngrams
        self.binarize_counts = binarize_counts
        
        self.vectorizer = CountVectorizer
        
        self.word_features = None
        self.subset = None
        self.p = None
        
        self.get_vectorizer()
        self.get_word_features()
    
    
    def get_vectorizer(self):
        '''
            Input: word_tokens in the format output by words_to_tokens
            Output: A sparse matrix of counts where each row a
                    "document" in the corpus.
        '''
        min_ngram = self.n_order
        if self.append_ngrams:
            min_ngram = 1
        ngram_range = (min_ngram, self.n_order)

        vectorizer = self.vectorizer(analyzer="char_wb",
                                    encoding="utf-8",
                                    strip_accents='unicode',
                                    ngram_range=ngram_range)
        
        vectorizer.fit(self.wordlist)
        self._ngram_vectorizer = vectorizer


    def set_subset(self, idxs):
        self.subset = self.word_features[idxs,:].sum(axis=0)
   
    
    def add_to_subset(self, idx):
        self.subset += self.word_features[idx, :]

        
    def run(self, idx_new):
        return self._g(self.word_features[idx_new, :] + self.subset).sum()

    
    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.test_wordlist)
        if self.binarize_counts:
            self.word_features[self.word_features > 0 ] = 1.0
        self.p = self.word_features.sum(axis=0) / float(self.word_features.sum())


    def compute_entropy(self):
        prob_vec = self.subset / float(self.subset.sum())
        return -(np.multiply(prob_vec, np.log2(prob_vec + sys.float_info.epsilon))).sum()


    def compute_kl(self):
        prob_vec = (1.0 + self.subset) / (float(self.subset.sum()) + self.subset.shape[1])
        return np.multiply(self.p, np.log2(self.p + sys.float_info.epsilon) - np.log2(prob_vec)).sum() 


# Coverage Method
class FeatureCoverageObjective(FeatureObjective):
    def __init__(self, *args, **kwargs):
        self.total_counts = None
        self.K = None
        kwargs["binarize_counts"] = True
        super(FeatureCoverageObjective, self).__init__(*args, **kwargs)
          

    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.test_wordlist)
        self.word_features[self.word_features > 0 ] = 1.0
        self.total_counts = self.word_features.sum(axis=0)
        self.p = self.total_counts / float(self.word_features.sum())
        self.K = self.total_counts.sum()


    def run(self, idx_new):
        vec = self.subset + self.word_features[idx_new, :]
        p_vec = np.squeeze(np.asarray(self.total_counts))
        vec = np.squeeze(np.asarray(vec))
        return self.K - np.multiply(p_vec, (1.0 / (8.0 ** vec))).sum()


