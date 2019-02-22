from __future__ import print_function
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Base method
class FeatureObjective(object):
    '''
        FeatureObjective: Base class for all other feature-based submodular
                          selection methods.
        
        Inputs (required): wordlist -- A list of words from which n-gram
                            statistics are extracted. These statistics are then
                            used to select words, from a set
                            (potentially the same) of words in an optimal order
                            according to the submodular objective used.
        
        Inputs (optional): test_wordlist
                                type: array
                                description: A list of word from which to select
                                default: None
                           n_order
                                type: int
                                description: ngram order
                                default: 4
                           append_ngrams
                                type: bool
                                description: Also use statistics from all lower-order ngrams
                                default: True
                           binarize_counts
                                type: bool
                                description: Count any given ngram once per word
                                default: False
                           g
                                type: func
                                description: The submodular function to use
                                default: np.sqrt
                           vectorizer
                                type: str
                                description: Weight applied to ngram counts (tfidf or count)
                                default: 'count'
            
    '''
    def __init__(self, wordlist, test_wordlist=None, n_order=4,
                append_ngrams=True, binarize_counts=False,
                g=np.sqrt, vectorizer='count', phoneme_wordlist=None): # TODO phoneme_wordlist shouldn't be here but it needed the consistent interface with PhonemeFeatureCoverageObjective.

        self.wordlist = wordlist
        
        if not test_wordlist:
            self.test_wordlist = wordlist
        else:
            self.test_wordlist = test_wordlist
        
        self._g = g
        self._ngram_vectorizer = None
        
        self.n_order = n_order
        self.append_ngrams = append_ngrams
        self.binarize_counts = binarize_counts
        
        if vectorizer == 'count':
            self.vectorizer = CountVectorizer
        else:
            self.vectorizer = TfidfVectorizer

        self.word_features = None
        self.subset = None
        self.p = None
        
        self.get_vectorizer()
        self.get_word_features()
    
    
    def get_vectorizer(self):
        '''
            Input: word_tokens in the format output by words_to_tokens
            Description: Sets a vectorizer which can generate a sparse matrix of counts
                    from a corpus. where each row is a "document" in the corpus,
                    and each column is a feature seen in the corpus self.wordlist
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
        '''
            Sets the subset (sum of hitherto selected words) to the sum of
            those words in self.wordlist indexed by idxs.

            Input: idxs -- list of indicies into self.test_wordlist
        '''
        self.subset = self.word_features[idxs,:].sum(axis=0)
   
    
    def add_to_subset(self, idx):
        '''
            Add to the subset the counts from the word corresponding to the
            single index (idx)

            Input: idx -- index of single entry in self.test_wordlist
        '''
        self.subset += self.word_features[idx, :]

        
    def run(self, idx_new):
        '''
            Return the submodular objective obtained by the addition of the word
            indexed by idx_new

            Input: idx_new -- the index of a single new word from
                              self.test_wordlist
        '''
        return self._g(self.word_features[idx_new, :] + self.subset).sum()

    
    def get_word_features(self):
        '''
            Precompute all the word features (ngram statistics basically) for
            the words from both self.wordlist as well as the words in
            self.test_wordlist.
        '''
        wordlist = self.test_wordlist
        if not self.test_wordlist:
            wordlist = self.wordlist

        self.word_features = self._ngram_vectorizer.transform(wordlist) 
        print("wordlist:")
        print(wordlist)
        print("self.word_features:")
        print(self.word_features)
        train_word_features = self._ngram_vectorizer.transform(self.wordlist)
        print("self.train_word_features:")
        print(train_word_features)
        
        if self.binarize_counts:
            train_word_features[train_word_features > 0] = 1.0
            self.word_features[self.word_features > 0 ] = 1.0
        
        self.total_counts = train_word_features.sum(axis=0)
        print("len(total_counts):")
        print(len(self.total_counts))
        self.p = self.total_counts / float(train_word_features.sum())


    def compute_entropy(self):
        '''
            Compute the empirical Entropy of the feature distribution induced
            by the currently selected subset
        '''
        prob_vec = self.subset / float(self.subset.sum())
        return -(np.multiply(prob_vec, np.log2(prob_vec + sys.float_info.epsilon))).sum()


    def compute_kl(self, lam=0.001):
        '''
            Compute the KL-Divergence between the feature distribution induced
            by the currently selected subset and the empirical distribution of
            the words in self.wordlist
        '''
        prob_vec = (lam + self.subset) / (float(self.subset.sum()) + lam * self.subset.shape[1])
        return np.multiply(self.p, np.log2(self.p + sys.float_info.epsilon) - np.log2(prob_vec)).sum() 


# Coverage Method
class FeatureCoverageObjective(FeatureObjective):
    def __init__(self, *args, **kwargs):
        self.total_counts = None
        self.K = None
        kwargs["binarize_counts"] = True
        super(FeatureCoverageObjective, self).__init__(*args, **kwargs)
          

    def get_word_features(self):
        super(FeatureCoverageObjective, self).get_word_features()
        
        test_word_counts = self.word_features.sum()
        self.total_counts = (test_word_counts / self.total_counts.sum()) * self.total_counts
        self.K = test_word_counts


    def run(self, idx_new):
        vec = self.subset + self.word_features[idx_new, :]
        p_vec = np.squeeze(np.asarray(self.total_counts))
        vec = np.squeeze(np.asarray(vec))
        return self.K - np.multiply(p_vec, (1.0 / (8.0 ** vec))).sum()

class PhonemeFeatureCoverageObjective(FeatureCoverageObjective):
    """ An extension of FeatureCoverageObjective that also includes phoneme
    features. """
    def __init__(self, *args, **kwargs):
        self.phoneme_vectorizer = CountVectorizer
        #self.phoneme_inventory = kwargs["phoneme_inventory"]
        self.phoneme_inventory = ["a", "n"] #TODO replace with commented line above
        self.test_phoneme_wordlist = kwargs["phoneme_wordlist"]
        self.get_phoneme_vectorizer()
        super(PhonemeFeatureCoverageObjective, self).__init__(*args, **kwargs)

    def get_phoneme_vectorizer(self):
        '''
            Input: word_tokens in the format output by words_to_tokens
            Description: Sets a vectorizer which can generate a sparse matrix of counts
                    from a corpus. where each row is a "document" in the corpus,
                    and each column is a feature seen in the corpus self.wordlist
        '''
        # We use phoneme uni-grams because we don't know any thing about
        # potential higher order n-grams; only what the phoneme inventory is.
        phoneme_vectorizer = self.phoneme_vectorizer(analyzer="char_wb",
                                    encoding="utf-8",
                                    strip_accents='unicode',
                                    ngram_range=(1,1))
        phoneme_vectorizer.fit(self.phoneme_inventory)
        print("feature names:")
        print(phoneme_vectorizer.get_feature_names())
        self._phoneme_vectorizer = phoneme_vectorizer

    def get_word_features(self):
        '''
            Precompute all the word features (ngram statistics basically) for
            the words from both self.wordlist as well as the words in
            self.test_wordlist.

            Also precompute all the word phoneme features (1-gram statistics of
            phones from the pronunciation lexicon) for
            the words from words in self.test_wordlist (candidate words from
            another resource). We can't get them from wordlist unless we
            assumed we had a pronunciation lexicon in the target language.
        '''
        if not self.test_wordlist:
            raise NotImplementedError("""For now we're assuming we only have
            pronunciations for test_wordlist (the candiate words from other
            languages). We could use existing pronunciation lexicons in the
            target language but that's not the point for the ACL paper.""")
            wordlist = self.wordlist

        wordlist = self.test_wordlist

        # Pronunciations of the words that are in wordlist
        pronun_list = ["".join(pronunciation) for _word, pronunciation in self.test_phoneme_wordlist]
        print(len(wordlist))
        print(len(pronun_list))
        assert len(wordlist) == len(pronun_list)
        print("pronun_list:")
        print(pronun_list[:10])

        # self.word_features are the features for the words we are selecting
        # from ('test word features')
        self.word_features = self._ngram_vectorizer.transform(wordlist) 
        print("wordlist:")
        print(wordlist)
        print("self.word_features:")
        print(self.word_features)
        # train_word_features gets us our initial word features.
        train_word_features = self._ngram_vectorizer.transform(self.wordlist)
        print("self.train_word_features:")
        print(train_word_features)

        self.word_phoneme_features = self._phoneme_vectorizer.transform(pronun_list) 
        print("self.word_phoneme_features:")
        print(self.word_phoneme_features)
        # TODO We don't have train_word_features, so we'll have to fake them by
        # taking the phoneme inventory and scaling the frequency appropriately.
        # Then that train_word_phoneme_features variable will be
        # appended to train_word_features.
        # train_word_phoneme_features = <get some scaled uniform features based
        # on the phoneme inventory of the language>

        # Append the word_phoneme_features to word_features.
        print("type(self.word_features)")
        print(type(self.word_features))

        if self.binarize_counts:
            train_word_features[train_word_features > 0] = 1.0
            self.word_features[self.word_features > 0 ] = 1.0
            #self.word_phoneme_features[self.word_phoneme_features > 0 ] = 1.0

        self.total_counts = train_word_features.sum(axis=0)
        print("len(total_counts):")
        print(len(self.total_counts))
        self.p = self.total_counts / float(train_word_features.sum())

        # NOTE I've copy-pasted code a bit here. Above includes stuff from
        # FeatureObjective.get_word_features(). Below is stuff from
        # FeatureCoverageObjective.get_word_features().
        test_word_counts = self.word_features.sum()
        self.total_counts = (test_word_counts / self.total_counts.sum()) * self.total_counts
        self.K = test_word_count
