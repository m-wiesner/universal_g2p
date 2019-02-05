from __future__ import print_function
import numpy as np
import codecs
import argparse
import sys
import os
import json
from G2PSelection import BatchActiveSubset, RandomSubset
from SubmodularObjective import FeatureObjective, FeatureCoverageObjective
import pdb

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("output",
        help="output file with selected g2p words",
        type=str)
    parser.add_argument("wordlist",
        help="Path with a list of words",
        type=str)
    parser.add_argument("kl_smooth",
        help="Amount to smooth KL div computation",
        type=float)
    parser.add_argument("--n-order",
        help="The maximum order n-gram",
        type=int,
        default=4,
        action="store")
    parser.add_argument("--objective",
        help="The submodular objective used",
        choices=['Feature', 'FeatureCoverage'],
        default="FeatureCoverage",
        action="store")
    parser.add_argument("--append-ngrams",
        help="Append, n, n-1, ..., 1-gram features",
        action="store_true")
    parser.add_argument("--binarize-counts",
        help="Counts are 0 or 1",
        action="store_true")
    parser.add_argument("--test-wordlist",
        help="Path with a list of words from which to select",
        type=str,
        default=None,
        action="store")


    return parser.parse_args()


def main():
    args = parse_input()

    def card(w):
        return 1.0

    # ---------------------------------------
    # Extract train and test word sets
    # ---------------------------------------
    words = []
    freqs = {}
    
    with codecs.open(args.wordlist, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                words.append(l.strip())
    def freq(w):
        try:
            return freqs[w]
        except KeyError:
            return 1.0

    cost_functions = {'card': card, 'len': len, 'freq': freq}
    methods = {'BatchActive': BatchActiveSubset, 'Random': RandomSubset}

    objectives = {
                    'Feature': FeatureObjective,
                    'FeatureCoverage': FeatureCoverageObjective,
                 }

    if args.test_wordlist is not None:
        test_words = []
        with codecs.open(args.test_wordlist, "r", encoding="utf-8") as f:
            for l in f:
                test_words.append(l.strip())
    else:
        test_words = words 

    owords = []
    with codecs.open(args.output, 'r', encoding='utf-8') as f:
        for l in f:
            owords.append(l.strip()) 

    # The submodular objective
    fobj = objectives[args.objective](
        words,
        test_wordlist=test_words,
        n_order=args.n_order,
        append_ngrams=args.append_ngrams,
        binarize_counts=args.binarize_counts
    )
    
    fobj.set_subset([])
    word_map = {w: i for i, w in enumerate(test_words)}
    pdb.set_trace()
    KL = []
    for w in owords:
        fobj.add_to_subset(word_map[w])
        KL.append(fobj.compute_kl(lam=args.kl_smooth))
    
    print(np.argmin(KL))

if __name__ == "__main__":
    main()
