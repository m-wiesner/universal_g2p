from __future__ import print_function
import numpy as np
import codecs
import argparse
import sys
import os
import json
from G2PSelection import BatchActiveSubset, RandomSubset
from SubmodularObjective import FeatureObjective, FeatureCoverageObjective


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("output",
        help="output file with selected g2p words",
        type=str)
    parser.add_argument("wordlist",
        help="Path with a list of words",
        type=str)
    parser.add_argument("budget",
        help="The total budget allocated for words",
        type=float)
    parser.add_argument("--n-order",
        help="The maximum order n-gram",
        type=int,
        default=4,
        action="store")
    parser.add_argument("--constraint",
        help="The type of constraint to limit the number of selected words",
        choices=['card', 'len', 'freq'],
        default="card",
        action="store")
    parser.add_argument("--subset-method",
        help="The method used to select data",
        choices=['BatchActive', 'Random'],
        default="BatchActive",
        action="store")
    parser.add_argument("--objective",
        help="The submodular objective used",
        choices=['Feature', 'FeatureCoverage'],
        default="FeatureCoverage",
        action="store")
    parser.add_argument("--cost-select",
        help="Use cost in selection",
        action="store_true")
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
    
    if args.constraint == "freq":
        total_count = 0.0
        try:
            with codecs.open(args.wordlist, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        word, count = l.strip().split('\t', 1)
                        words.append(word)
                        freqs[word] = float(count)
                        total_count += float(count)
        except IndexError:
            print("Poorly formatted wordlist file. When using the "
                  "--constraint freq option, words and counts are required"
                  "in two tab separated columns")
             
        for w, c, in freqs.iteritems():
            freqs[w] = -np.log2(c / total_count + sys.float_info.epsilon)

    else:
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


    fobj = objectives[args.objective](
        words,
        test_wordlist=test_words,
        n_order=args.n_order,
        append_ngrams=args.append_ngrams,
        binarize_counts=args.binarize_counts
    )
    
    # Store experiment configurations
    #with codecs.open(args.output + ".conf", "w", encoding="utf-8") as f:
    #    json.dump(args, f, indent=4)
     
    bas = methods[args.subset_method](
        fobj,
        args.budget,
        test_words,
        cost=cost_functions[args.constraint],
        cost_select=args.cost_select
    )
    
    print("Begin Selection ...")
    selected_words = bas.run_lazy()
    
    if os.path.dirname(args.output) not in ('', '.'):
        if not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))
    
    with codecs.open(args.output, "w", encoding="utf-8") as f:
        for w in selected_words:
            print(w, file=f) 
  
     
if __name__ == "__main__":
    main() 
 
