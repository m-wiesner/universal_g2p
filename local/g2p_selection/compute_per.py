#!/usr/bin/python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import codecs
import re

MAX_VAL = 9999999

# Evaluate the PER and WER for a hypothesized (hyp) lexicon.
def eval_g2p(hyp, ref, pron2pron):
    # Find the set of words unique to the reference. None of these words
    # occur in hyp so we will count them all as having PER=1
    ref_words_only = set(ref.keys()).difference(hyp.keys())
   
    # Counter for the total number of phoneme errors, word errors, number of
    # phonemes, and number of words
    p_errs = 0.0
    num_phones = 0.0
    w_errs = float(len(ref_words_only))
    num_words = float(len(ref_words_only))
    
    # For each word in the hypothesis (that also occurs in reference) we 
    # accumulate the number of phoneme errors. If multiple pronunciations exist
    # in the reference, we score against the best possible pronunciation
    for w, prons_hyp in hyp.iteritems():
        if w in ref:
            min_per = MAX_VAL
            for p_hyp in prons_hyp:
                for p_ref in ref[w]:
                    num_errs = edit_distance(pron2pron(p_hyp.split()), p_ref.split())
                    p_len = float(len(p_ref.split()))
                    p_per = num_errs / p_len
                    if p_per <= min_per:
                        min_per = p_per
                        min_num_errs = num_errs
                        min_len = p_len
            
            p_errs += min_num_errs
            num_phones += min_len
            w_errs += (min_num_errs != 0)
            num_words += 1
   
    # For each word occurring only in reference, we use the average number of
    # phonemes in the reference pronunciation for scoring.
    for w in ref_words_only:
        errs_w = sum([len(p) for p in ref[w]]) / float(len(ref[w]))
        num_phones += errs_w
        p_errs += errs_w
         
    return p_errs/num_phones, w_errs/num_words 


# Load lexicon into dictionary of sets. The keys are words and the values are
# sets of pronunciations for each word
def load_lex(lex):
    lexicon = {}
    phones = set()
    for i_l, l in enumerate(lex):
        word_and_pron = l.strip().split('\t', 1)
        if len(word_and_pron) == 0:
            continue
        elif len(word_and_pron) == 1:
            word = word_and_pron[0]
            pron = ""
        elif len(word_and_pron) == 2:
            word, pron = word_and_pron
        pron = re.sub(r'\s+', " ", pron)
        pron = re.sub(r'\s+$', "", pron)
    
        if word in lexicon:
            lexicon[word].add(pron)
        else:
            lexicon[word] = set([pron])
        phones.update(pron.split())
    
    return lexicon, phones


# Compute edit distance between two iterable objects 
def edit_distance(h, r):
    if not h and not r:
        return 0
    elif not h:
        return len(r) 
    
    r_len = len(r)
    DP = range(r_len + 1)
    for i in range(1, len(h) + 1):
        DP_prev = i-1
        tmp = DP[0] + DP_prev
        for j in range(1, len(r) + 1):
            DP_prev = min(tmp + (r[j-1] != h[i-1]), DP[j] + 1, DP_prev + 1) 
            tmp = DP[j]
            DP[j] = DP_prev  
    return DP[-1]


def load_phon2phon(f, iphones, ophones):
    # 1 state fst loop
    #(ilabel, olabel, w)
    phon2phon = {p: (p, -MAX_VAL) for p in iphones}
    for l in f:
        ilabel, olabel, w = l.strip().split('\t')           
        if olabel in ophones and ilabel in iphones: 
            if float(w) > phon2phon[ilabel][1]:
                phon2phon[ilabel] = (olabel, float(w))
    return phon2phon 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp',
        help='hypothesized lexicon',
        type=str
    )
    parser.add_argument('ref',
        help='reference lexicon',
        type=str
    )
    parser.add_argument('--phon2phon',
        help='phoneme similarity table',
        type=str,
        default=None,
        action='store'
    )
    args = parser.parse_args()
    
    # Load hypothesis lexicon
    with codecs.open(args.hyp, 'r', encoding='utf-8') as f:
        hyp, iphones = load_lex(f)

    # Load reference lexicon
    with codecs.open(args.ref, 'r', encoding='utf-8') as f:
        ref, ophones = load_lex(f)

    # Phoneme mapping from predicted phones to closest in-language phones
    pron2pron = lambda x: x
    if args.phon2phon:
        with codecs.open(args.phon2phon, 'r', encoding='utf-8') as f:
            phon2phon = load_phon2phon(f, iphones, ophones) 
        pron2pron = lambda x: [phon2phon[p][0] for p in x] 
    
    # Evaluate
    per, wer = eval_g2p(hyp, ref, pron2pron=pron2pron)
    print("PER: ", per)
    print("WER: ", wer)


if __name__ == "__main__":
    main()

