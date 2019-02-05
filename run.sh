#!/bin/bash

#                              Example
# _____________________________________________________________________________

#  Below is an example of how to run the G2P selection for universal G2P transfer
#  to other languages.
#  
#  1. We select from words in words_roman.txt for which we have
#  pronunciations in lexicon_roman.txt. 
#  
#  2. We place the result of the entire experiment
#  in SPNBDA_random_roman_selection. We limit ourselves to picking 50000 words from
#  this list.
#  
#  3. We will evaluate the quality of our G2P by creating a lexicon for words in
#  SPA_words.txt. These are Spanish words that had pronunciations on wiktionary
#  
#  4. We score our induced lexicon against a reference lexicon, the path to which
#  is shown below next to the option --test-ref-lexicon
#  
#  5. Resulting statistics about the tested word list should be in a file in:
#     
#     SPNBDA_random_roman_selection/stats.${WORDLIST_NAME}.txt 
#
#==============================================================================

# Random Selection 
#./local/run_g2p_selection.sh --cmd "queue.pl" \
#                             --stage 3 \
#                             --num-trials 20 \
#                             --constraint card \
#                             --intervals "100 200 400 800 1600 3200 6400 12800 25600 36000 50000 100000 200000 400000" \
#                             --vectorizer count \
#                             --objective FeatureCoverage \
#                             --method Random \
#                             --cost-select false \
#                             --append-ngrams true \
#                             --score true \
#                             --select-lexicon train_spa_lexicon.txt \
#                             --select-words train_spa_words.txt \
#                             --test-words spa_words.txt \
#                             --test-ref-lexicon spa_lexicon.txt \
#                             spa_random_selection train_spa_words.txt train_spa_lexicon.txt 400000

# This script will run the Batch active selection with automatic stopping
# criterion
# Batch Active Selection                             
./local/run_g2p_selection.sh --cmd "queue.pl" \
                             --num-trials 1 \
                             --constraint card \
                             --stage 2 \
                             --intervals "100 200 400 800 1600 3200 6400 12800 25600 36000 50000 100000" \
                             --optimal-only false \
                             --vectorizer count \
                             --objective FeatureCoverage \
                             --method BatchActive \
                             --cost-select true \
                             --append-ngrams true \
                             --score true \
                             --select-lexicon train_spa_lexicon.txt \
                             --select-words train_spa_words.txt \
                             --test-words spa_words.txt \
                             --test-ref-lexicon spa_lexicon.txt \
                             spa_batchactiveLen5_selection spa_words.txt train_spa_lexicon.txt 100000
#
