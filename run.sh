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
./local/run_g2p_selection.sh --cmd "queue.pl" \
                             --num-trials 20 \
                             --constraint card \
                             --intervals "100 200 400 500" \
                             --vectorizer count \
                             --objective FeatureCoverage \
                             --method Random \
                             --cost-select false \
                             --append-ngrams true \
                             --score true \
                             --select-lexicon lexicon_roman.txt \
                             --select-words words_roman.txt \
                             --test-words SPA_words.txt \
                             --test-ref-lexicon /export/b09/mwiesner/LORELEI/tasks/process_set0/set0/local/lexicons/Spanish \
                             SPNBDA_random_roman_selection words_roman.txt lexicon_roman.txt 500

# Batch Active Selection                             
#./local/run_g2p_selection.sh --cmd "queue.pl" \
#                             --num-trials 1 \
#                             --constraint card \
#                             --intervals "100 200 400 800 1600 3200 6400 12800 20000 25600 36000 50000" \
#                             --vectorizer count \
#                             --objective FeatureCoverage \
#                             --method BatchActive \
#                             --cost-select true \
#                             --append-ngrams true \
#                             --score true \
#                             --select-lexicon lexicon_roman.txt \
#                             --select-words words_roman.txt \
#                             --test-words SPA_words.txt \
#                             --test-ref-lexicon /export/b09/mwiesner/LORELEI/tasks/process_set0/set0/local/lexicons/Spanish \
#                             SPNBDA_random_roman_selection words_SPNBDA_allowed.txt lexicon_roman.txt 50000

