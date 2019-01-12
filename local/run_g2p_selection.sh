#!/bin/bash

. ./path.sh

cmd="run.pl" # Run on single (multi-thread) cpu or distribute to cluster
num_trials=1 # Number of randomized trials to be conducted
stage=1 # Which part of training
n_order=1 # N-gram order. Useful only for BatchActive Selection (not Random)
constraint="card" # Type of limiting budget constraint. Card is number words, 'len' is number of letters
vectorizer="count" # How to count tokens. "count" gives each n-gram a count of 1, while tf-idf scales by idf and applies log
objective="Feature" # Which n-gram feature type to use for BatchActive Selection 
method="BatchActive" # Selection Method. BatchActive or Random
cost_select=true # Use cost-normalized value for BatchActive Selection 
append_ngrams=true # Use all order < n, n-grams as features in addition to all order-n n-gram features
binarize_counts=true # For BatchActive selection, a single feature is not counted twice within a single word
intervals="100 200 400 800 1600 3200 6400 12800 20000 25600 36000 50000" # Selection intervals
score=true # To score or not score

###############################################################################
# The below options are more useful for the Universal G2P Experiments
###############################################################################

test_words= # Words whose pronunciation we will derive from g2p
test_ref_lexicon= # The reference lexicon corresponding to the test_words
select_words= # Word list from which to select in order to train g2p 
select_lexicon= # Words an assosiated pronunciation from which we will select

. ./utils/parse_options.sh
if [ $# -eq 0 ]; then
  echo "Usage: ./local/run_g2p_rand_select.sh [opts] <odir> <words> <lexicon> <budget>"
  echo "    [opts] --n-order : max n-gram order for bow features"
  echo "           --num_trials : number of random trials to perform"
  echo "           --constraint : the type of budget constraint ['card', 'freq', 'len', 'rootlen']" 
  echo "           --intervals : the subset sizes (in words)"
  echo "           --stage : default=1"
  echo "           --vectorizer : count, tfidf"
  echo "           --objective : Feature, CrossEntropy"
  echo "           --method : BatchActive, Random"
  echo "           --cost-select : Select based on cost (knapsack constrained)"
  echo "           --append-ngrams : Append all ngrams upto order n_order"
  echo "           --binarize-counts : ngrams get a maximum count of 1 per word (default true)"
  echo "           --cmd : run.pl, queue.pl"
  echo "           --score : true, false (default true)"
  echo "           --test-words : path/to/test/words (default empty)"
  echo "           --test-ref-lexicon : path/to/reference/lexicon (default empty)"
  exit 1
fi 

odir=$1
words=$2
ref_lex=$3
budget=$4

###############################################################################
#                              Process input arguments
###############################################################################
# Check input paths
[ ! -z $words ] || [ -f $words ] || { echo "Expected $words to exist" && exit 1; }
[ ! -z $ref_lex ] || [ -f $ref_lex ] || { echo "Expected $words to exist" && exit 1; }

# Create output directory
owords=${odir}/budget_${budget}
mkdir -p ${owords}/log

# Set extra opts for g2p selection
extra_opts=
if $cost_select; then
  extra_opts="$extra_opts--cost-select"
fi
if $append_ngrams; then
  extra_opts="$extra_opts --append-ngrams"
fi
if $binarize_counts; then
  extra_opts="$extra_opts --binarize-counts"
fi
if [ ! -z $select_words ]; then
  extra_opts="$extra_opts --test-wordlist $select_words"
fi

# Check $num_trials
[[ $method = "BatchActive" ]] && [[ $num_trials -ne 1 ]] && echo "BatchActive" \
  "selection is deterministic, requiring a single run. Setting num_trials=1" \
  && num_trials=1

# Remove all intervals > budget and append budget to interavls
new_intervals=""
for i in ${intervals}; do
  if [ $i -lt $budget ]; then
    new_intervals="$new_intervals $i"
  fi
done
new_intervals="$new_intervals $budget"
intervals=$new_intervals

# This sets empty variables to default to selection within and evaluation from 
# and on the same set of words
[ ! -z $test_words ] && [ -f $test_words ] || test_words=$words 
[ ! -z $test_ref_lexicon ] && [ -f $test_ref_lexicon ] || test_ref_lexicon=$ref_lex

echo "---------- Parameters -----------"
echo $0
echo "Intervals: $intervals"
echo "num_trials: $num_trials"
echo "extra_opts: $extra_opts"
echo "Words: $words"
echo "Test Words: $test_words"
echo "Reference Lexicon: $ref_lex"
echo "---------------------------------"

###############################################################################
#                            Run G2P Selection
###############################################################################
# Run this first 
if [ $stage -le 1 ]; then
  $cmd ITER=1:$num_trials $owords/log/run.ITER.log \
    python local/g2p_selection/g2p/select_g2p.py --n-order $n_order \
                                   --constraint $constraint \
                                   --subset-method $method \
                                   --objective $objective \
                                   $extra_opts \
                                   $owords/trial.ITER/words.txt $words $budget || exit 1;
fi
###############################################################################
#  Get pronunciations for all words in each g2p selection produced in stage=1
###############################################################################
if [ $stage -le 2 ]; then
  # Next run the following
  for num in ${intervals[@]}; do
    num_dir=${odir}/budget_${num}
    if [ ! -d $num_dir ]; then
      for i in `seq 1 $num_trials`; do
        mkdir -p $num_dir/trial.$i/log
      done
    fi
    run.pl ITER=1:$num_trials $num_dir/trial.ITER/log/lexicon_retrieve.orig.log \
      ./local/g2p_selection/word_subset2lexicon.sh --constraint $constraint $owords/trial.ITER/words.txt $select_lexicon $num \> ${num_dir}/trial.ITER/lexicon.orig
    
    run.pl ITER=1:$num_trials $num_dir/trial.ITER/log/lexicon_retrieve.log \
      ./local/g2p_selection/transform_4_g2p.sh ${num_dir}/trial.ITER/lexicon.orig \> ${num_dir}/trial.ITER/lexicon.orig.transform.tmp
 
    run.pl ITER=1:$num_trials $num_dir/trial.ITER/log/lexicon_retrieve.log \
      python ./local/translit_oov_letters.py $test_words ${num_dir}/trial.ITER/lexicon.orig.transform.tmp \> ${num_dir}/trial.ITER/lexicon.orig.transform

  done
fi

if [ $method = "Random" ]; then 
  extra_scoring_opts="${extra_scoring_opts}--random true"
else
  extra_scoring_opts="${extra_scoring_opts}--random false"
fi

if [ $constraint = "len" ]; then
  extra_scoring_opts="${extra_scoring_opts} --len true"
else
  extra_scoring_opts="${extra_scoring_opts} --len false"
fi

echo "Test Words: $test_words"
test_words_name=`basename $test_words`
###############################################################################
#         Learn G2P-based lexicon for all new words in $words
###############################################################################
echo "Scoring opts: $extra_scoring_opts"
if [ $stage -le 3 ]; then
  # This script trains and applies a G2P for each lexicon created from
  # word_subset2lexicon.sh. It then also scores it against a reference lexicon. 
 
  if [ -d ${odir}/g2p ]; then
    rm -r ${odir}/g2p
  fi
  
  ./local/g2p_selection/run_all_budgets.sh --cmd "queue.pl" --ref-lex $test_ref_lexicon \
                             --words $test_words --score $score $odir

  # The --words option is to specify which set of tested words should be scored
  # It's possible that we test in training vs. heldout words for instance and
  # to specify which set of experiments should be scored we use the --words opt
  if $score; then
    ./local/g2p_selection/run_average_ser.sh ${extra_scoring_opts} --words $test_words_name $odir
  fi
fi


exit 0;
