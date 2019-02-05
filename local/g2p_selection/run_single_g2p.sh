#!/bin/bash

. ./path.sh

nbest=1
score=true
phon2phon=/export/b14/mwiesner/JSALT_07_25_2018/espnet/tools/kaldi_github/egs/universal_g2p/phon2phon/ipa.bitdist.table

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/run_single_g2p.sh <lex> <words> <lexicon>"
  exit 1
fi

trial=$1
words=$2
ref_lex=$3

trial=`readlink -f $trial`
words_name=`basename $words`
words_name=${words_name%.*} # Remove extension
echo $trial

# Exclude all words from $words that are in training
#comm -23 <(LC_ALL=C sort $words) \
#         <(awk -F'\t' '{print $1}' ${trial}/lexicon.orig |\
#           LC_ALL=C sort -u) |\
#           LC_ALL=C sort > ${trial}/${words_name}.test

cp ${words} ${trial}/${words_name}.test

# Transform words
./local/g2p_selection/transform_4_g2p.sh ${trial}/${words_name}.test |\
  paste - ${trial}/${words_name}.test | LC_ALL=C sort -t$'\t' -k1,1 |\
  tee ${trial}/${words_name}.test.transform.map.tmp | awk -F'\t' '{print $1}' \
  > ${trial}/${words_name}.test.transform.tmp
 
python ./local/translit_oov_letters.py $trial/lexicon.orig.transform ${trial}/${words_name}.test.transform.tmp > ${trial}/${words_name}.test.transform
python ./local/translit_oov_letters.py $trial/lexicon.orig.transform ${trial}/${words_name}.test.transform.map.tmp > ${trial}/${words_name}.test.transform.map

# Train g2p on the selected lexicon
./local/g2p_selection/train_g2p.sh $trial/lexicon.orig.transform $trial/g2p

# Apply g2p
./local/g2p_selection/apply_g2p.sh --nbest $nbest --nj 1\
  ${trial}/${words_name}.test.transform $trial/g2p $trial/g2p/${words_name}

# Get lexicon
cat ${trial}/g2p/${words_name}/*_out.* |\
  tee ${trial}/g2p/${words_name}/lexiconp.txt | cut -f1,3 |\
  paste <(cut -f2- ${trial}/${words_name}.test.transform.map) <(cut -f2- -) \
  > ${trial}/g2p/${words_name}/lexicon.txt 

# Score
if $score; then
  python local/g2p_selection/compute_per.py --phon2phon ${phon2phon} ${trial}/g2p/${words_name}/lexicon.txt ${ref_lex} 2>&1 |\
    tee ${trial}/g2p/${words_name}/score.txt
fi
