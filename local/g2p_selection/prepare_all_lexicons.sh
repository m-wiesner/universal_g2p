#!/bin/bash

. ./path.sh

langs="101 102 103 104 105 106 107 201 202 203 204 205 206 207 301 302 303 \
       304 305 306 307 401 402 403 404"

stage=1
dictionary_dir=data/local

. ./utils/parse_options.sh


# Go fetch the right (FLP) lexicon

cwd=$PWD
if [ $stage -le 1 ]; then
  for l in ${langs}; do
    echo "Language: ${l}"
    mkdir -p ${cwd}/data/${l}
    cd ${cwd}/data/${l}
    langconf=$(find ../../conf/ -name "${l}*.FLP.*.conf")
    ln -s $langconf lang.conf
    ln -s ../../local .
    ln -s ../../conf .
    . ./lang.conf
  
    ./local/prepare_lexicon.pl $lexiconFlags --oov "<unk>" $lexicon_file $dictionary_dir
    
    # Save map from integer keys to original words and visa versa
    mkdir -p ${dictionary_dir}/g2p_sel
    
    # Remove <.*> words, <unk>, <silence>, etc.. since no grapheme-to-phoneme map
    # can be learned for these
    grep -v '<.*>' ${dictionary_dir}/lexicon.txt > ${dictionary_dir}/g2p_sel/filtered_lexicon.txt 
    
    cut -f2- ${dictionary_dir}/g2p_sel/filtered_lexicon.txt |\
      LC_ALL= sed 's/\t//g' |\
      paste <(cut -f1 ${dictionary_dir}/g2p_sel/filtered_lexicon.txt) - |\
      tee ${dictionary_dir}/g2p_sel/lexicon.txt | cut -f1 | LC_ALL=C sort -u \
      > ${dictionary_dir}/g2p_sel/words.txt
    cd ${cwd}
  done
fi

if [ $stage -le 2 ]; then
  for l in ${langs}; do
    cat $(find data/*/data/local/g2p_sel/lexicon.txt | grep -v "${l}") > data/${l}/lexicon_pool
    awk '{print $1}' data/${l}/lexicon_pool > data/${l}/word_pool
  done
fi
