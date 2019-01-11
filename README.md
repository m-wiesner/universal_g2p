# G2P Selection

This repo provides tools for generating pronunciations from words in a new
language and evaluating the quality of the induced pronunciation lexicon (w/r
to a reference).  

##Software Requirements
1. Phonetisaurus.
2. srilm (ngram-count)

You can install these very easily with kaldi. So these experiments should be
run in a new example directory inside a kaldi installation, such as what is
installed in ESPNET.

Copy this file ...

/export/b14/mwiesner/JSALT_07_25_2018/espnet/tools/kaldi_github/tools/srilm.tgz

to ${KADLI_PATH}/tools/ 

Then go to ${KALDI_PATH}/tools and run ...

./extras/install_phonetisaurus.sh
./extras/install_srilm.sh

Otherwise I included my own installations in path.sh that you can use. Just
comment (in/out) the appropriate lines in path.sh

##Example Experiment
First link some kaldi stuff ..

ln -s ../../wsj/s5/{steps,utils} . 

Relevant Files
-------------------------------------------------------------------------------
For an example experiment, use the provided files:
  1. SPA_words.txt     -- small list of test words for Spanish.

  2. lexicon_roman.txt -- pooled pronunciation lexicon of all BABEL langauges
                          with Latin orthography.
  
  3. words_roman.txt   -- list of all candidate words from which to select G2P
                          training set.
  
  4. words_SPNBDA_allowed.txt -- list of words without weird puntuation from
                                 which to derive n-gram features used in 
                                 Batch Active Selection.

Example run script
-------------------------------------------------------------------------------

Look inside run.sh for an example of how to run the G2P experiment and
for an explanation of all of the arguments.

The main script that runs this experiment is:

  ./local/run_g2p_selection.sh

This is worth looking at for more information about optional arguments especially.

More information
-------------------------------------------------------------------------------

All things pertaining to G2P selection are in the folder:
  ./local/g2p_selection

All of the actual word selection is handled by:
  ./local/g2p_selection/g2p/{SubmodularObjective,G2PSelection}.py
  And run via python script in the same directory select_g2p.py.
  See that file for more information.
