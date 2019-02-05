#!/bin/bash

words="words.txt"
. ./utils/parse_options.sh

dir=$1

[ -f ${dir}/symb_er.txt ] && rm ${dir}/symb_er.txt
scores=( `find $dir -name "score.txt" | grep "${words}"` )

for f in ${scores[@]}; do
  grep "PER:" $f | tail -1 | awk '{print $2}' |\
    sed 's/[()%]//g' | awk -v var=$f '{print var": "$0}' 
done >> ${dir}/symb_er.txt

[ -f ${dir}/str_er.txt ] && rm ${dir}/str_er.txt
for f in ${scores[@]}; do
  grep "WER:" $f | tail -1 | awk '{print $2}' |\
    sed 's/[()%]//g' | awk -v var=$f '{print var": "$0}' 
done >> ${dir}/str_er.txt

grep "${words}" ${dir}/symb_er.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/symb_er.txt
grep "${words}" ${dir}/str_er.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/str_er.txt

exit 0;
