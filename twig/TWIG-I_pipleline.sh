#!/bin/bash

datasets=$1 #delimit by -
epochs=$2
npp=$3
lr=$4 #5e-4 works for UMLS; 5e-5 is good for Kinships; 
sampler=$5 #simple or vector
hyp_validation_mode=$6
batch_size=$7
batch_size_test=$8
tag=$9

version="0"
normalisation="zscore"
use_train_filter='0' #was 0
use_valid_and_test_filters='1' #was 0

export TWIG_CHANNEL=3

cd TWIG-I/
python ../utils/twig_alerts.py "re-eval pipeline configured with $num_processes processes" &
python ../utils/twig_alerts.py "if memory issues occur, please restart with fewer processes" &
out_file="rec_v${version}_${datasets}_norm-${normalisation}_e${epochs}-lr${lr}_bs${batch_size}_npp${npp}_${sampler}-sampler_filter-code${use_train_filter}${use_valid_and_test_filters}_tag-${tag}.log"

start=`date +%s`
python -u run_exp.py \
    $version \
    $datasets \
    $epochs \
    $lr \
    $normalisation \
    $batch_size \
    $batch_size_test \
    $npp \
    $use_train_filter \
    $use_valid_and_test_filters \
    $sampler \
    $hyp_validation_mode \
    &> $out_file

end=`date +%s`
runtime=$((end-start))
echo "Experiments took $runtime seconds on " &>> $out_file
python ../utils/twig_alerts.py "I have just finished run $run_name" &
python ../utils/twig_alerts.py "Experiments took $runtime seconds" &
