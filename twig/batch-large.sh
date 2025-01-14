#!/bin/bash

# TWIG-I hyp search
epochs=20
sampler='simple'
hyp_validation_mode=1
batch_size_test=64
tag="negfix"

# todo -- make sure batch size test does not affect eval!

dataset=FB15k237
batch_size=128
lr="5e-4"
npp=500
./TWIG-I_pipleline.sh \
    $dataset \
    $epochs \
    $npp \
    $lr \
    $sampler \
    $hyp_validation_mode \
    $batch_size \
    $batch_size_test \
    $tag

for dataset in FB15k237
do
    for batch_size in 128
    do
        for lr in 5e-5
        do
            for npp in 30 100 500
            do
                ./TWIG-I_pipleline.sh \
                    $dataset \
                    $epochs \
                    $npp \
                    $lr \
                    $sampler \
                    $hyp_validation_mode \
                    $batch_size \
                    $batch_size_test \
                    $tag
            done
        done
    done
done

for dataset in WN18RR
do
    for batch_size in 128
    do
        for lr in 5e-3 5e-4 5e-5
        do
            for npp in 30 100 500
            do
                ./TWIG-I_pipleline.sh \
                    $dataset \
                    $epochs \
                    $npp \
                    $lr \
                    $sampler \
                    $hyp_validation_mode \
                    $batch_size \
                    $batch_size_test \
                    $tag
            done
        done
    done
done
