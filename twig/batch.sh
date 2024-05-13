#!/bin/bash

# TWIG-I hyp search
epochs=20
sampler='simple'
hyp_validation_mode=1
batch_size_test=64
tag="negfix"

# todo -- make sure batch size test does not affect eval!

for dataset in DBpedia50 CoDExSmall OpenEA
do
    for npp in 30 100 500
    do
        for lr in 5e-3 5e-4 5e-5
        do
            for batch_size in 64 128 256
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


# training TWIG-I -- from mrr: 0.2627256214618683
# epochs=200
# sampler='simple'
# hyp_validation_mode=0
# batch_size_test=64

# dataset="DBpedia50"
# npp=500
# lr=5e-3
# batch_size=128
# ./TWIG-I_pipleline.sh \
#     $dataset \
#     $epochs \
#     $npp \
#     $lr \
#     $sampler \
#     $hyp_validation_mode \
#     $batch_size \
#     $batch_size_test \
#     $tag


# # Finetuning TWIG-I from DBpedia50 to FB15k-237
# ./TWIG-I_from_checkpoint.sh \
#     chkpt-ID_8726812426797882 \
#     200 \
#     TWIG-I/override-fb.json \
#     finetune-fb-10e

# ./TWIG-I_from_checkpoint.sh \
#     chkpt-ID_8726812426797882 \
#     200 \
#     TWIG-I/override-wn.json \
#     finetune-wn-10e
