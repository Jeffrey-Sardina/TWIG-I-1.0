#!/bin/bash

# # constants for TWIG-I from-scratch baselines
# sampler='simple'
# hyp_validation_mode=0
# batch_size_test=64

for epochs in 20
do
    # # baselines
    # npp=500
    # lr=5e-3
    # batch_size=128
    # ./TWIG-I_pipleline.sh \
    #     "DBpedia50" \
    #     $epochs \
    #     $npp \
    #     $lr \
    #     $sampler \
    #     $hyp_validation_mode \
    #     $batch_size \
    #     $batch_size_test \
    #     "dbpedia-from-scrach-e${epochs}"

    # npp=100
    # lr=5e-3
    # batch_size=64
    # ./TWIG-I_pipleline.sh \
    #     "CoDExSmall" \
    #     $epochs \
    #     $npp \
    #     $lr \
    #     $sampler \
    #     $hyp_validation_mode \
    #     $batch_size \
    #     $batch_size_test \
    #     "codex-from-scrach-e${epochs}"

    # FB -- from just 10 or 20 epochs
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_2302135203951868 \
    #     ${epochs} \
    #     TWIG-I/overrides/from-scratch/override-fb-e0.json \
    #     fb-fom-scratch-e${epochs}

    # # WN -- from just 10 or 20 epochs
    # # note we use the 1st checkpoint ID that was trained for 60 epochs
    # # since 20 < 60
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     ${epochs} \
    #     TWIG-I/overrides/from-scratch/override-wn-e0.json \
    #     wn-from-scratch-e${epochs}


    # # # FB -- we use the model version as it was after 100 epochs
    # optimal_config="override-CoDExSmall-e10-bs64-lr5e-3-npp100.json"
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_2302135203951868 \
    #     100 \
    #     TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
    #     fb-to-codex-e${epochs}-opt_hyp

    # optimal_config="override-DBpedia50-e10-bs128-lr5e-4-npp30.json"
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_2302135203951868 \
    #     100 \
    #     TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
    #     fb-to-dbpedia-e${epochs}-opt_hyp

    # WN -- this is from a resuimed model (epoch 40 was resumed from a previous 60
    # epochs) so this is still with a totl of 100 epochs of pretraining
    # optimal_config="override-CoDExSmall-e10-bs128-lr5e-3-npp500.json"
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     40 \
    #     TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
    #     wn-to-codex-e${epochs}-opt_hyp

    # optimal_config="override-DBpedia50-e10-bs64-lr5e-4-npp30.json"
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     40 \
    #     TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
    #     wn-to-dbpedia-e${epochs}-opt_hyp


    # # full training on dbpedia and codex
    # npp=100
    # lr=5e-3
    # batch_size=64
    # ./TWIG-I_pipleline.sh \
    #     "CoDExSmall" \
    #     100 \
    #     $npp \
    #     $lr \
    #     $sampler \
    #     $hyp_validation_mode \
    #     $batch_size \
    #     $batch_size_test \
    #     "dbpedia-from-scrach-e100"

    # npp=100
    # lr=5e-3
    # batch_size=128
    # ./TWIG-I_pipleline.sh \
    #     "DBpedia50" \
    #     100 \
    #     $npp \
    #     $lr \
    #     $sampler \
    #     $hyp_validation_mode \
    #     $batch_size \
    #     $batch_size_test \
    #     "dbpedia-from-scrach-e100"

    optimal_config="override-WN18RR-e20-bs128-lr5e-4-npp500.json"
    ./TWIG-I_from_checkpoint.sh \
        chkpt-ID_2302135203951868 \
        100 \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        fb-to-wn-e${epochs}-opt_hyp

    optimal_config="override-FB15k237-e20-bs128-lr5e-4-npp100.json"
    ./TWIG-I_from_checkpoint.sh \
        chkpt-ID_5336826154703196 \
        40 \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        wn-to-fb-e${epochs}-opt_hyp
done
