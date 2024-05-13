#!/bin/bash

# for epochs in 10
# do
#     for dataset in CoDExSmall DBpedia50 WN18RR FB15k237
#     do
#         for batch_size in 64 128 256
#         do
#             for lr in 5e-3 5e-4 5e-5
#             do
#                 for npp in 30 100 500
#                 do
#                     # write the json config
#                     hyp_condig_file="TWIG-I/overrides/hyp-search/override-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp.json"
#                     echo "{" > $hyp_condig_file
#                     echo "\"epochs\": $epochs," >> $hyp_condig_file
#                     echo "\"dataset_names\": [\"$dataset\"]," >> $hyp_condig_file
#                     echo "\"batch_size\": $batch_size," >> $hyp_condig_file
#                     echo "\"lr\": $lr," >> $hyp_condig_file
#                     echo "\"npp\": $npp," >> $hyp_condig_file
#                     echo "\"hyp_validation_mode\": 1" >> $hyp_condig_file
#                     echo "}" >> $hyp_condig_file
#                 done
#             done
#         done
#     done
# done
# exit # we don't want to generate json and run in the same run
# # just for implicity

# for epochs in 10
# do
#     for dataset in CoDExSmall DBpedia50
#     do
#         for batch_size in 64 128 256
#         do
#             for lr in 5e-3 5e-4 5e-5
#             do
#                 for npp in 30 100 500
#                 do
#                     # write the json config
#                     hyp_condig_file="TWIG-I/overrides/hyp-search/override-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp.json"
                    
#                     # FB -- we use the model version as it was after 100 epochs
#                     ./TWIG-I_from_checkpoint.sh \
#                         chkpt-ID_2302135203951868 \
#                         100 \
#                         $hyp_condig_file \
#                         fb-to-$dataset-e${epochs}-bs${batch_size}-lr$lr-npp$npp

#                     # WN -- this is from a resuimed model (epoch 40 was resumed from a previous 60
#                     # epochs) so this is still with a totl of 100 epochs of pretraining
#                     ./TWIG-I_from_checkpoint.sh \
#                         chkpt-ID_5336826154703196 \
#                         40 \
#                         $hyp_condig_file \
#                         wn-to-$dataset-e${epochs}-bs${batch_size}-lr$lr-npp$npp
#                 done
#             done
#         done
#     done
# done

for epochs in 10
do
    for batch_size in 128
    do
        for lr in 5e-3 5e-4 5e-5
        do
            for npp in 30 100 500
            do
                # FB -- we use the model version as it was after 100 epochs
                # we finetune to WN18RR
                dataset="WN18RR"
                hyp_condig_file="TWIG-I/overrides/hyp-search/override-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp.json"
                ./TWIG-I_from_checkpoint.sh \
                    chkpt-ID_2302135203951868 \
                    100 \
                    $hyp_condig_file \
                    fb-to-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp

                # WN -- this is from a resuimed model (epoch 40 was resumed from a previous 60
                # epochs) so this is still with a totl of 100 epochs of pretraining
                # we finetune to FB15k237
                dataset="FB15k237"
                hyp_condig_file="TWIG-I/overrides/hyp-search/override-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp.json"
                ./TWIG-I_from_checkpoint.sh \
                    chkpt-ID_5336826154703196 \
                    40 \
                    $hyp_condig_file \
                    wn-to-$dataset-e$epochs-bs${batch_size}-lr$lr-npp$npp
            done
        done
    done
done

    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_2302135203951868 \
    #     100 \
    #     TWIG-I/override-dbpedia-e${epochs}.json \
    #     fb-to-dbpedia-e${epochs}

    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_2302135203951868 \
    #     100 \
    #     TWIG-I/override-wn-e${epochs}.json \
    #     fb-to-wn-e${epochs}

    # # WN -- this is from a resuimed model (epoch 40 was resumed from a previous 60
    # # epochs) so this is still with a totl of 100 epochs of pretraining
    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     40 \
    #     TWIG-I/override-codex-e${epochs}.json \
    #     wn-to-codex-e${epochs}

    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     40 \
    #     TWIG-I/override-dbpedia-e${epochs}.json \
    #     wn-to-dbpedia-e${epochs}

    # ./TWIG-I_from_checkpoint.sh \
    #     chkpt-ID_5336826154703196 \
    #     40 \
    #     TWIG-I/override-fb-e${epochs}.json \
    #     wn-to-fb-e${epochs}
# done
