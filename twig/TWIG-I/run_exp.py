import sys
from load_data import do_load, get_adj_data
from twig_nn import *
from trainer import run_training
import glob
import os
import torch
from utils import get_triples, calc_graph_stats, get_triples_by_idx
from negative_sampler import Optimised_Negative_Sampler
import pykeen
import random
import pickle

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
================
Module Functions
================
'''
def load_nn(version):
    ''''
    load_nn() loads a PyTorch model from twig_nn.py. 

    The arguments it accepts are:
        - version (int): the version of the NN to load. CUrrently, 0 and 1
          are supported.

    The values it returns are:
        - model (torch.nn.Module): the PyTorch NN Model object containing
          TWIG's neural architecture.
    '''

    print('loading NN')
    n_local = 22 # 6 if in 1hop only mode, else 22
    if version == 0:
        model = TWIG_KGL_v0(
            n_local=n_local
        )
    elif version == 1:
        model = TWIG_KGL_v1(
            n_local=n_local
        )
    else:
        assert False, f"Invald NN version given: {version}"
    print("done loading NN")
    return model

def load_dataset(
        dataset_names,
        normalisation,
        batch_size,
        batch_size_test
    ):
    ''''
    load_dataset() loads all training, testing, and validation data
    assocaited with a single dataset. The dataset is given by name and
    must be one of the datasets defined in PyKEEN (https://github.com/pykeen/pykeen#datasets)

    The dataset load uses function implementation in load_data.py and also
    performs all needed preprocessing, normalisation, etc such that the returned
    data can be directly used in learning.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as
          strings
        - normalisation (str): A string representing the method that should
          be used to normalise all data. Currently, "zscore" and "minmax" are
          implemented; "none" can be given to not use normalisation. Defaults
          to "none".
        - batch size (int): the batch size to use during training. It is needed
          here to construct the dataloader objects that are returned, as PyTorch
          bundles batch size into the dataloader object.
        - batch size_test (int): the batch size to use during testing and
          validation. It is needed here to construct the dataloader objects
          that are returned, as PyTorch bundles batch size into the dataloader
          object.

    The values it returns are:
        - dataloaders (dict str -> str -> torch.utils.data.DataLoader): a dict
          that maps a training split ("train", "test", or "valid") and a dataset
          (with a name as in dataset_names) to a DataLoader that can be used to
          load batches for that dataset on that training split. An example of
          accessing its data could be dataloaders["train"]["UMLS"], which would
          return the training dataloader for the UMLS dataset.
        - norm_basis (torch.Tensor): the Tensor whose values for a basis for
          normalisation. It's values are the used to compute the normalisation
          parameters (such as min and max values for minmax normalisation). As an
          example, when normalising the test set, you want to normalise it
          relative to the values **in the train set** so as to avoid data
          leakage. In this case the norm_basis would be be training tensor. It
          is returned so that the negative sampler can use it to normalise the
          values of generated negative triples the same way that all other data
          was normalised.
    '''

    print('loading dataset')

    dataloaders, norm_funcs = do_load(
        dataset_names,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test
    )
    print("done loading dataset")
    return dataloaders, norm_funcs

def load_filters(
        dataset_names,
        use_train_filter,
        use_valid_and_test_filters
    ):
    '''
    load_filters() loads the filters that should be used for a given dataset.
    Filters are split by their purpose (what phase they are ussed at; i.e. train,
    test, or valid). In the current implmenetation,
    - the train filters consist of all training triples
    - the valid filters consist of all training and validation triples
    - the tetst filters consist of all training, validation, and testing triples
      (i.e. all triples).

    This is done to ensure that the filters themsevles do not allow for test
    or validation leakage during model training or creation.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as
          strings
        - use_train_filter (bool): True if negative samples generated during
          training should be filtered, False if they should not be filtered
        - use_valid_and_test_filters (bool): True if negative samples generated
          during validation and testing should be filtered, False if they
          should not be filtered

    The values it returns are:
        - filters (dict str -> str -> dict): a dict that maps a dataset name
          (str) and a training split name (i.e. "train", "test", or "valid") to 
          a dictionary describing the triples to use i filtering. To be exact,
          this second triples_dict has the structure
          (dict str -> int -> tuple<int,int,int>). It maps first the training
          split name to the triple index, and that trtrriple index maps to a
          single triple expressed as (s, p, o) with integral representations
          of each triple element.
    '''
    print('loading filters')

    filters = {dataset_name:{} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        pykeen_dataset = pykeen.datasets.get_dataset(dataset=dataset_name)
        triples_dicts = get_triples(pykeen_dataset)
        filters[dataset_name] = {
            'train': set(triples_dicts['train']) if use_train_filter else set(), #train triples
            'valid': set(triples_dicts['train'] + triples_dicts['valid']) if use_valid_and_test_filters else set(), #all train and valid triples
            'test': set(triples_dicts['all']) if use_valid_and_test_filters else set() #all train, valid, and test triples
        }
    print('done loading filters')
    return filters

def load_negative_samplers(
        dataset_names,
        filters,
        normalisation,
        norm_funcs,
        sampler_type,
    ):
    '''
    load_negative_samplers() loads all negative samplers. Note that negative
    samplers are dataset-specific, so when multiple datasets are being learned
    on, multiple negative samplers must be created.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as
          strings
        - filters (dict str -> str -> dict): a dict that maps a dataset name
          (str) and a training split name (i.e. "train", "test", or "valid") to 
          a dictionary describing the triples to use i filtering. To be exact,
          this second triples_dict has the structure
          (dict str -> int -> tuple<int,int,int>). It maps first the training
          split name to the triple index, and that trtrriple index maps to a
          single triple expressed as (s, p, o) with integral representations
          of each triple element.
        - normalisation (string): A strong representing the methos that should
          be used to normalise all data. Currently, "zscore" and "minmax" are
          implemented; "none" can be given to not use normalisation. Defaults
          to "none".
        - norm_basis (torch.Tensor): the Tensor whose values for a basis for
          normalisation. It's values are the used to compute the normalisation
          parameters (such as min and max values for minmax normalisation).
        - sampler_type (str) a string defining what type of negative sampler is
          desired. Current options are 'simple' and 'vector'; see
          negative_sampler.py for details.
        - raw_X_train_pos (torch.Tensor): the tensor containing all data in the
          dataloader as a single object. This is needed for use with the
          VectorNegativeSampler to build a triple ID -> triple feature vector
          map.

    The values it returns are:
        - negative_samplers (dict str -> Negative_Sampler): a dict that
          maps a dataset name to the negative sampler associated with that
          dataset. 
    '''
    print('loading negative samplers')

    negative_samplers = {}
    for dataset_name in dataset_names:
        pykeen_dataset = pykeen.datasets.get_dataset(dataset=dataset_name)
        triples_dicts = get_triples(pykeen_dataset)
        graph_stats = calc_graph_stats(triples_dicts, do_print=False)
        triples_map = get_triples_by_idx(triples_dicts, 'all')
        ents_to_triples = get_adj_data(triples_map)
        simple_sampler = Optimised_Negative_Sampler(
            filters[dataset_name],
            graph_stats,
            triples_map=triples_map,
            ents_to_triples=ents_to_triples,
            normalisation=normalisation,
            norm_func=norm_funcs[dataset_name],
            dataset_name=dataset_name,
            prefilter=False
        )
        if sampler_type == 'simple':
            negative_samplers[dataset_name] = simple_sampler
        else:
            assert False, f'Unknown negative sampler type requested: {sampler_type}. Must be "vector" or "simple".'
    print('done loading negative samplers')
    return negative_samplers

def train_and_eval(
        model,
        training_dataloaders,
        testing_dataloaders,
        valid_dataloaders,
        epochs,
        lr,
        npp,
        negative_samplers,
        verbose=True,
        model_name_prefix='model',
        checkpoint_dir='checkpoints/',
        checkpoint_every_n=5,
        valid_every_n=10
    ):
    ''''
    train_and_eval() runs the training and evaluation loops on the data, and
    prints all results.

    The arguments it accepts are:
        - model (torch.nn.Module): the PyTorch NN Model object containing
          TWIG's neural architecture.
        - training_dataloaders (dict str -> torch.utils.data.DataLoader): a dict
          that maps a dataset name string to a DataLoader that can be used to
          load batches for that dataset on that training split. An example of
          accessing its data could be training_dataloaders["UMLS"], which would
          return the training dataloader for the UMLS dataset.
        - testing_dataloaders (dict str -> torch.utils.data.DataLoader): a dict
          that maps a dataset name string to a DataLoader that can be used to
          load batches for that dataset on that testing split. An example of
          accessing its data could be testing_dataloaders["UMLS"], which would
          return the testing dataloader for the UMLS dataset.
        - valid_dataloaders (dict str -> torch.utils.data.DataLoader): a dict
          that maps a dataset name string to a DataLoader that can be used to
          load batches for that dataset on that validation split. An example of
          accessing its data could be valid_dataloaders["UMLS"], which would
          return the validation dataloader for the UMLS dataset.
        - epochs (int): the number of epochs to train for
        - lr (float): the learning rate to use during training
        - npp (int): the number of negative samples to generate per positive triple
          during training.
        - negative_samplers (dict str -> Negative_Sampler): a dict that
          maps a dataset name to the negative sampler associated with that
          dataset.
        - verbose (bool): whether or not all information should be output. If
          True, TWIG will be run in verbose mode, whhich means more information
          will be printed.
        - model_name_prefix (str): the prefix to prepend to the model name when
          saving checkpoints (currently unused, as checkpoints are not saved)
        - checkpoint_dir (str): the directory in which to save checkpoints 
          (currently unused, as checkpoints are not saved)
        - checkpoint_every_n (int): the interval of epochs after which a
          checkpoint should be saved during training. (currently unused, as
          checkpoints are not saved)
        - valid_every_n (int): the interval of epochs after which TWIG should
          be evaluated on its validation dataset.

    The values it returns are:
        - No values are returned.
    '''
    print("running training and eval")
    run_training(
        model,
        training_dataloaders,
        testing_dataloaders,
        valid_dataloaders,
        epochs=epochs,
        lr=lr,
        npp=npp,
        negative_samplers=negative_samplers,
        verbose=verbose,
        model_name_prefix=model_name_prefix,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=checkpoint_every_n,
        valid_every_n=valid_every_n
    )
    print("done with training and eval")

def main(
        version,
        dataset_names,
        epochs,
        lr,
        normalisation,
        batch_size,
        batch_size_test,
        npp,
        sampler_type,
        use_train_filter,
        use_valid_and_test_filters,
        hyp_validation_mode,
        preexisting_model=None
    ):
    '''
    main() coordinates all of the major modules of TWIG, loads and prepares
    data, the NN, filters, and model hyparapmeters, (etc) and then call the
    training and evaluation loop. Since all of these functionalities are
    implemented in various modules, main() is more or less just a list of
    function calls that coordinate them all.

    The arguments it accepts are:
        - version (int): the version of the Neural Network to run. Default: 0
        - dataset_names (list of str): the name (or names) of datasets to train and
          test on. If multiple are given, their training triples will be concatenated
          into a single tensor that is trained on, and testing will be done individually
          on each testing set of each dataset.
        - epochs (int): the number of epochs to train for
        - lr (float): the learning rate to use during training
        - normalisation (str): the normalisation method to be used when loading data
          (and when created vectorised forms of negatively sampled triples).
          "zscore", "minmax", and "none" are currently supported.
        - batch_size (int): the batch size to use while training.
        - batch_size_test (int): the batch size to use while testing /
          validating.
        - npp (int): the number of negative samples to generate per positive triple
          during training
        - use_train_filter (bool): True if negative samples generated during training
          should be filtered, False if they should not be filtered
        - use_valid_and_test_filters (bool): True if negative samples generated during
          validation and testing should be filtered, False if they should not be filtered
        - sampler_type (str) a string defining what type of negative sampler is
          desired. Current options are 'simple' and 'vector'; see
          negative_sampler.py for details.
        - hyp_validation_mode (bool): a bool that is True if TWIG should be
          run in hyperparameter validation mode (i.e. not doing validating
          during training and outputing results on the validation set rather
          than the test set); False if it should be run in regular mode (eval).
    
    The values it returns are:
        - results (dict str -> str -> float): the results output from
          train_and_eval(). The first key is the dataset name, and the second
          key is the name of the metric. The value contained is a float value
          of the specified metric on the specified dataset. An example of
          accessing its data could thus be results['UMLS']['mrr'], which will
          return the MRR value TWIG acheived on the UMLS dataset.
    '''
    checkpoint_dir = 'checkpoints/'
    model_name_prefix = 'chkpt-ID_' + str(int(random.random() * 10**16))
    if preexisting_model is not None:
        print('Using provided pre-existing model')
        model = preexisting_model
    else:
        print('Creating a new model from scratch')
        model = load_nn(version)

    # save hyperparameter settings
    checkpoint_config_name = os.path.join(checkpoint_dir, f'{model_name_prefix}.pkl')
    with open(checkpoint_config_name, 'wb') as cache:
        to_save = {
            "version": version,
            "dataset_names": dataset_names,
            "epochs": epochs,
            "lr": lr,
            "normalisation": normalisation,
            "batch_size": batch_size,
            "batch_size_test": batch_size_test,
            "npp": npp,
            "sampler_type": sampler_type,
            "use_train_filter": use_train_filter,
            "use_valid_and_test_filters": use_valid_and_test_filters,
            "hyp_validation_mode": hyp_validation_mode
        }
        pickle.dump(to_save, cache)

    dataloaders, norm_funcs = load_dataset(
        dataset_names,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test
    )

    filters = load_filters(
        dataset_names,
        use_train_filter=use_train_filter,
        use_valid_and_test_filters=use_valid_and_test_filters
    )
    negative_samplers = load_negative_samplers(
        dataset_names,
        filters,
        normalisation,
        norm_funcs,
        sampler_type=sampler_type,
    )

    if hyp_validation_mode:
        print('Running in hyperparameter evaluation mode')
        print('TWIG will be evaulaited on the validation set')
        print('and will not be tested each epoch on the validation set')
        valid_every_n = -1
        data_to_test_on = dataloaders['valid']
    else:
        valid_every_n = -1 #was 1
        data_to_test_on = dataloaders['test']

    train_and_eval(
        model,
        dataloaders['train'],
        data_to_test_on,
        dataloaders['valid'],
        epochs,
        lr,
        npp,
        negative_samplers=negative_samplers,
        verbose=True,
        model_name_prefix=model_name_prefix,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=5,
        valid_every_n=valid_every_n
    )

if __name__ == '__main__':
    '''
    This section just exists to collect command line arguments and pass them to
    the main() function. Command line arguments are converted to the correct
    type (such as int or boolean) before being passed on.

    The command-line arguments accepted, and their meaning, is described below
        - version: the version of the Neural Network to run. Default: 0
        - dataset_names: the name (or names) of datasets to train and test on.
          If multiple are given, their training triples will be concatenated
          into a single tensor that is trained on, and testing will be done individually on each testing set of each dataset. When multiple datasets are given, they should be delimited by "-", as in "UMLS-DBpedia50-Kinships"
        - epochs: the number of epochs to train for
        - lr = the learning rate to use during training
        - normalisation: the normalisation method to be used when loading data
          (and when created vectorised forms of negatively sampled triples).
          "zscore", "minmax", and "none" are currently supported.
        - batch_size: the batch size to use while training
        - npp: the number of negative samples to generate per positive triple
          during training
        - use_train_filter: "1" if negative samples generated during training
          should be filtered, "0" if they should not be filtered
        - use_valid_and_test_filters: "1" if negative samples generated during
          validation and testing should be filtered, "0" if they should not be filtered
    
    Once all data is collected, main() is called with it as arguments.
    '''
    print(sys.argv)
    version = int(sys.argv[1])
    dataset_names = sys.argv[2].split('-')
    epochs = int(sys.argv[3])
    lr = float(sys.argv[4])
    normalisation = sys.argv[5]
    batch_size = int(sys.argv[6])
    batch_size_test = int(sys.argv[7])
    npp = int(sys.argv[8])
    use_train_filter = sys.argv[9] == '1'
    use_valid_and_test_filters = sys.argv[10] == '1'
    sampler_type = sys.argv[11]
    if len(sys.argv) > 12:
        hyp_validation_mode = sys.argv[12] == '1'
    else:
        hyp_validation_mode = False

    main(
        version,
        dataset_names,
        epochs,
        lr,
        normalisation,
        batch_size,
        batch_size_test,
        npp,
        sampler_type,
        use_train_filter,
        use_valid_and_test_filters,
        hyp_validation_mode,
    )
