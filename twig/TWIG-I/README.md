# TWIG-I 1.0


## Architecture of the Library
TWIG itself is a sort of library for learning to solve the LP task without embeddings. It does this using pre-defined structural features (approximately 30) that describe the local degrees and predicate frequencies in a 1-hop distance around a triple, as well as (potentially) global graph structure. 

As much as is possible, all functionality is placed into its own module files. These are
- **load_data.py** -- contains all logic for loading, pre-processing, and normalising data used for training, testing, and validation.
- **loss.py** -- contains implementations of all loss functions
- **negative_sampler.py** -- contains all implementations of negative samplers; i.e. the part of TWIG that creates randomly corrupted triples to uses as negative examples during training.
- **run_exp.py** -- contains all data needed to run TWIG from the command line, including accepting command line arguments, and orchestrating all other modules to make TWIG work.
- **trainer.py** -- contains implementations for processing batches, generating negatives, and calculating losses during training and testing. It also reports all test statistics and reports on validation data during training.
- **twig_alerts.py** -- contains telemetry implementations to allow reporting data continuously to a discord server.
- **twig_nn.py** -- contains the TWIG neural network implementation -- i.e. the forward pass in PyTorch.
- **utils.py** -- contains implementations of vavious utility functions, mostly for for loading and processing graph data and putting it into a form TWIG can recognise.

The next few sections will go through each of these in detail.


## Data Loading (load_data.py)
The load_data.py file's primary interface is the do_load(.) function. It takes in a list of dataset names (these must be dadtasets defined in PyKEEN; see here: https://github.com/pykeen/pykeen?tab=readme-ov-file#datasets) and returns a dictionary of all dataloaders. This dictionary has the structure:
dict -> 
    "train" -> 
        dataset_name ->
            PyTorch dataloader
    "test" -> 
        dataset_name ->
            PyTorch dataloader
    "valid" -> 
        dataset_name ->
            PyTorch dataloader

Optionally, the normalization method (default: no normalization) and batch size (default: 4096) can be specified.

The rest of the functions in do_load exist entirely to support this loading, which includes:
- obtaining raw data
- calculating structural data for each triple
- vectorising structural features
- validating the data integrity
- normalsing data (if requested)
- convertng the data into PyTorch dataloaders for use in training and evaluation

Looking at this from a call stack perspective, we see the following dependencies:
- we load the dataset from PyKEEN and use some functions in utils (namely get_triples and calc_graph_stats to convert it into a directly usable form)
- we call calc_split_data with this dataset information. This function serves to load data for one "split" of the data -- i.e. either training, testing, or validation.
- in calc_split_data, we use some util functions to calculate graph structural features and then call call get_twm_data_augment to convert them into a vectorised form
- get_twm_data_augment uses get_spo_data as a triple-by-triple subroutine to extract the features for a specific triple

Looking at this from a bottom-up narrative perspective, we see the following overall programmatic structure:
- Our goal is to creature vectorised representations of every triple based on its structure alone. The function that does this is get_spo_data.
- the data from all individual triples is taken aggregated in get_twm_data_augment to produce a single, large dataframe of triple data.
- all of this data is then processed into a single "split" of the dataset (i.e. the training, testing, or validation split) in calc_split_data
- finally, do_load takes all of the data splits and aggregates them into a single dictionry object that is returned. This contains all loaded data.

For a detailed description of the functions, their inputs, and their outputs, see the comments on them in the source file.


## Loss Function Definitions (loss.py)
The loss.py file contains all loss function definitions used to help TWIG learn. Every loss function has a common signature. The arguments it takes, in turn, are:
- score_pos -- a tensor with the scores of positive triples
- score_neg -- a tensor with the scores of negative triples
- reduction  -- how to reduce the loss values (default is mean, but sum is also implemented)
- **kwargs -- keyword arguments for other loss functions with hyperparameters or specific settings

Note that due to negative sampling (or taking all possible negative) resulting in many negatives being generated for one positive triple, this could create asymetry between score_pos and score_neg in size. **However, this is not the case here**. We pre-process the scores in trainer.py such that for every index i in the negative score tensor, index i in the positive tensor contains the score for the positive triple from which the negative was generated. socre_pos, as thus, has many repeats, but can be used directly and without modification for pairwise operations.

See comments on each function for notes on how it works, how it is implemented, and what the motivation for its use is.


## Negative Sampling (negative_sampler.py)
The negative_sampler.py file contains the Simple_Negative_Sampler class, which is currently the only method for negative sampling supported by TWIG. It implements what the literature calls "random negative sampling" in which a triple (s,p,o) is corrupted to (s',p,o) or (s,p,o') where both corruptions are chosen randomly from the list of all possible entities.

Filtering known true statements is implemented, but no other options are. Notably, this means there is no option to only corrupt the subject or only corrupt the object. Note also that since the negative sampler collects information on a KG, in TWIG negative samplers are constructed for a specific dataset and can only generate negatives for one dataset's data.

The negative sampler class has three main functions. Perhaps the most important to understand (and le least immediately obvious) is the init function.

**The init Function**
init takes as arguments several parameters:
- filters: a dictionary with "train", "test", and "valid" as keys that maps to filters that should be used on each data split. 
- graph_stats: a global summary of graph structure, needed to calculate the features for each generated negative triple
- triples_map: a map relating a triple ID to its s, p, and o items. Since the negative sampler produces corruptions from a given triple ID, this is needed to determine the elements of the triple in question. 
- ents_to_triples: a dictionary mapping every entity in the KG to all triples that contain it. It is currently only used to extract the set of all entities; the mapping itself is not used.
- normalisation: the method that was used to normalise the data when loading (as in load_data.py).
- norm_basis: the basis that was used to normalise data, returned from the normalisation function of load_data.py. This is, for example, the tensor of all training data that can be used to find the min and max values for min-max normalisation. (No, this is not the most efficient implementation. But it's just what I have in there for now. Later, it will be replaced with much smaller vectors of just the parameters like min and max) needed to compute the normalisation)

It then saves all of this data for use later in generative negatives.

**The get_negatives Function**
get_negatives takes as arguments several parameters:
- purpose: 'train', 'test', or 'valid' -- the purpose / phase of training and evaluation for which these negatives will be used. This is used to determine which filter to apply to generated negatives, if filters are given.
- triple_index: the index of the triple whose corruptions are wanted.
- npp: the number of **n**egatives **p**er **p**ositive that should be generated. In other words, this is the number of negative examples that a ccall to get_negatives should return for the single positive triple input to it.
- allow_upsampling: (default True), whether or not up-samplig should be allowed if more negative are requested than corruptions are available. For example, if only 10 corruptions are possible (perhaps to due filters on other possible ones) but 15 negatives are desired, then a total of 10, not 10, corruptions will be randomly selected from the pool of all possible corruptions (with replacement). Note that in the current implementation, it is possible that not all 10 possible corruptions are returned by chance, since the choosing is purely random.

It uses this to do full-random negative sampling in which half of all generated negatives corrupt the subject and half corrupt the object. 

**The get_batch_negatives Function**
get_batch_negatives takes as argument several parameters:
- purpose: see above
- triple_idxs: a iterable collection of triple IDs. For what triple IDs represent, see the triple_index parameter above.
- npp: see above

It exists for one purpose: to run get_negatives on multiple triple IDs at once, and to return all of the concatenated results. It is used when an entire batch of triples are given, to generate corruptions for the entire batch all at once.


## Neural Network Definition (twig_nn.py)
The twig_nn.py file contains the implementation of all of TWIG's neural network layers, and defines its neural architecture. Each NN contains a common interface for creation, which onyl requires two parameters:
- n_global: the number of global (graph-level) structural feature columns in the input data
- n_local: the number of local structural features (around a triple)

All also define the standard PyTorch forward pass. Currently there are two NN versions implemented:
- V0, which is made to learn a single KG (and does not use global features in any way)
- V1, which is made to learn multiple KGs at the same time (and makes use of both global and local features in learning)

Note that global features are of no use when learning only one KG, since the global feature representation will contain the exact same data for all samples, since it is a representation of the same dataset.


## Other Files (utils.py and twig_alerts.py)
utils.py contains a variety of general utility functions that assist in the various aspects of TWIG, mostly used when loaded or processing data (in load_data.py and also in negative_sampler.py). See its functions for documentation of how they work.

## Training and Evaluating the Model (trainer.py)
The functionality of trainer.py is centred around run_training, which more or less orchestrates everything. It takes in dadtaloaders and hyparameters, configures and runs the training loop, and implements validation checks (on the validation data set) and final evaluation for testing. 

Training and evaluation are managed in the train_epoch(.) and test(.) functions respectively. Both of these functions are largely the same in their input and the procedures they run. Essentially all of the real logic for computation and learning is pushed into do_batch, which is a sub-routine that both call. do_batch is the part that actually extracts batch data, generates negatives, computes positive and negative triple scores with a pass through the model, and computes loss based off of these scores.

In evaluation, test(.) uses the calc_ranks function to turn scores of triples (positive and negative) into the rank of each positive triple among its negative triples. It the calculates evaluation metrics, (MR, MRR, and H@K) on these ranks and outputs the results.

You can think of trainer.py as the part of TWIG that actually does the heavy lifting -- the rest of TWIG is created to provide it with the data it needs to do that.


## Putting it all Together (run_exp.py)
The run_exp.py file is TWIG's command line interface with the user. It sets a single, constant random seed for reproducibility collects command line arguments, loads data and initialises all modules, and finally calls the trainer to proceed with training. 

The command line arguments it accepts are (in order):
- version: the version of the Neural Network to run. Default: 0
- dataset_names: the name (or names) of datasets to train and test on. If multiple are given, their training triples will be concatenated into a single tensor that is trained on, and testing will be done individually on each testing set of each dataset. When multiple datasets are given, they should be delimited by "-", as in "UMLS-DBpedia50-Kinships"
- epochs: the number of epochs to train for
- lr = the learning rate to use during training
- normalisation: the normalisation method to be used when loading data (and when created vectorised forms of negatively sampled triples). "zscore", "minmax", and "none" are currently supported.
- batch_size: the batch size to use while training
- npp: the number of negative samples to generate per positive triple during training
- use_train_filter: "1" if negative samples generated during training should be filtered, "0" if they should not be filtered
- use_valid_and_test_filters: "1" if negative samples generated during validation and testing should be filtered, "0" if they should not be filtered

These command line arguments are then directly passed to the main() function, which handles all of the logic and actions of TWIG. 

All functions in run_exp are implemented as independent modules, calling one aspect of TWIG (such as load_data.py), providing logging information (and telemetry) on the progress of all of these tasks, and validating some inputs.

These function modules are as follows:
- load_nn(version): loads the given version of the TWIG neural network.
- load_filters(dataset_name): loads all filters that should be used for a given dataset for all different phases of learning. What this really means is that the training triples are used as filters during the training phase; the training and validation triples are used as filters during the validation phase; and all triples (training, validation, and testing) are used as filters during the testing phase.
- load_negative_samplers(dataset_names): loads a negative sampler for each dataset. Note, as mentioned above, that since each negative sampler collects data on KG structure, each is created in a dataset-dependent manner.
- train_and_eval: this takes all data produced from the previous steps, as well as hyperparameter and model configuration information from the command line arguments, to call the trainer (train.py) via its run_training(.) function.

