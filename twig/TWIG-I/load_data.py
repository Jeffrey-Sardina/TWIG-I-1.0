import pandas as pd
from utils import get_triples, calc_graph_stats, get_triples_by_idx
import torch
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from pykeen import datasets

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
====================
Constant Definitions
====================
'''
device = "cuda"

'''
=======
Classes
=======
'''
class Structure_Loader():
    def __init__(
            self,
            triples_map,
            metadata,
            ents_to_triples,
            use_2_hop_fts=True
        ):
        '''
        Note: use_2_hop_fts is very computationally expensive, and so far is
        intractible on larger datasets (larger than UMLS / Kinships)
        '''
        self.triples_map = triples_map
        self.metadata = metadata
        self.ents_to_triples = ents_to_triples
        self.use_2_hop_fts = use_2_hop_fts

        if use_2_hop_fts:
            self.neighbour_nodes, self.neighbour_preds = self.build_neighbour_cache()

    def build_neighbour_cache(self):
        neighbour_nodes = {}
        neighbour_preds = {}
        for target in self.ents_to_triples:
            neighbour_nodes[target] = {}
            neighbour_preds[target] = {}
            for t_idx in self.ents_to_triples[target]:
                t_s, t_p, t_o = self.triples_map[t_idx]
                ent = t_s if target != t_s else t_o
                if not ent in neighbour_nodes[target]:
                    neighbour_nodes[target][ent] = self.metadata['degrees'][ent]
                if not t_p in neighbour_preds[target]:
                    neighbour_preds[target][t_p] = self.metadata['pred_freqs'][t_p]

        for target in self.ents_to_triples:
            neighbour_nodes[target]['mean'] = np.mean( #this must be first; otherwise mean will incl min and max and that is not good!!
                list(neighbour_nodes[target].values())
            )
            neighbour_nodes[target]['min'] = np.min(
                list(neighbour_nodes[target].values())
            )
            neighbour_nodes[target]['max'] = np.max(
                list(neighbour_nodes[target].values())
            )
            count = len(neighbour_nodes[target]) - 3 #-3 for min max and mean
            neighbour_nodes[target]['count'] = count

        for target in self.ents_to_triples:
            neighbour_preds[target]['mean'] = np.mean( #this must be first; otherwise mean will incl min and max and that is not good!!
                list(neighbour_preds[target].values())
            )
            neighbour_preds[target]['min'] = np.min(
                list(neighbour_preds[target].values())
            )
            neighbour_preds[target]['max'] = np.max(
                list(neighbour_preds[target].values())
            )
            count = len(neighbour_preds[target]) - 3 #-3 for min max and mean
            neighbour_preds[target]['count'] = count

        return neighbour_nodes, neighbour_preds

    def __call__(self, s, p, o):
        data = []
        s_deg = self.metadata['degrees'][s] \
            if s in self.metadata['degrees'] else 0
        o_deg = self.metadata['degrees'][o] \
            if o in self.metadata['degrees'] else 0
        p_freq = self.metadata['pred_freqs'][p] \
            if p in self.metadata['pred_freqs'] else 0

        s_p_cofreq = self.metadata['subj_relationship_degrees'][(s,p)] \
            if (s,p) in self.metadata['subj_relationship_degrees'] else 0
        o_p_cofreq = self.metadata['obj_relationship_degrees'][(o,p)] \
            if (o,p) in self.metadata['obj_relationship_degrees'] else 0
        s_o_cofreq = self.metadata['subj_obj_cofreqs'][(s,o)] \
            if (s,o) in self.metadata['subj_obj_cofreqs'] else 0

        data.extend([
            s_deg,
            o_deg,
            p_freq,
            s_p_cofreq,
            o_p_cofreq,
            s_o_cofreq
        ])

        if self.use_2_hop_fts:
            target_dict = {'s': s, 'o': o}
            for target_name in target_dict:
                target = target_dict[target_name]

                min_deg_neighbour = self.neighbour_nodes[target]['min']
                max_deg_neighbour = self.neighbour_nodes[target]['max']
                mean_deg_neighbour = self.neighbour_nodes[target]['mean'] #err
                num_neighbours = self.neighbour_nodes[target]['count']

                min_freq_rel = self.neighbour_preds[target]['min']
                max_freq_rel = self.neighbour_preds[target]['max']
                mean_freq_rel = self.neighbour_preds[target]['mean'] #err
                num_rels = self.neighbour_preds[target]['count']

                data.extend([
                    min_deg_neighbour,
                    max_deg_neighbour,
                    mean_deg_neighbour,
                    num_neighbours,
                    min_freq_rel,
                    max_freq_rel,
                    mean_freq_rel,
                    num_rels
                ])

        return data

'''
=========
Functions
=========
'''
def get_adj_data(triples_map):
    '''
    get_adj_data() generates a mapping from every entity to all triples that
    contain it as a subject or object.

    The arguments it accepts are:
        - triples_map (dict int to tuple<int,int,int>): a dict mapping from a
          triple ID to the IDs of the three elements (subject, predicate, and
          object) that make up that triple. 

    The values it returns are:
        - ents_to_triples (dict int -> list of tuple <int,int,int>): a dict
          that maps an entity ID to a list of all triples (expressed as
          the IDs for the subj, pred, and obj) containing that original
          entity.
    '''
    ents_to_triples = {} # entity to all relevent data
    for t_idx in triples_map:
        s, p, o = triples_map[t_idx]
        if not s in ents_to_triples:
            ents_to_triples[s] = set()
        if not o in ents_to_triples:
            ents_to_triples[o] = set()
        ents_to_triples[s].add(t_idx)
        ents_to_triples[o].add(t_idx)
    return ents_to_triples

def get_twm_data_augment(
        triples_map,
        graph_stats,
        ents_to_triples
    ):
    '''
    get_twm_data_augment() generates all feature vectors for the given input
    data (triple IDs) and KG metadata. These will contain both local and global
    features.

    The arguments it accepts are:
        - triples_map (dict int to tuple<int,int,int>): a dict mapping from a
          triple ID to the IDs of the three elements (subject, predicate, and
          object) that make up that triple. 
        - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
        - ents_to_triples (dict int -> list of tuple <int,int,int>): a dict
          that maps an entity ID to a list of all triples (expressed as
          the IDs for the subj, pred, and obj) containing that original
          entity.

    The values it returns are:
        - X_p (Tensor): A matrix of feature vectors for all triples described
          in triples_map.
    '''
    metadata = graph_stats['train'] #always use data from training set to annotate

    struct_loader  = Structure_Loader(
        triples_map,
        metadata,
        ents_to_triples,
        use_2_hop_fts=True
    )
    
    all_data_pos = []
    for triple_idx in triples_map:
        s, p, o = triples_map[triple_idx]
        data_pos = struct_loader(s, p, o)
        all_data_pos.append([triple_idx] + data_pos)
    X_p = torch.tensor(all_data_pos, dtype=torch.float32)

    return X_p

def get_norm_func(
        X,
        norm_col_0,
        normalisation,
    ):
    assert normalisation in ('minmax', 'zscore', 'none')
    
    if normalisation == 'none':
        def norm_func(base_data):
            return base_data
        return norm_func
    
    elif normalisation == 'minmax':
        running_min = torch.min(X, dim=0).values
        running_max = torch.max(X, dim=0).values

        def norm_func(base_data, col_0_removed):
            return minmax_norm_func(
                base_data,
                running_min,
                running_max,
                norm_col_0,
                col_0_removed
            )
        
        return norm_func

    elif normalisation == 'zscore':
        # running average has been verified to be coreect
        num_samples = 0.
        num_samples += X.shape[0]
        running_avg = torch.sum(X, dim=0) / num_samples

        # running std has been verified to be coreect
        running_std = torch.sum(
            (X - running_avg) ** 2,
            dim=0
        )
        running_std = torch.sqrt(
            (1 / (num_samples - 1)) * running_std
        )

        def norm_func(base_data, col_0_removed):
            return zscore_norm_func(
                base_data,
                running_avg,
                running_std,
                norm_col_0,
                col_0_removed
            )
        
        return norm_func

def minmax_norm_func(X, train_min, train_max, norm_col_0, col_0_already_removed):
    if not norm_col_0:
        if not col_0_already_removed:
            X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the max rank, and we needs its original value!
            X_other = (X_other - train_min[1:]) / (train_max[1:] - train_min[1:])
            X_norm = torch.concat(
                [X_graph, X_other],
                dim=1
            )
        else:
            X_norm = (X - train_min[1:]) / (train_max[1:] - train_min[1:])
    else:
        assert False, 'Col 0 should not be normalised -- it contains indicies!'
        X_norm = (X - train_min) / (train_max - train_min)

    # if we had nans (i.e. min = max) set them all to 0.5
    X_norm = torch.nan_to_num(X_norm, nan=0.5, posinf=0.5, neginf=0.5) 
    return X_norm

def zscore_norm_func(X, train_mean, train_std, norm_col_0, col_0_already_removed):
    # col_0_already_removed is for the input X only
    if not norm_col_0:
        if not col_0_already_removed:
            X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the max rank, and we needs its original value!
            X_other = (X_other - train_mean[1:]) / train_std[1:]
            X_norm = torch.concat(
                [X_graph, X_other],
                dim=1
            )
        else:
            X_norm = (X - train_mean[1:]) / train_std[1:]
    else:
        assert False, 'Col 0 should not be normalised -- it contains indicies!'
        X_norm = (X - train_mean) / train_std

    # if we had nans (i.e. min = max) set them all to 0
    X_norm = torch.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0) 
    return X_norm 

def calc_split_data(
        triples_dicts,
        graph_stats,
        purpose,
        normalisation,
        norm_func=None,
        batch_size=4096
    ):
    '''
    calc_split_data() creates a full dataloader for a given data split (i.e.
    train, test, valid).
    
    The arguments it accepts are:
        - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
        - purpose (str): "train", "valid", or "test" -- the phase of
          training / evaluation for which data is being collected.
        - norm_basis (torch.Tensor): the Tensor whose values for a basis for
          normalisation. It's values are the used to compute the normalisation
          parameters (such as min and max values for minmax normalisation). As an
          example, when normalising the test set, you want to normalise it
          relative to the values **in the train set** so as to avoid data
          leakage. In this case the norm_basis would be be training tensor.
          If None, a new norm basis will be generated from the input data.
          Default None.
        - normalisation (str): A strong representing the methos that should
          be used to normalise all data. Currently, "zscore" and "minmax" are
          implemented; "none" can be given to not use normalisation. Defaults
          to "none".
        - batch_size (int): the batch size to use while training.

    The values it returns are:
        - dataloader_pos (torch.utils.data.DataLoader): A dataloader configured
          to load the input data with the requested batch size.
        - norm_basis (torch.Tensor): the Tensor whose values for a basis for
          normalisation. See above (arguments) for a full description of what
          it does.
        - X_p (torch.Tensor): the tensor containing all data in the dataloader
          as a single object. This is needed for use with the
          VectorNegativeSampler to build a triple ID -> triple feature vector
          map.
    
    '''
    triples_map = get_triples_by_idx(triples_dicts, purpose)
    ents_to_triples = get_adj_data(triples_map)

    X_p = get_twm_data_augment(
        triples_map,
        graph_stats,
        ents_to_triples,
    )
    X_p = X_p.to(device)

    print('X_p:', X_p.shape)

    if norm_func is None:
        norm_func = get_norm_func(
            X_p,
            norm_col_0=False,
            normalisation=normalisation,
        )
    X_p = norm_func(X_p, col_0_removed=False)

    torch_dataset_pos = TensorDataset(X_p)
    dataloader_pos = DataLoader(
        torch_dataset_pos,
        batch_size=batch_size
    )

    return dataloader_pos, norm_func, X_p

def do_load(
        all_datasets,
        normalisation,
        batch_size,
        batch_size_test
    ):
    '''
    do_load() is the port-of-call function for loading data. It orchestrates
    all helpers to load data from disk (or from a saved intermediary file with
    all the pre-processing already done) and returns ready-to-use Dataloaders.
    
    The arguments it accepts are:
        - all_datasets (list of str): a list of the names of the datasets that
          should be loaded.
        - normalisation (str): the normalisation method to be used when loading data
          (and when created vectorised forms of negatively sampled triples).
          "zscore", "minmax", and "none" are currently supported.
        - batch_size (int): the batch size to use while training.
        - batch_size_test (int): the batch size to use during testing and validation.

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
          leakage. In this case the norm_basis would be be training tensor.
    '''
    dataloaders = {
        'train': dict(),
        'test': dict(),
        'valid': dict()
    }
    norm_funcs = {}
    for dataset_name in all_datasets:
        print(dataset_name)
        pykeen_dataset = datasets.get_dataset(dataset=dataset_name)
        triples_dicts = get_triples(pykeen_dataset)
        graph_stats = calc_graph_stats(triples_dicts, do_print=False)

        train_X_pos, norm_func, _ = calc_split_data( #3rd was raw_X_train_pos
            triples_dicts,
            graph_stats,
            purpose='train',
            normalisation=normalisation,
            norm_func=None,
            batch_size=batch_size
        )
        test_X_pos, _, _ = calc_split_data(
            triples_dicts,
            graph_stats,
            purpose='test',
            normalisation=normalisation,
            norm_func=norm_func,
            batch_size=batch_size_test
        )
        valid_X_pos, _, _ = calc_split_data(
            triples_dicts,
            graph_stats,
            purpose='valid',
            normalisation=normalisation,
            norm_func=norm_func,
            batch_size=batch_size_test
        )

        dataloaders['train'][dataset_name] = train_X_pos
        dataloaders['test'][dataset_name] = test_X_pos
        dataloaders['valid'][dataset_name] = valid_X_pos
        norm_funcs[dataset_name] = norm_func

    return dataloaders, norm_funcs
