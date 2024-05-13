from load_data import Structure_Loader
import random
import torch
import datetime
import pickle

'''
====================
Constant Definitions
====================
'''
device = "cuda"

class Negative_Sampler:   
    def get_negatives(self, purpose, triple_index, npp):
        raise NotImplementedError
    
    def get_batch_negatives(self, purpose, triple_idxs, npp):
        '''
        get_batch_negatives() generates a batch of negatives for a given set of
        triples.
        
        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of
              training / evaluation for which these negatives are being
              generated. This is used to determine what filters to use.
            - triple_idxs (Tensor of int): a tensor containing the triple
              indicies of all triples for which negatives are wanted.
            - npp (int): the number of negative samples to generate per
              positive triple during training. If the current purpose if not
              training, it MUST be -1 to generate all triples and avoid bias.

        The values it returns are:
            - all_negs (Tensor): a tensor containing all negatives that are
              generated, in blocks in the same order as the order of the triple
              indicies that are given. 
            - npps (Tensor): a tensor containing the number of negatives per
              per positive that were used in the negative generation. All values
              in npps will be equal to the input npp unless upsampling is disabled
              in the negative sampler. 
        '''
        all_negs = None
        npps = []
        for idx in triple_idxs:
            negs, npp_returned = self.get_negatives(purpose, int(idx), npp)
            npps.append(npp_returned)
            if all_negs is None:
                all_negs = negs
            else:
                all_negs = torch.concat(
                    [all_negs, negs],
                    dim=0
                )
        npps = torch.tensor(npps, device=device)
        return all_negs, npps


class Vector_Negative_Sampler(Negative_Sampler):
    def __init__(
            self,
            X_pos,
            n_bins,
            fts_to_sample,
            invert,
            dataset_name,
            simple_negative_sampler
        ):
        self.fts_to_sample = fts_to_sample
        self.invert = invert
        self.dataset_name = dataset_name
        self.simple_negative_sampler = simple_negative_sampler #for use in testing, as this cannot be used there

        # generate map of triple index to fts
        triple_idx_to_ft_vec = {}
        for row in X_pos:
            triple_idx = int(row[0])
            fts = row[1:]
            triple_idx_to_ft_vec[triple_idx] = fts
        all_fts = X_pos[:, 1:]

        # generate map of features of their distributions
        num_cols = all_fts.shape[1]
        ft_idx_to_hist = [0 for _ in range(num_cols)]
        for col in range(num_cols):
            col_vals = all_fts[:, col]
            min_val = torch.min(col_vals)
            max_val = torch.max(col_vals)
            ft_hist = torch.histc(
                col_vals,
                bins=n_bins,
                min=min_val,
                max=max_val
            )
            # the below is verified
            bin_edges = torch.linspace(min_val, max_val, steps=n_bins+1)
            ft_idx_to_hist[col] = ft_hist, bin_edges

        self.ft_idx_to_hist = ft_idx_to_hist
        self.triple_idx_to_ft_vec = triple_idx_to_ft_vec
        if len(self.fts_to_sample) == 0:
            self.fts_to_sample = [x for x in range(len(self.ft_idx_to_hist))]

    def randomise_fts_to_sample(self, num_total=-1):
        '''
        For use with a scheuler, if one is wanted
        '''
        ft_idx_list = [x for x in range(self.ft_idx_to_hist)]
        if num_total == -1:
            self.fts_to_sample = ft_idx_list # if empty, tells it to sample all
        else:
            random.shuffle(ft_idx_list)
            ft_idx_list = ft_idx_list[:num_total]
            self.fts_to_sample = ft_idx_list

    def set_invert_dists(self, invert):
        self.invert = invert

    def sample_ft_vec(self, pos_vector):
        sampled_vec = pos_vector
        for ft_idx in self.fts_to_sample:
            sampled_ft_value = self.sample_ft(ft_idx)
            sampled_vec[ft_idx] = sampled_ft_value
            
        return sampled_vec

    def sample_ft(self, ft_idx):
        histogram, bin_edges = self.ft_idx_to_hist[ft_idx]
        if self.invert:
            # weights need not sum to 1, but must be in the right proportions
            weights = torch.sum(histogram) - histogram
        else:
            weights = histogram

        sampled_bin = random.choices(range(histogram.shape[0]), weights=weights)[0]
        bin_width = (bin_edges[sampled_bin+1] - bin_edges[sampled_bin])
        bin_min = bin_edges[sampled_bin]
        sample = random.random() * bin_width + bin_min
        return sample

    def get_negatives(self, purpose, triple_index, npp):
        if purpose != 'train':
            return self.simple_negative_sampler.get_negatives(purpose, triple_index, npp)

        print('triple index', triple_index)
        pos_vector = self.triple_idx_to_ft_vec[triple_index]

        # generate a structure-corrupted vector
        negs = []
        for _ in range(npp):
            neg_vec = self.sample_ft_vec(pos_vector)
            negs.append(neg_vec)
        negs = torch.stack(negs)
        return negs, npp
    

class Optimised_Negative_Sampler(Negative_Sampler):
    def __init__(
            self,
            filters,
            graph_stats,
            triples_map,
            ents_to_triples,
            normalisation,
            norm_func,
            dataset_name,
            prefilter=False,
        ):
        '''
        init() initialises the negative sampler with all data it will need to
        generate negatives -- including pre-calculation of all negative triple
        feature vectors so that they can be accessed rapidly during training.

        The arguments it accepts are:
            - filters (dict str -> str -> dict): a dict that maps a dataset name
              (str) and a training split name (i.e. "train", "test", or "valid") to 
              a dictionary describing the triples to use i filtering. To be exact,
              this second triples_dict has the structure
              (dict str -> int -> tuple<int,int,int>). It maps first the training
              split name to the triple index, and that trtrriple index maps to a
              single triple expressed as (s, p, o) with integral representations
              of each triple element.
            - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
            - triples_map (dict int -> tuple (int, int int)): a dict that maps
              a triple index to (s,p,o) integer IDs for nodes and edges. 
            - ents_to_triples (dict int -> int): a dict that maps an entity ID
              to a list of all triples IDs (expressed as containing that
              original entity.
            - normalisation (str): A strong representing the method that should
              be used to normalise all data; must be the same at that used in loading data.
            - norm_basis (torch.Tensor): the Tensor whose values for a basis for
              normalisation. It's values are the used to compute the normalisation
              parameters (such as min and max values for minmax normalisation). As an
              example, when normalising the test set, you want to normalise it
              relative to the values **in the train set** so as to avoid data
              leakage. In this case the norm_basis would be be training tensor. It
              is returned so that the negative sampler can use it to normalise the
              values of generated negative triples the same way that all other data
              was normalised.
            - dataset_name (str): the name of the dataset that should be used to
              save a cache of all precalculated features of avoid the need for
              redundant compuation each time TWIG is run.

        The values it returns are:
            - None (init function to create an object)

        Optimisations
            stoge a local ggrpah as an adj list, 
            and a node / rel dict mapping them to their local neighbourhood struct stats
        '''
        self.filters = filters
        self.metadata = graph_stats['train']
        self.triples_map = triples_map
        self.normalisation = normalisation
        self.norm_func = norm_func
        self.struct_loader = Structure_Loader(
            self.triples_map,
            self.metadata,
            ents_to_triples,
            use_2_hop_fts=True
        )
        self.dataset_name = dataset_name
        self.prefilter = prefilter
        self.all_ents = list(ents_to_triples.keys())

        # try to read precalc'd ft vec data from cache; if there is no cache, create it
        try:
            with open(f'twig_cache/{self.dataset_name}.opt-neg-samp.cache.pkl', 'rb') as cache:
                print('loading triple features from cache')
                self.ent_env_stats, self.rel_env_stats = pickle.load(cache)
        except:
            print('precalculating all 2-hop env stats')
            print(f'time: {datetime.datetime.now()}')
            self.ent_env_stats, self.rel_env_stats = self.precalc_env_stats()
            print('done with 2-hop env stats precalculation')
            print(f'time: {datetime.datetime.now()}')

            print('saving precalculated features to cache')
            with open(f'twig_cache/{self.dataset_name}.opt-neg-samp.cache.pkl', 'wb') as cache:
                stats = (self.ent_env_stats, self.rel_env_stats)
                pickle.dump(stats, cache)

        if self.prefilter:
            # try to read precalc'd ft data from cache; if there is no cache, create it
            try:
                with open(f'twig_cache/{self.dataset_name}.opt-prefilter.cache.pkl', 'rb') as cache:
                    print('loading precalculated corruptions from cache')
                    self.possible_corrupt_ents_for = pickle.load(cache)
            except:
                print('pre-filtering all_ents')
                print(f'time: {datetime.datetime.now()}')
                self.possible_corrupt_s, self.universally_permissible_s, self.possible_corrupt_o, self.universally_permissible_o = self.prefilter_corruptions()
                print('done pre-filtering all_ents')
                print(f'time: {datetime.datetime.now()}')

                with open(f'twig_cache/{self.dataset_name}.opt-prefilter.cache.pkl', 'wb') as cache:
                    # pass # don't save while testing
                    to_save = (self.possible_corrupt_s, self.universally_permissible_s, self.possible_corrupt_o, self.universally_permissible_o)
                    pickle.dump(to_save, cache)
        else:
            if len(self.filters['train']) > 0:
                assert False, "Filterng in training not currently supported"

    def prefilter_corruptions(self):
        print('prefiltering s corruptions:')
        possible_corrupt_s, universally_permissible_s = self.prefilter_s_corruptions()

        print('prefiltering o corruptions:')
        possible_corrupt_o, universally_permissible_o = self.prefilter_o_corruptions()

        return possible_corrupt_s, universally_permissible_s, possible_corrupt_o, universally_permissible_o

    def prefilter_s_corruptions(self):
        possible_corrupt_s = {
            'train': {},
            'valid': {},
            'test': {}
        }

        universally_permissible_s = {
            'train': {},
            'valid': {},
            'test': {}
        }

        # po extraction optimisation
        observed_po = set()
        for triple_idx in self.triples_map:
            _, p, o = self.triples_map[triple_idx]
            observed_po.add((p,o))

        i = 0

        # pre-fill possible_corrupt_ents_for
        for p, o in observed_po:
            for purpose in ('train', 'valid', 'test'):
                if not (p,o) in possible_corrupt_s[purpose]:
                    possible_corrupt_s[purpose][(p,o)] = []

        # do the prefiltering step
        for purpose in ('train', 'valid', 'test'):
            potentially_impermissible_s = set(s for s, _, _ in self.filters[purpose])
            universally_permissible_s[purpose] = list(
                set(ent for ent in self.all_ents) - potentially_impermissible_s
            )
            for p, o in observed_po:
                i += 1
                if i % 1000 == 0:
                    print(f'prefilter ({purpose}): i={i} / {3*len(observed_po)}: {datetime.datetime.now()}')
                # get what ent is a valid corruption for
                for s in potentially_impermissible_s:
                    if not (s, p, o) in self.filters[purpose]:
                        possible_corrupt_s[purpose][(p,o)].append(s)

        return possible_corrupt_s, universally_permissible_s

    def prefilter_o_corruptions(self):
        possible_corrupt_o = {
            'train': {},
            'valid': {},
            'test': {}
        }

        universally_permissible_o = {
            'train': {},
            'valid': {},
            'test': {}
        }

        # po extraction optimisation
        observed_sp = set()
        for triple_idx in self.triples_map:
            s, p, _ = self.triples_map[triple_idx]
            observed_sp.add((s,p))

        i = 0

        # pre-fill possible_corrupt_ents_for
        for s, p in observed_sp:
            for purpose in ('train', 'valid', 'test'):
                if not (s,p) in possible_corrupt_o[purpose]:
                    possible_corrupt_o[purpose][(s,p)] = []

        # do the prefiltering step
        for purpose in ('train', 'valid', 'test'):
            if len(self.filters[purpose]) == 0:
                universally_permissible_o = self.all_ents
            potentially_impermissible_o = set(o for _, _, o in self.filters[purpose])
            universally_permissible_o[purpose] = list(
                set(ent for ent in self.all_ents) - potentially_impermissible_o
            )
            for s, p in observed_sp:
                i += 1
                if i % 1000 == 0:
                    print(f'prefilter ({purpose}): i={i} / {3*len(observed_sp)}: {datetime.datetime.now()}')
                # get what ent is a valid corruption for
                for o in potentially_impermissible_o:
                    if not (s, p, o) in self.filters[purpose]:
                        possible_corrupt_o[purpose][(s,p)].append(o)
        
        return possible_corrupt_o, universally_permissible_o

    def get_negatives(self, purpose, triple_index, npp):
        if self.prefilter:
            return self.get_negatives_prefiltered(purpose, triple_index, npp)
        else:
            return self.get_negatives_running_filter(purpose, triple_index, npp)

    def get_negatives_running_filter(self, purpose, triple_index, npp):
        '''
        get_negatives() generates negates for a given triple. All sampling is
        done with replacement.

        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of
              training / evaluation for which these negatives are being
              generated. This is used to determine what filters to use.
            - triple_index (int): the triple index of the triple for which
              negatives are wanted.
            - npp (int): the number of negative samples to generate per
              positive triple during training. If the current purpose if not
              training, it MUST be -1 to generate all triples and avoid bias.

        The values it returns are:
            - negs (Tensor): a tensor containing all negatives that are
              generated for the given triple.
            - npp_returned (int): the number of negatives actually returned.
              This differs from npp only in the case that there are most
              negatives requested than can be generated (such as due to
              filters) and when upsampling is turned off. 
        '''
        if purpose == 'test' or purpose == 'valid':
            assert npp == -1, "npp = -1 should nbe used always in testing and validation"
            gen_all_negs = True
        else:
            gen_all_negs = False

        s, p, o = self.triples_map[triple_index]

        s_corrs = self.all_ents
        o_corrs = self.all_ents
 
        # trim so we only get npp negs (and have random ones)
        if not gen_all_negs:
            # generate corruptions
            npp_s = npp // 2
            npp_o = npp // 2
            if npp % 2 != 0:
                add_extra_to_s = random.random() > 0.5
                if add_extra_to_s:
                    npp_s += 1
                else:
                    npp_o += 1
                    
            if len(s_corrs) == 0 and len(o_corrs) == 0:
                # if there are not non-filtered corruptions,
                # just randomly sample from all possible ones
                s_corrs = self.all_ents
                o_corrs = self.all_ents
                assert False, 'This REALLY should not happen, pleases check on your prefilter calculations'
            elif len(s_corrs) == 0:
                # if we can't corrupt s, corrupt o more
                npp_o = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'
            elif len(o_corrs) == 0:
                # if we can't corrupt o, corrupt s more
                npp_s = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'

            if len(s_corrs) > 0:
                s_corrs = random.choices(
                    s_corrs,
                    k=npp_s
                )
            if len(o_corrs) > 0:
                o_corrs = random.choices(
                    o_corrs,
                    k=npp_o
                )

        # construct negative triples
        negs = []
        for s_corr in s_corrs:
            if not gen_all_negs:
                negs.append(
                    self.spo_to_vec(s_corr, p, o)
                )
            elif not (s_corr, p, o) in self.filters[purpose]:
                negs.append(
                    self.spo_to_vec(s_corr, p, o)
                )
        for o_corr in o_corrs:
            if not gen_all_negs:
                negs.append(
                    self.spo_to_vec(s, p, o_corr)
                )
            elif not (s, p, o_corr) in self.filters[purpose]:
                negs.append(
                    self.spo_to_vec(s, p, o_corr)
                )
        npp_returned = len(negs) #len(s_corrs) + len(o_corrs)
        negs = torch.tensor(negs, dtype=torch.float32, device=device)

        # normalise the generated negatives
        negs = self.norm_func(negs, col_0_removed=True)

        # randomise row order
        negs = negs[torch.randperm(negs.size()[0])]

        # validation
        assert negs.shape[0] == npp_returned, f'{negs.shape}[0] =/= {npp_returned}'
        if npp != -1:
            assert npp == npp_returned, f'{npp} =/= {npp_returned}'

        return negs, npp_returned
    
    def get_negatives_prefiltered(self, purpose, triple_index, npp):
        '''
        get_negatives() generates negates for a given triple. All sampling is
        done with replacement.

        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of
              training / evaluation for which these negatives are being
              generated. This is used to determine what filters to use.
            - triple_index (int): the triple index of the triple for which
              negatives are wanted.
            - npp (int): the number of negative samples to generate per
              positive triple during training. If the current purpose if not
              training, it MUST be -1 to generate all triples and avoid bias.

        The values it returns are:
            - negs (Tensor): a tensor containing all negatives that are
              generated for the given triple.
            - npp_returned (int): the number of negatives actually returned.
              This differs from npp only in the case that there are most
              negatives requested than can be generated (such as due to
              filters) and when upsampling is turned off. 
        '''
        print('WARNING -- This code is not fully tested and may have errors. Citing it is not recommended.')
        if purpose == 'test' or purpose == 'valid':
            assert npp == -1, "npp = -1 should nbe used always in testing and validation"
            gen_all_negs = True
        else:
            gen_all_negs = False

        s, p, o = self.triples_map[triple_index]

        s_corrs = self.possible_corrupt_s[purpose][(p,o)] + self.universally_permissible_s[purpose]
        o_corrs = self.possible_corrupt_o[purpose][(s,p)] + self.universally_permissible_o[purpose]
 
        # trim so we only get npp negs (and have random ones)
        if not gen_all_negs:
            # generate corruptions
            npp_s = npp // 2
            npp_o = npp // 2
            if npp % 2 != 0:
                add_extra_to_s = random.random() > 0.5
                if add_extra_to_s:
                    npp_s += 1
                else:
                    npp_o += 1
                    
            if len(s_corrs) == 0 and len(o_corrs) == 0:
                # if there are not non-filtered corruptions,
                # just randomly sample from all possible ones
                s_corrs = self.all_ents
                o_corrs = self.all_ents
                assert False, 'This REALLY should not happen, pleases check on your prefilter calculations'
            elif len(s_corrs) == 0:
                # if we can't corrupt s, corrupt o more
                npp_o = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'
            elif len(o_corrs) == 0:
                # if we can't corrupt o, corrupt s more
                npp_s = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'

            if len(s_corrs) > 0:
                s_corrs = random.choices(
                    s_corrs,
                    k=npp_s
                )
            if len(o_corrs) > 0:
                o_corrs = random.choices(
                    o_corrs,
                    k=npp_o
                )

        # construct negative triples
        negs = []
        for s_corr in s_corrs:
            negs.append(
                self.spo_to_vec(s_corr, p, o)
            )
        for o_corr in o_corrs:
            negs.append(
                self.spo_to_vec(s, p, o_corr)
            )
        npp_returned = len(s_corrs) + len(o_corrs)
        negs = torch.tensor(negs, dtype=torch.float32, device=device)

        # normalise the generated negatives
        negs = self.norm_func(negs)

        # randomise row order
        negs = negs[torch.randperm(negs.size()[0])]

        # validation
        assert negs.shape[0] == npp_returned, f'{negs.shape}[0] =/= {npp_returned}'
        if npp != -1:
            assert npp == npp_returned, f'{npp} =/= {npp_returned}'

        return negs, npp_returned
   
    def spo_to_vec(self, s, p, o):
        # 1-hop stats
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
        
        # 2-hop stats
        s_min_deg_neighbour, s_max_deg_neighbour, s_mean_deg_neighbour, s_num_neighbours = self.ent_env_stats[s]
        s_min_freq_rel, s_max_freq_rel, s_mean_freq_rel, s_num_rels = self.rel_env_stats[s]

        o_min_deg_neighbour, o_max_deg_neighbour, o_mean_deg_neighbour, o_num_neighbours = self.ent_env_stats[o]
        o_min_freq_rel, o_max_freq_rel, o_mean_freq_rel, o_num_rels = self.rel_env_stats[s]

        ft_vec = [
            s_deg,
            o_deg,
            p_freq,
            s_p_cofreq,
            o_p_cofreq,
            s_o_cofreq,
            s_min_deg_neighbour,
            s_max_deg_neighbour,
            s_mean_deg_neighbour,
            s_num_neighbours,
            s_min_freq_rel,
            s_max_freq_rel,
            s_mean_freq_rel,
            s_num_rels,
            o_min_deg_neighbour,
            o_max_deg_neighbour,
            o_mean_deg_neighbour,
            o_num_neighbours,
            o_min_freq_rel,
            o_max_freq_rel,
            o_mean_freq_rel,
            o_num_rels,
        ]

        return ft_vec

    def precalc_env_stats(self):
        ent_env_stats = {}
        rel_env_stats = {}

        for i, ent in enumerate(self.all_ents):
            if i % 100 == 0:
                print(f'precalc_env_stats: i={i}: {datetime.datetime.now()}')

            neighbour_ent_stats, neighbour_rel_stats = self.precalc_node_env_stats(ent)
            ent_env_stats[ent] = neighbour_ent_stats
            rel_env_stats[ent] = neighbour_rel_stats

        return ent_env_stats, rel_env_stats

    def precalc_node_env_stats(self, ent):  
        min_deg_neighbour = self.struct_loader.neighbour_nodes[ent]['min']
        max_deg_neighbour = self.struct_loader.neighbour_nodes[ent]['max']
        mean_deg_neighbour = self.struct_loader.neighbour_nodes[ent]['mean']
        num_neighbours = self.struct_loader.neighbour_nodes[ent]['count']

        min_freq_rel = self.struct_loader.neighbour_preds[ent]['min']
        max_freq_rel = self.struct_loader.neighbour_preds[ent]['max']
        mean_freq_rel = self.struct_loader.neighbour_preds[ent]['mean']
        num_rels = self.struct_loader.neighbour_preds[ent]['count']

        neighbour_ent_stats =  [
            min_deg_neighbour,
            max_deg_neighbour,
            mean_deg_neighbour,
            num_neighbours
        ]

        neighbour_rel_stats = [
            min_freq_rel, 
            max_freq_rel,
            mean_freq_rel,
            num_rels
        ]

        return neighbour_ent_stats, neighbour_rel_stats
