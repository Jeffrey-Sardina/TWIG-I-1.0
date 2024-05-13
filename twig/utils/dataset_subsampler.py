from utils import get_triples
import sys
import random
from pykeen import datasets

def get_nodes(triples):
    nodes = set()
    for s, _, o in triples:
        nodes.add(s)
        nodes.add(o)
    return nodes

def get_subset(triples, nodes_to_keep):
    subset = []
    for s, p, o in triples:
        if s in nodes_to_keep and o in nodes_to_keep:
            subset.append((s, p, o))

    print(f'total triple count: {len(triples)}')
    print(f'triples kept: {len(subset)}')
    print(f'triples removed: {len(triples) - len(subset)}')
    print()

    return subset

def save_subsets(base_name, percent_nodes_to_keep, subset_dict):
    subset_prefix = f'custom_datasets/{base_name}-node_subset-{percent_nodes_to_keep}pc'
    train_out = f'{subset_prefix}.train'
    test_out = f'{subset_prefix}.test'
    valid_out = f'{subset_prefix}.valid'

    with open(train_out, 'w') as tro:
        for s, p, o in subset_dict['train']:
            print(s, p, o, sep='\t', file=tro)

    with open(test_out, 'w') as teo:
        for s, p, o in subset_dict['test']:
            print(s, p, o, sep='\t', file=teo)

    with open(valid_out, 'w') as vo:
        for s, p, o in subset_dict['valid']:
            print(s, p, o, sep='\t', file=vo)
    
def main(dataset_name, percent_nodes_to_keep):
    dataset = datasets.get_dataset(dataset=dataset_name)
    triples_dict = get_triples(dataset)
    nodes = get_nodes(triples_dict['train'])
    nodes = list(nodes)
    random.shuffle(nodes)
    n_nodes_to_keep = int(len(nodes) * percent_nodes_to_keep)
    to_keep = nodes[:n_nodes_to_keep]
    to_keep = set(to_keep)

    subset_dict = {
        'all': get_subset(triples_dict['all'], to_keep),
        'train': get_subset(triples_dict['train'], to_keep),
        'test': get_subset(triples_dict['test'], to_keep),
        'valid': get_subset(triples_dict['valid'], to_keep)
    }
    save_subsets(base_dataset, percent_nodes_to_keep, subset_dict)

    return subset_dict

if __name__ == '__main__':
    base_dataset = sys.argv[1]
    percent_nodes_to_keep = float(sys.argv[2])
    main(base_dataset, percent_nodes_to_keep)
