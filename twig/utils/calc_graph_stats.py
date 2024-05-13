from utils import calc_graph_stats, get_triples, load_custom_dataset
from pykeen import datasets
import sys

def main(dataset_name):
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)

    triples_dicts = get_triples(dataset)
    calc_graph_stats(triples_dicts, do_print=True)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    main(dataset_name)
