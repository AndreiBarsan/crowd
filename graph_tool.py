"""Utility for experimenting with graph sampling. Helps profile stuff."""

import logging
import time

from crowd.config import FULLTEXT_FOLDER
from crowd.data import read_useful_judgement_labels, read_ground_truth, \
    get_topic_judgements_by_doc_id
from crowd.graph import build_nx_document_graph
from crowd.graph_sampling import build_seed_set_lg
from crowd.topic import load_topic_metadata

import click


@click.group()
def cli():
    # Entry Point
    logging.basicConfig(level=logging.DEBUG)


# TODO(andrei): Write test: given a budget == n_nodes, ensure that full spread
# is reached at the end and never decreases.


@cli.command()
@click.option('--budget', default=15, help="Up to how many nodes to sample.")
@click.option('--iteration_count', default=10, help="How many times to sample"
                                                    " the graph when computing"
                                                    " the best expected"
                                                    " spread.")
def solo(budget, iteration_count):
    """Experiments on a single graph."""
    sim_threshold = 0.75
    discard_empty = True

    id_topic_info = load_topic_metadata()
    turk_judgements = read_useful_judgement_labels()
    ground_truth = read_ground_truth()

    # Topic: 20814 Elvish Language
    topic_id = '20814'
    # topic_id = '20704'
    graph = build_nx_document_graph(
        id_topic_info[topic_id],
        ground_truth,
        get_topic_judgements_by_doc_id(topic_id, turk_judgements),
        FULLTEXT_FOLDER,
        sim_threshold=sim_threshold,
        discard_empty=discard_empty)

    n_docs = graph.nx_graph.number_of_nodes()
    print("Total nodes: {0}".format(n_docs))

    iteration_count = 10
    start = time.time()
    result, stats, best_spread = build_seed_set_lg(graph.nx_graph,
                                                   budget=budget,
                                                   iteration_count=iteration_count)

    print("Stats: {0}".format(stats))
    delta_ms = int((time.time() - start) * 1000)
    print("budget={0}, iterations={1}, time={2}ms".format(budget,
                                                          iteration_count,
                                                          delta_ms))

    # This part just figures out what set we computed.
    # clean_res = []
    # for r in result:
    #     d_id = r.document_id
    #     d_id = d_id[d_id.find('-') + 1:]
    #     d_id = d_id[d_id.find('-') + 1:]
    #     clean_res.append(d_id)
    # print(sorted(clean_res)


if __name__ == '__main__':
    cli()
