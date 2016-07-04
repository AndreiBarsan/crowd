#!/usr/bin/env python
"""Tool for generating aggregate plots for various techniques."""

import logging
import os
import pickle
from typing import Sequence, Mapping

import click

from crowd.aggregation import *
from crowd.config import JUDGEMENT_FILE, FULLTEXT_FOLDER
from crowd.cross_topic import *
from crowd.data import *
from crowd.experiment_config import ExperimentConfig
from crowd.graph import *
from crowd.graph_sampling import lgss_graph_factory
from crowd.topic import load_topic_metadata


# Various experiment configs from the original comparison notebook.
mv_config = ExperimentConfig(aggregate_MV, "LV-MV", {})
mv_nn_config = ExperimentConfig(aggregate_MV_NN, "LV-MVNN(0.5)", {'rho_s': 0.5})
mv_nn_09_config = ExperimentConfig(aggregate_MV_NN, "LV-MVNN(0.9)", {'rho_s': 0.9})

mvpp_graph_opts={'marker': 'x', 'markevery': 15}
mv_nn_plus = ExperimentConfig(aggregate_MV_NN, "LV-MVNN+(0.9)", {'rho_s': 0.9, 'seek_good_neighbor': True}, graph_opts=mvpp_graph_opts)
mv_nn_plus_hardcore = ExperimentConfig(aggregate_MV_NN, "LV-MVNN+(0.995)", {'rho_s': 0.995, 'seek_good_neighbor': True}, graph_opts=mvpp_graph_opts)

mev_graph_opts = {'marker': '^', 'markevery': 15}
mev_1_config = ExperimentConfig(aggregate_mev, "LV-MEV(1)", {'C': 1}, graph_opts=mev_graph_opts)
mev_2_config = ExperimentConfig(aggregate_mev, "LV-MEV(2)", {'C': 2}, graph_opts=mev_graph_opts)
mev_3_config = ExperimentConfig(aggregate_mev, "LV-MEV(3)", {'C': 3}, graph_opts=mev_graph_opts)
mev_4_config = ExperimentConfig(aggregate_mev, "LV-MEV(4)", {'C': 4}, graph_opts=mev_graph_opts)
mev_5_config = ExperimentConfig(aggregate_mev, "LV-MEV(5)", {'C': 5}, graph_opts=mev_graph_opts)

# This is the focus of our experiments! This (or something similar to it) must
# beat everything else!
# We must also focus on optimizing its implementation, which is currently VERY
# slow.
experimental_IC_config = ExperimentConfig(aggregate_mev_nx,
                                          "IC5-MEV",
                                          {},
                                          nx_graph=True,
                                          document_sampler=lgss_graph_factory(5),
                                          graph_opts={'marker': 'o',
                                                      'markevery': 15})

# TODO(andrei; research): 0.5 might make this shittier than conventional
# techniques, as seen in the generic experiment notebook (which had the default
# threshold of 0.5). 0.75, which was the default in the original graph sampling
# notebook, led to seemingly better results.
# This is a very important parameter. Lower values tend to add LOTS of edges,
# which can significantly slow down the graph sampling process.
SIM_THRESHOLD = 0.75
DISCARD_EMPTY_NODES = True


class ExperimentData(object):
    """Convenient holder for all available input data."""
    def __init__(self,
                 turk_judgements: Sequence[JudgementRecord],
                 ground_truth: Sequence[ExpertLabel],
                 topic_id_to_nx_graph: Mapping[str, NxDocumentGraph],
                 topic_id_to_graph: Mapping[str, DocumentGraph]):
        self.turk_judgements = turk_judgements
        self.ground_truth = ground_truth
        self.topic_id_to_nx_graph = topic_id_to_nx_graph
        self.topic_id_to_graph = topic_id_to_graph


def load_experiment_data() -> ExperimentData:
    id_topic_info = load_topic_metadata()

    cache_dir = os.path.join('.', 'cache')
    if not os.path.exists()
    cache_fname = 'exp-data-{0}-{1}.pkl'.format(
        SIM_THRESHOLD,
        'discard-empty' if DISCARD_EMPTY_NODES else 'no-discard-empty')
    cache_path = os.path.join(cache_dir, cache_fname)

    if os.path.exists(cache_path):
        logging.info("Loading cached graphs from %s.", cache_path)
        with open(cache_path, 'rb') as cache_file:
            pickled_experiment_data = pickle.load(cache_file)
            assert isinstance(pickled_experiment_data, ExperimentData)
            return pickled_experiment_data
    else:
        logging.info("No cache found in directory %s. Computing graph from"
                     " scratch.", cache_dir)

    # TODO(andrei): Better names for these functions!
    turk_judgements = read_useful_judgement_labels(JUDGEMENT_FILE)
    ground_truth = read_all_test_labels()

    # TODO-LOW(andrei): Consider parallelizing this somehow.
    id_topic_nx_graph = {topic_id: build_nx_document_graph(
        topic,
        ground_truth,
        get_topic_judgements_by_doc_id(topic_id, turk_judgements),
        FULLTEXT_FOLDER,
        sim_threshold=SIM_THRESHOLD,
        discard_empty=DISCARD_EMPTY_NODES)
                         for topic_id, topic in id_topic_info.items()}
    id_topic_graph = {topic_id: build_document_graph(
        topic, FULLTEXT_FOLDER, sim_threshold=SIM_THRESHOLD)
                      for topic_id, topic in id_topic_info.items()}

    experiment_data = ExperimentData(turk_judgements, ground_truth,
                                     id_topic_nx_graph, id_topic_graph)

    with open(cache_path, 'wb') as cache_file:
        logging.info("Caching experiment data...")
        pickle.dump(experiment_data, cache_file)
        logging.info("Finished caching experiment data.")

    return experiment_data


@click.command()
@click.option('--label', default="", help="A short description of the"
                                          " experiment.")
def learning_curves(label):
    cross_topic_experiments = [
        # experimental_IC_config,
        mv_config,
        mv_nn_config,
        mev_1_config,
        mev_2_config,
        mev_3_config]

    logging.info("Will compute experiments for %d configurations.",
                 len(cross_topic_experiments))

    logging.info("Loading experiment data...")
    experiment_data = load_experiment_data()
    logging.info("Finished loading experiment data.")

    up_to_votes_per_doc = 1
    aggregation_iterations = 10         # CHANGE THIS! AT LEAST 15!

    logging.info("Kicking off computation...")
    all_frames = compute_cross_topic_learning(
        cross_topic_experiments,
        up_to_votes_per_doc,
        aggregation_iterations,
        experiment_data,
        topic_limit=3)
    logging.info("Completed computation.")

    # TODO(andrei): Pickle intermediate results.
    # TODO(andrei): Add option to simple re-plot old pickle.

    logging.info("Plotting...")
    plot_cross_topic_learning(
        all_frames,
        up_to_votes_per_doc,
        aggregation_iterations,
        SIM_THRESHOLD)
    logging.info("Finished plotting.")


if __name__ == '__main__':
    # TODO(andrei): Allow specifying log level from file.
    logging.basicConfig(level=logging.DEBUG)
    learning_curves()

