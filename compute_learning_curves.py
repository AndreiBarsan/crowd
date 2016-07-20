#!/usr/bin/env python
"""Tool for generating aggregate plots for various techniques."""

from datetime import datetime
import logging
import os
import random
from typing import Sequence, Mapping

import click
import dill
import numpy as np

from crowd.aggregation import *
from crowd.config import JUDGEMENT_FILE, FULLTEXT_FOLDER
from crowd.cross_topic import *
from crowd.data import *
from crowd.experiment_config import ExperimentConfig
from crowd.graph import *
from crowd.graph_sampling import lgss_graph_factory, sample_edges_lt
from crowd.matlab.bridge import MatlabBridgeDriver
from crowd.topic import load_topic_metadata

random.seed(0x7788)
np.random.seed(0x7788)


# Various experiment configs from the original comparison notebook.
# LV = least-votes
mv_config = ExperimentConfig(aggregate_MV, "LV-MV", {})
mv_nn_config = ExperimentConfig(aggregate_MV_NN, "LV-MVNN(0.5)", {'rho_s': 0.5})
mv_nn_075_config = ExperimentConfig(aggregate_MV_NN, "LV-MVNN(0.75)", {'rho_s': 0.75})
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
experimental_LT_config = ExperimentConfig(aggregate_mev_nx,
                                          "LT5-MEV",
                                          {},
                                          nx_graph=True,
                                          document_sampler=lgss_graph_factory(
                                              5, edge_sampler=sample_edges_lt),
                                          graph_opts={'marker': 'x',
                                                      'markevery': 15})
experimental_sgd_config = ExperimentConfig(aggregate_lm,
                                          "LM-SGD",
                                          {},
                                          nx_graph=True,
                                           # Use vanilla random sampler
                                          graph_opts={'marker': 'o',
                                                      'markevery': 15})
experimental_gpml_config = ExperimentConfig(
    aggregate_gpml,
    "LV-GP",
    # {MATLAB_DRIVER_KEY: MatlabDiskDriver()},
    {MATLAB_DRIVER_KEY: MatlabBridgeDriver()},
    nx_graph=True,
    # Use default vanilla random sampler.
    graph_opts={'marker': 'o', 'markevery': 15})

# This is the config for evaluating the core contribution of the paper.
# Can graph-based document sampling + GP aggregation beat the state of the art
# from Martin's paper using randomized least-votes sampling + GP aggregation?
graph_sampling_with_gp = ExperimentConfig(aggregate_gpml,
                                          "IC5-GP",
                                          {},
                                          nx_graph=True,
                                          document_sampler=lgss_graph_factory(5),
                                          graph_opts={'marker': 's',
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


def load_experiment_data(use_cache=True) -> ExperimentData:
    """Utility for loading experiment data which caches preprocessing.

    Pre-computed document similarity graphs are cached for fast later retrieval.
    """
    id_topic_info = load_topic_metadata()

    if use_cache:
        cache_dir = os.path.join('.', 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        cache_fname = 'exp-data-{0}-{1}.pkl'.format(
            SIM_THRESHOLD,
            'discard-empty' if DISCARD_EMPTY_NODES else 'no-discard-empty')
        cache_path = os.path.join(cache_dir, cache_fname)

        if os.path.exists(cache_path):
            logging.info("Loading cached graphs from %s.", cache_path)
            with open(cache_path, 'rb') as cache_file:
                pickled_experiment_data = dill.load(cache_file)
                assert isinstance(pickled_experiment_data, ExperimentData)
                return pickled_experiment_data
        else:
            logging.info("No cache found in directory %s. Computing graph from"
                         " scratch.", cache_dir)
    else:
        logging.info("Not using caching.")

    turk_judgements = read_useful_judgement_labels(JUDGEMENT_FILE)
    ground_truth = read_ground_truth()

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

    if use_cache:
        with open(cache_path, 'wb') as cache_file:
            logging.info("Caching experiment data...")
            dill.dump(experiment_data, cache_file)
            logging.info("Finished caching experiment data.")

    return experiment_data


@click.command()
@click.option('--label', default="", help="A short description of the"
                                          " experiment.")
@click.option('--aggregation_iterations',
              default=15,
              help="How many times to simulate the entire voting process."
                   " Necessary due to the randomness of vote sampling, among"
                   " other things.")
@click.option('--result_pickle_root',
              default='experiments/',
              help="Folder where to pickle experiment results for later"
                   " inspection.")
@click.option('--git', default="NO-GIT", help="Current git revision hash."
                                              " Useful for reproducibility.")
@click.option('--topic_limit', default=-1, help="How many topics to process"
                                                " out of the total 30. Useful"
                                                " for limiting the processing"
                                                " for quick tests. -1 means"
                                                " no limit.")
def learning_curves(label, aggregation_iterations, result_pickle_root, git,
                    topic_limit):
    # TODO(andrei): Use label and pass git revision explicitly!
    cross_topic_experiments = [
        # graph_sampling_with_gp,
        # experimental_sgd_config,
        # experimental_IC_config,
        # experimental_LT_config,
        experimental_gpml_config,
        mv_config,
        mv_nn_config,
        # mv_nn_075_config,
        mev_1_config,
        mev_2_config,
        mev_3_config
    ]

    if not os.path.exists(result_pickle_root):
        os.mkdir(result_pickle_root)

    logging.info("Will compute experiments for %d configurations.",
                 len(cross_topic_experiments))

    logging.info("Loading experiment data...")
    # TODO(andrei): Figure out why enabling caching sometimes causes strange
    # issues with the graph. Are we still trying to read some original data
    # files after loading the pickle and screwing up because of IDs and ordering
    # and such?
    experiment_data = load_experiment_data(use_cache=False)
    logging.info("Finished loading experiment data.")

    # TODO(andrei): What about the loser topics (with WAY fewer votes)?
    up_to_votes_per_doc = 1
    if topic_limit > -1:
        print("Topic limit: {0}".format(topic_limit))
    else:
        print("No topic limit")

    now = datetime.now()
    timestamp = int(now.timestamp())

    logging.info("Kicking off computation...")
    # TODO(andrei): Aggregate, plot & pickle after every new topic.
    all_frames = compute_cross_topic_learning(
        cross_topic_experiments,
        up_to_votes_per_doc,
        aggregation_iterations,
        experiment_data,
        topic_limit=topic_limit)
    logging.info("Completed computation.")

    # TODO-LOW(andrei): Add option to simple re-plot old pickle.

    experiment_folder_name = 'curves-upto-{0}-topic-limit-{1}-{2}'.format(
        up_to_votes_per_doc, topic_limit, timestamp)
    experiment_folder_path = os.path.join(result_pickle_root, experiment_folder_name)
    os.mkdir(experiment_folder_path)

    result_path = os.path.join(experiment_folder_path, 'result-data.pkl')
    with open(result_path, 'wb') as result_file:
        dill.dump(all_frames, result_file)
        # TODO(andrei): Also write metainformation, such as full details of
        # all used experiment configurations.

    plot_dir = os.path.join(experiment_folder_path, 'plots')
    os.mkdir(plot_dir)
    logging.info("Plotting...")
    plot_cross_topic_learning(
        all_frames,
        up_to_votes_per_doc,
        aggregation_iterations,
        SIM_THRESHOLD,
        plot_dir)
    logging.info("Finished plotting.")


if __name__ == '__main__':
    # TODO(andrei): Allow specifying log level from file.
    logging.basicConfig(level=logging.DEBUG)
    learning_curves()

