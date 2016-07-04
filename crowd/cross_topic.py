"""Utilities for aggregating accuracy plots over multiple topics."""

from datetime import datetime
import logging
import os
from typing import Mapping, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_learning_curves import ExperimentData
from crowd.data import get_topic_judgements_by_doc_id, JudgementRecord
from crowd.experiment_config import ExperimentConfig
from crowd.graph import DocumentGraph
from crowd.graph import NxDocumentGraph
from crowd.simulation import learning_curve_frame
from crowd.util import get_git_revision_hash


# TODO(andrei): Reorder arguments in a more sensible fashion!
def compute_cross_topic_learning(experiments: Sequence[ExperimentConfig],
                                 max_votes: int,
                                 iterations: int,
                                 experiment_data: ExperimentData,
                                 **kw):
    """The main function which computes cross-topic learning curves.

    Returns:
        A list of tuples consisting of the original experiment configuration
        and the resulting dataframe.
    """
    frames = []
    for cfg in experiments:
        if cfg.nx_graph:
            logging.info("Sampling from NetworkX graph.")
            graphs_by_topic_id = experiment_data.topic_id_to_nx_graph
        else:
            logging.info("Sampling from regular graph.")
            graphs_by_topic_id = experiment_data.topic_id_to_graph

        frame = cross_topic_learning_curve_frame(
            graphs_by_topic_id,
            cfg,
            experiment_data.turk_judgements,
            experiment_data.ground_truth,
            iterations=iterations,
            max_votes=max_votes,
            **cfg.params, **kw)
        frames.append((cfg, frame))

    return frames


def cross_topic_learning_curve_frame(graphs_by_topic_id, cfg, judgements,
                                     test_data, **kw):
    curves = []
    index = None
    topic_limit = kw.get('topic_limit', -1)
    aggregation_function = cfg.vote_aggregator
    document_sampler = cfg.document_sampler
    label = cfg.name

    print("Processing aggregator: {0}.".format(label))
    topic_count = len(graphs_by_topic_id)
    for idx, (topic_id, topic) in enumerate(graphs_by_topic_id.items()):
        # This can be used to experiment with the code without committing to
        # go through all topics, which could be very, very time-consuming.
        if 0 < topic_limit <= idx:
            print("Reached limit. Stopping.")
            break

        if idx % 1 == 0:
            print("Processing topic number {0}/{1}.".format(idx, topic_count))

        topic_judgements = get_topic_judgements_by_doc_id(topic_id, judgements)
        document_count = len(topic_judgements)
        # if cfg.nx_graph:
        #     graph = id_topic_nx_graph[topic_id]
        # else:
        #     graph = id_topic_graph[topic_id]
        graph = graphs_by_topic_id[topic_id]
        # TODO(andrei): Get rid of 'document_count' argument.
        # TODO(andrei): Keep passing 'cfg' to this function instead of many params.
        curve = learning_curve_frame(
            graph,
            aggregation_function,
            label,
            document_count,
            # TODO(andrei): MORE USEFUL NAME for the ground truth. 'test_data' is confusing.
            # TODO(andrei): Properly document why we don't have to pass 'topic_judgements' here!!!
            judgements,
            test_data,
            document_sampler,
            **kw)
        curves.append(curve[label])
        index = curve.index

    curves = np.array(curves)
    means = np.mean(curves, axis=0)
    return pd.DataFrame({label: means}, index=index)


# TODO(andrei): Pass data like the similarity threshold in some wrapper, like
# with the whole graph, for instance.
def plot_cross_topic_learning(result_data_frames, max_votes, iterations,
                              similarity_threshold, plot_root):
    """Plots the results of 'compute_cross_topic_learning'.

    Saves the plot file to the disk for later inspection.
    """
    ax = None
    for cfg, frame in result_data_frames:
        ax = frame.plot(fontsize=12, ax=ax, **cfg.graph_opts)

    ax.set_xlabel("Mean votes per document", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    now = datetime.now()
    title = "Aggregate cross-topic plot; Date: {} Git: {}\n" \
            "Graph similarity threshold: {}; Iterations: {}" \
            .format(now.strftime("%Y-%m-%d %H:%M"),
                    get_git_revision_hash(),
                    similarity_threshold,
                    iterations)
    ax.set_title(title, fontsize=14)
    ax.grid()
    ax.legend(loc='lower right', fontsize=14)

    plot_name = "aggregate-{}-{}-{}-{}".format(
        # TODO(andrei): Extract common utility with nicer timestamps.
        now.strftime("%Y%m%dT%H%M"),
        # TODO(andrei): Pass this from higher level, since running on e.g.
        # Euler will NOT have this readily available!
        get_git_revision_hash(),
        similarity_threshold,
        iterations)
    # Save both an easy-to-load PNG, and a lossless EPS.
    plot_fname = '{}.eps'.format(plot_name)
    png_plot_fname = '{}.png'.format(plot_name)
    plot_path = os.path.join(plot_root, plot_fname)
    png_plot_path = os.path.join(plot_root, png_plot_fname)
    plt.savefig(plot_path)
    plt.savefig(png_plot_path)
    logging.info("Saved plot to file %s.", plot_path)
    logging.info("Saved PNG plot to file %s.", png_plot_path)
    return ax


