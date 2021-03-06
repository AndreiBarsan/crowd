"""Utilities for simulating the crowdsourced voting process."""

from abc import ABCMeta, abstractmethod
import logging
import random
import time
from typing import Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from crowd.config import MATLAB_DRIVER_FACTORY_KEY
from crowd.data import *
from crowd.graph import NxDocumentGraph, DocumentGraph
from crowd.matlab.bridge import MatlabBridgeDriver
from crowd.matlab.disk import MatlabDiskDriver
from crowd.matlab.matlabdriver import MatlabDriver, MatlabDriverFactory
from crowd.topic import Topic

DEFAULT_BUDGET = 250

# TODO(andrei): De-constantify and pass as argument to program.
# Use all cores for parallel stuff.
N_CORES = -1
# N_CORES = -4


# TODO(andrei): Move sampling and aggregation to their own MODULES.
class DocumentSampler(metaclass=ABCMeta):
    """ABC for components which sample a document from the existing pool.

    These modules can either just randomly sample, or use some smart criteria
    for picking the next document for which to request a vote.
    """
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, existing_votes, available_topic_judgements):
        """Selects a document based on the existing votes."""
        pass


class LeastVotesSampler(DocumentSampler):

    def __init__(self):
        super().__init__()

    def sample(self, existing_votes, available_topic_judgements):
        """Selects a document from the set of documents with minimal votes."""
        # TODO(andrei) Use heap, or even integrate in graph.

        # Sort the map by the values and return the key with the smallest value.
        return (sorted(existing_votes.items(), key=lambda i: len(i[1])))[0][0]


def request_vote(topic_judgements, document_id) -> JudgementRecord:
    # TODO(andrei): Option to request multiple votes at once.
    votes = topic_judgements[document_id]
    if len(votes) == 0:
        raise ValueError("No votes found for document [{}].".format(document_id))

    vote = random.choice(votes)
    return vote


def measure_accuracy(evaluated_judgements, ground_truth, topic_judgements):
    """Compares a set of computed judgements with the ground truth.

    Returns:
        A floating point number in [0, 1] quantifying the accuracy of the
        computed judgements.
    """
    # TODO(andrei) Vectorize this computation as much as possible.
    match = 0
    fail = 0

    if len(evaluated_judgements) > len(ground_truth):
        logging.warning("Computed more predictions than ground truth data"
                        " points (%d vs %d). You may be doing a bit of"
                        " redundant work.",
                        len(evaluated_judgements),
                        len(ground_truth))

    if len(evaluated_judgements) != len(ground_truth):
        raise ValueError("Not enough predictions computed!")

    for doc_id in ground_truth:
        if ground_truth[doc_id].label < 0:
            raise ValueError("Unwanted ground truth label with junk relevance."
                             " (negative, meaning label is useless)")

        gt_label = (ground_truth[doc_id].label > 0)
        evaluated_label = evaluated_judgements[doc_id]
        if gt_label == evaluated_label:
            match += 1
        else:
            fail += 1

    return match / (match + fail)


def evaluate_iteration(topic_graph, topic_judgements, topic_ground_truth,
                       document_sampler, vote_aggregation, **kw):
    """ Performs a single iteration of a learning curve simulation.

    Please see the 'evaluate' function for more information.
    """

    # Ensures we start the driver for every worker separately.
    # TODO(andrei): Consider bundling factories together into larger object
    # which is more convenient to pass around.
    if MATLAB_DRIVER_FACTORY_KEY in kw:
        mdf = kw[MATLAB_DRIVER_FACTORY_KEY]
        assert isinstance(mdf, MatlabDriverFactory)
        kw['matlab_driver'] = mdf.build()
        logging.warning("Built driver for worker: %s", kw['matlab_driver'])
        kw['matlab_driver'].start()
        logging.warning("Driver started OK.")
    else:
        kw['matlab_driver'] = MatlabDiskDriver()
        logging.warning("No MATLAB factory in config. Used default driver: %s",
                        kw['matlab_driver'])

    # Ensure we don't try to sample votes where we have NO votes, and no ground
    # truth. However, if for a document we have a ground truth but no votes,
    # then it will always drag our numbers down, but we would still have to
    # keep track of it.
    sampled_votes = {n.document_id: [] for n in topic_graph.nodes
                     if n.document_id in topic_ground_truth or
                     n.document_id in topic_judgements}

    budget = kw['budget'] if 'budget' in kw else DEFAULT_BUDGET
    # How often we actually want to compute the accuracy, in terms of votes
    # sampled. We likely don't need to recompute everything after every
    # single new vote.
    # TODO(andrei): Is this still working correctly and is it still necessary?
    accuracy_every = kw['accuracy_every'] if 'accuracy_every' in kw else 1

    # This contains the IDs of the documents which actually have ground truth
    # information associated with them.
    kw['ground_truth_docs'] = topic_ground_truth.keys()

    # TODO(andrei) Numpyfy: accuracies = np.zeros(budget // accuracy_every)
    accuracies = []
    for i in range(budget):
        # 1. Pick document according to sampling strategy.
        document_id = document_sampler.sample(sampled_votes, topic_judgements)

        # 2. Request a vote for that document (sample with replacement).
        vote = request_vote(topic_judgements, document_id)
        sampled_votes[document_id].append(vote)

        if i % accuracy_every == 0:
            # 3. Perform the aggregation.
            evaluated_judgements = vote_aggregation(topic_graph,
                                                    sampled_votes,
                                                    topic_ground_truth.keys(),
                                                    **kw)

            # 4. Measure accuracy.
            accuracy = measure_accuracy(evaluated_judgements, topic_ground_truth,
                                        topic_judgements)
            accuracies.append(accuracy)

    return accuracies

# Avoids re-creating the worker pool every time we invoke 'evaluate'.
WORKER_POOL = Parallel(n_jobs=N_CORES)


def evaluate(topic_graph,
             topic_judgements: Mapping[str, Sequence[JudgementRecord]],
             # TODO(andrei): rename to e.g. ground_truth_by_doc_id
             topic_ground_truth: Mapping[str, ExpertLabel],
             document_sampler,
             vote_aggregation,
             **kw
             ) -> Tuple[Sequence[Sequence[float]], float]:
    """Evaluates a vote aggregation strategy for the specified topic.

    Args:
        topic_graph: The document graph of the topic on which we want to
            perform the evaluation.
        topic_judgements: The votes from which we sampled, as a map from
            document ID to a list of 'JudgementRecord's.
        topic_ground_truth: A map of document IDs to ground truth
            'ExpertJudgement's.
        document_sampler: Sampler or, if callable, a sampler factory,
            in case samplers have to have state (we need to perform document
            sampling in parallel, so we need multiple samplers if they are
            stateful). TODO(andrei): refactor this to avoid the duck typing.
        vote_aggregation: Function used to aggregate a document's votes and
            produce a final judgement.

    Returns:
        A list of learning curves and the total execution time, in seconds.
        The list is 'iterations' long, and each learning curve is 'budget'
        entries long.

    Notes:
        TODO(andrei) Consider using loggers to control output verbosity.
    """

    iterations = kw.get('iterations', 10)

    print("Performing evaluation of topic [{}].".format(topic_graph.topic))
    print("Aggregation function: [{}]".format(vote_aggregation))
    print("Useful ground truth labels available: [{}]".format(len(topic_ground_truth)))

    start = time.time()

    # TODO(andrei): Make this cleaner, and get rid of the factory hack.
    if hasattr(document_sampler, '__call__'):
        # Treat it as a factory. Useful if the document sampler itself has its
        # own state, in which case we want each task to have its own copy.
        all_accuracies = WORKER_POOL(
            delayed(evaluate_iteration)(topic_graph, topic_judgements,
                                        topic_ground_truth,
                                        document_sampler(topic_graph),
                                        vote_aggregation, **kw)
            for _ in range(iterations))
    else:
        # TODO(andrei): Consider using serial processing for less than k
        # iterations.
        all_accuracies = WORKER_POOL(
            delayed(evaluate_iteration)(topic_graph, topic_judgements,
                                        topic_ground_truth, document_sampler,
                                        vote_aggregation, **kw)
            for _ in range(iterations))

    end = time.time()
    duration = end - start
    return all_accuracies, duration


# TODO(andrei): Experiment or Context class to wrap around judgement and
# ground truth data.

def build_learning_curve(
        graph: Union[NxDocumentGraph, DocumentGraph],
        aggregation_function,   # TODO(andrei): Class for all aggregation & Type hint.
        judgements: Mapping[str, Sequence[JudgementRecord]],
        ground_truth: Sequence[ExpertLabel],
        document_sampler: DocumentSampler,
        **kw):
    """Returns a numpy array with growing accuracy, up to 'max_votes'."""

    print("Start: build_learning_curve")

    # TODO(andrei): Rename topic.
    topic = graph.topic
    topic_judgements = get_topic_judgements_by_doc_id(topic.topic_id,
                                                      judgements)
    # Build a map from document id to its ground truth, making sure we only
    # put in entries from the ground truth which are valid, meaning their
    # label is 0 (non-relevant) or > 0 (relevant). Negative labels mean
    # "unknown", and we don't want them.
    topic_ground_truth = {truth.document_id: truth for truth in ground_truth
                          if truth.topic_id == topic.topic_id and
                          truth.label >= 0}

    # i.e. up to target_votes votes per doc, on average.
    # TODO(andrei) Make this cleaner and more seamless.
    max_votes = kw.get('max_votes', 3)
    bud = len(topic_judgements) * max_votes
    print("Budget being used: {0}".format(bud))
    acc, eval_time_s = evaluate(
        graph,
        topic_judgements,
        topic_ground_truth,
        document_sampler,
        aggregation_function,
        budget=bud,
        **kw)

    # TODO(andrei): [Separation] This seems like a good point to dump all the
    # raw data for subsequent analysis, instead of doing the means and
    # discarding the raw 'acc'.

    # Note: this result is still indexed by vote. 'learning_curve_frame' can be
    # used to resample stuff so that the indexing happens on a mean-votes-per-
    # -document basis.
    acc = np.array(acc)
    acc_avg = np.mean(acc, axis=0)
    return acc_avg


def learning_curve_frame(graph,
                         aggregation,
                         label,
                         document_count,
                         judgements: Mapping[str, Sequence[JudgementRecord]],
                         ground_truth: Sequence[ExpertLabel],
                         document_sampler: DocumentSampler,
                         **kw):
    """Helper proxy for 'build_learning_curve'.

    Wraps its result in a pandas dataframe, and normalizes the index so that
    it represents average votes per document. This makes it much, much easier
    to perform e.g. cross-topic aggregations, when different topics have
    different document counts (which means we need different numbers of votes
    in order to reach the same mean nr. votes per document).

    Returns:
        A pandas frame of the resampled data, as well as the raw old data.
    """

    sample_count = kw.get('sample_count', 100)
    # Accuracy as expressed in votes_per_doc space.
    acc_avg = build_learning_curve(graph, aggregation, judgements, ground_truth,
                                   document_sampler, **kw)
    print("Finished building learning curve. Will resample to {0}.".format(
        sample_count))
    frame = pd.DataFrame({label: acc_avg})

    # This reindexes the learning curves onto a real mean-votes-per-doc
    # axis, which involves a little resampling.
    new_index = np.linspace(0, len(acc_avg) / document_count, sample_count)
    acc_avg_resampled = np.interp(
        new_index,
        # Sampling x coords
        np.linspace(0, len(acc_avg) / document_count, len(acc_avg)),
        # Values to sample
        acc_avg)

    frame = pd.DataFrame({label: acc_avg_resampled}, index=new_index)
    return frame, acc_avg

# This code showcases the (very small) discrepancy between the original dataset
# and the resampled one.
# f, classic = learning_curve_frame(
#     '20424',
#     mv_config.vote_aggregator,
#     "lbl",
#     len(get_topic_judgements_by_doc_id('20424', judgements)))
# plt.plot(f.index, f['lbl'])
# plt.plot(classic.index / 100.0, classic['lbl'])

