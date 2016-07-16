"""Utilities for simulating the crowdsourced voting process."""

# pylint: disable=missing-docstring, superfluous-parens
import logging
import random
from typing import Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from crowd.data import ExpertLabel, JudgementRecord, \
    get_topic_judgements_by_doc_id
from crowd.graph import NxDocumentGraph, DocumentGraph
from crowd.topic import Topic

DEFAULT_BUDGET = 250

# TODO(andrei): De-constantify and pass as argument to program.
# Use all cores for parallel stuff.
# N_CORES = -1
N_CORES = -4


# TODO(andrei): Python 3.3-compatible ABC! Check out estimator base classes
# in sklearn for cross-version-friendly solution!
class DocumentSampler(object):
    """ABC for components which sample a document from the existing pool.

    These modules can either just randomly sample, or use some smart criteria
    for picking the next document for which to request a vote.
    """
    def __init__(self):
        pass

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


def evaluate_iteration(topic_graph, topic_judgements, ground_truth,
                       document_sampler, vote_aggregation, **kw):
    """ Performs a single iteration of a learning curve simulation.

    Please see the 'evaluate' function for more information.
    """

    # The votes we aggregated so far.

    # Ensure we don't try to sample votes where we have NO votes, and no ground
    # truth. However, if for a document we have a ground truth but no votes,
    # then it will always drag our numbers down, but we would still have to
    # keep track of it.
    sampled_votes = {n.document_id: [] for n in topic_graph.nodes
                     if n.document_id in ground_truth or
                        n.document_id in topic_judgements}

    budget = kw['budget'] if 'budget' in kw else DEFAULT_BUDGET
    # How often we actually want to compute the accuracy, in terms of votes
    # sampled. We likely don't need to recompute everything after every
    # single new vote.
    # TODO(andrei): Is this still necessary?
    accuracy_every = kw['accuracy_every'] if 'accuracy_every' in kw else 1

    # This contains the IDs of the documents which actually have ground truth
    # information associated with them.
    kw['ground_truth_docs'] = ground_truth.keys()

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
                                                    ground_truth.keys(),
                                                    **kw)

            # 4. Measure accuracy.
            accuracy = measure_accuracy(evaluated_judgements, ground_truth,
                                        topic_judgements)
            accuracies.append(accuracy)

    return accuracies

# Avoids re-creating the worker pool every time we invoke 'evaluate'.
WORKER_POOL = Parallel(n_jobs=N_CORES)


def evaluate(topic_graph,
             topic_judgements: Mapping[str, Sequence[JudgementRecord]],
             # TODO(andrei): rename to e.g. ground_truth_by_doc_id
             ground_truth: Mapping[str, ExpertLabel],
             document_sampler,
             vote_aggregation,
             **kw
) -> Tuple[Sequence[Sequence[float]], float]:
    """ Evaluates a vote aggregation strategy for the specified topic.

    Args:
        topic_graph: The document graph of the topic on which we want to
            perform the evaluation.
        topic_judgements: The votes from which we sampled, as a map from
            document ID to a list of 'JudgementRecord's.
        ground_truth: A map of document IDs to ground truth 'ExpertJudgement's.
        document_sampler: TODO(andrei): refactor.
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
    # verbose = kw['verbose'] if 'verbose' in kw else False

    print("Performing evaluation of topic [{}].".format(topic_graph.topic))
    print("Aggregation function: [{}]".format(vote_aggregation))
    print("Useful ground truth labels available: [{}]".format(len(ground_truth)))

    import time
    start = time.time()

    # TODO(andrei): Make this cleaner, and get rid of the factory hack.
    if hasattr(document_sampler, '__call__'):
        # Treat it as a factory. Useful if the document sampler itself has its
        # own state, in which case we want each task to have its own copy.
        all_accuracies = WORKER_POOL(
            delayed(evaluate_iteration)(topic_graph, topic_judgements,
                                        ground_truth,
                                        document_sampler(topic_graph),
                                        vote_aggregation, **kw)
            for idx in range(iterations))
    else:
        # TODO(andrei) Consider using serial processing for less than k iterations.
        all_accuracies = WORKER_POOL(
            delayed(evaluate_iteration)(topic_graph, topic_judgements,
                                        ground_truth, document_sampler,
                                        vote_aggregation, **kw)
            for idx in range(iterations))

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
    topic_ground_truth = {truth.document_id: truth for truth in ground_truth
                          if truth.topic_id == topic.topic_id and
                          truth.label >= 0}

    # i.e. up to target_votes votes per doc, on average.
    # TODO(andrei) Make this cleaner and more seamless.
    max_votes = kw['max_votes'] if 'max_votes' in kw else 3
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
    #     classic = frame
    frame = pd.DataFrame({label: acc_avg_resampled}, index=new_index)

    #     return frame, classic
    return frame

# This code showcases the (very small) discrepancy between the original dataset
# and the resampled one.
# f, classic = learning_curve_frame(
#     '20424',
#     mv_config.vote_aggregator,
#     "lbl",
#     len(get_topic_judgements_by_doc_id('20424', judgements)))
# plt.plot(f.index, f['lbl'])
# plt.plot(classic.index / 100.0, classic['lbl'])

