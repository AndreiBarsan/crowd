"""Utilities for simulating the crowdsourced voting process."""

# pylint: disable=missing-docstring, superfluous-parens

from abc import ABC, abstractmethod
import random
from typing import Mapping, Sequence, Tuple

from sklearn.externals.joblib import Parallel, delayed

from crowd.data import ExpertLabel, JudgementRecord


DEFAULT_BUDGET = 250

# Use all cores for parallel stuff.
N_CORES = -1


class DocumentSampler(ABC):
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


def request_vote(topic_judgements, document_id):
    votes = topic_judgements[document_id]
    if len(votes) == 0:
        print("No votes found for document [{}].".format(document_id))
        return None

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

    for doc_id in ground_truth:
        # TODO(andrei): Don't make it this far with shitty data. Filter useless
        # GTs earlier on! Unestabilshed relevance in ground truth.
        if ground_truth[doc_id].label < 0:
            continue

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
    # Don't keep track of anything for which we don't have the ground truth,
    # since there's no way to evaluate that.
    sampled_votes = {n.document_id: [] for n in topic_graph.nodes
                     if n.document_id in ground_truth}

    budget = kw['budget'] if 'budget' in kw else DEFAULT_BUDGET
    # How often we actually want to compute the accuracy, in terms of votes
    # sampled. We likely don't need to recompute everything after every
    # single new vote.
    accuracy_every = kw['accuracy_every'] if 'accuracy_every' in kw else 1

    # TODO(andrei) Numpyfy: accuracies = np.zeros(budget // accuracy_every)
    accuracies = []
    for i in range(budget):
        # 1. Pick document according to sampling strategy
        document_id = document_sampler.sample(sampled_votes, topic_judgements)

        # 2. Request a vote for that document (sample with replacement)
        vote = request_vote(topic_judgements, document_id)
        sampled_votes[document_id].append(vote)

        if i % accuracy_every == 0:
            # 3. Perform the aggregation
            evaluated_judgements = vote_aggregation(topic_graph, sampled_votes,
                                                    **kw)

            # 4. Measure accuracy
            accuracy = measure_accuracy(evaluated_judgements, ground_truth,
                                        topic_judgements)
            accuracies.append(accuracy)

    return accuracies

# Avoids re-creating the worker pool every time we invoke 'evaluate'.
WORKER_POOL = Parallel(n_jobs=N_CORES)


def evaluate(topic_graph,
             topic_judgements: Mapping[str, Sequence[JudgementRecord]],
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

#     print("Performing evaluation of topic [{}].".format(topic_graph.topic))
#     print("Aggregation function: [{}]".format(vote_aggregation))
    import time
    start = time.time()

    # TODO(andrei): Make this cleaner, and get rid of the factory hack.
    if hasattr(document_sampler, '__call__'):
        # Treat it as a factory. Useful if the document sampler itself has its
        # own state, in which case we want each task to have its own copy.
        all_accuracies = WORKER_POOL(
            delayed(evaluate_iteration)(topic_graph, topic_judgements,
                                        ground_truth, document_sampler(),
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

