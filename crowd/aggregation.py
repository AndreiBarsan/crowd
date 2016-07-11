"""Vote aggregation algorithms."""
import datetime
import random
import shutil
import subprocess
import sys
from typing import Mapping, Sequence, Tuple

import numpy as np
import scipy as sp
from scipy import sparse, io
from sklearn.linear_model import SGDClassifier


from crowd.data import JudgementRecord
from crowd.graph import NxDocumentGraph

COIN_FLIP = "COIN_FLIP"


def full_mv_aggregation(document_id, topic_judgements):
    """Computes the voter consensus for the specified document.

    Calculation perform directly on all votes, without relying on any
    sampling.
    """

    if document_id not in topic_judgements:
        raise ValueError("Document ID#{0} doesn't have any votes to "
                         "aggregate.".format(document_id))

    votes = topic_judgements[document_id]
    rel_votes, non_rel_votes = count_votes(votes)

    if rel_votes == 0 and non_rel_votes == 0:
        raise ValueError("No votes for ground truth document ID#{0}. That's a "
                         "shame.".format(document_id))

    return rel_votes >= non_rel_votes


def count_votes(votes):
    relevant_votes = 0
    non_relevant_votes = 0
    for vote in votes:
        if vote.is_relevant > 0:
            relevant_votes += 1
        elif vote.is_relevant == 0:
            non_relevant_votes += 1
        else:
            raise ValueError("Non 0/1 vote.")

    return relevant_votes, non_relevant_votes


def majority(votes, tie_handling=COIN_FLIP) -> bool:
    """Computes the majority of a list of 'JudgementRecord's.

    Args:
        votes: A list of 'JudgementRecord' objects.
        tie_handling: An enum specifying how ties are to be treated.

    Returns:
        A boolean indicating whether the consensus is "relevant".
        Ties are broken as specified by the 'tie_handling' parameter.
    """
    # Note: mind the Nones!
    relevant = len([vote for vote in votes if vote.is_relevant is True])
    non_relevant = len([vote for vote in votes if vote.is_relevant is False])

    if relevant > non_relevant:
        return True
    elif relevant < non_relevant:
        return False
    elif tie_handling == COIN_FLIP:
        return random.choice([True, False])
    else:
        raise ValueError("Unknown tie handling technique: [{}]."
                         .format(tie_handling))


def aggregate_MV_NN(topic_graph, all_sampled_votes, **kw) -> Mapping[str, bool]:
    """Majority voting which tries to steal votes from the closest neighbor.

    This method is similar to 'aggregate_MV', but also tries to take some
    document similarity information into account.

    Args:
        topic_graph: The current topic's document graph. Not used.
        all_sampled_votes: A map from document ID to a list of sampled
            'JudgementRecord's.
        rho_s: The similarity threshold below which we ignore the nearest
            neighbor.
        seek_good_neighbor: Enables a more thorough nearest neighbor search,
            not stopping at the very first neighbor, but at the nearest
            neighbor which also has at least 'min_votes' votes.

    Returns:
        A map which contains a boolean relevance for every document.
    """

    rho_s = kw.get('rho_s', 0.9)
    del kw['rho_s']  # TODO(andrei) Do this in a nicer way.
    seek_good_neighbor = kw.get('seek_good_neighbor', False)

    if seek_good_neighbor:
        aggregator = majority_with_nn_seek
    else:
        aggregator = majority_with_nn

    return {document_id: aggregator(topic_graph, document_id,
                                    document_votes, all_sampled_votes,
                                    rho_s, **kw)
            for (document_id, document_votes) in all_sampled_votes.items()}


def majority_with_nn(topic_graph, doc_id, votes, all_sampled_votes, rho_s,
                     **kw) -> bool:
    node = topic_graph.get_node(doc_id)

    if len(node.neighbors) == 0:
        # No neighbors in graph.
        return majority(votes)

    nn = node.sim_sorted_neighbors[0]
    if nn.similarity < rho_s:
        # Nearest neighbor is not near enough.
        return majority(votes)

    if nn.to_document_id not in all_sampled_votes:
        # Neighbor has no votes.
        # Note: The original MVNN code simply looks at the most similar
        # neighbor, and adds its votes. If it has no votes, then tough luck.
        return majority(votes)

    neighbor_votes = all_sampled_votes[nn.to_document_id]
    return majority(votes + neighbor_votes)


def majority_with_nn_seek(topic_graph, doc_id, votes, all_sampled_votes, rho_s,
                          **kw) -> bool:
    """Seeks the nearest neighbor which also has some votes."""
    node = topic_graph.get_node(doc_id)
    min_votes = kw.get('min_votes', 1)

    if len(node.neighbors) == 0:
        # No neighbors in graph.
        return majority(votes)

    index = 0
    while index < len(node.neighbors):
        nn = node.sim_sorted_neighbors[index]
        index += 1

        if nn.similarity < rho_s:
            # Nearest neighbor is not near enough.
            continue

        if nn.to_document_id not in all_sampled_votes:
            # Neighbor has no vote data.
            continue

        neighbor_votes = all_sampled_votes[nn.to_document_id]
        if len(neighbor_votes) < min_votes:
            # Neighbor doesn't have enough votes for us to care.
            continue

        # Found a good neighbor. Stop the search.
        return majority(votes + neighbor_votes)

    # No good neighbor found.
    return majority(votes)


def aggregate_MV(topic_graph, all_sampled_votes, **kw) -> Mapping[str, bool]:
    """The default way of aggregating crowdsourcing votes.

    Args:
        topic_graph: The current topic's document graph. Not used.
        all_sampled_votes: A map from document ID to a list of
            'JudgementRecord's which have been sampled so far in our
            simulation.

    Returns:
        A map which contains a boolean relevance for every document.
    """

    return {document_id: majority(document_votes)
            for (document_id, document_votes) in all_sampled_votes.items()}


def mev_aggregator(topic_graph, doc_id, votes, all_sampled_votes, **kw) -> bool:
    """Performs vote aggregation using the 'MergeEnoughVotes' technique.

    Kwargs:
        C: The desired vote count.
    """
    C = kw.get('C', 1)
    node = topic_graph.get_node(doc_id)

    if len(node.neighbors) == 0 or len(votes) >= C:
        return majority(votes)

    # $U_i$ = augmented_votes
    # Make sure we don't modify the original vote list.
    augmented_votes = list(votes)

    # Merge the most similar neighbor's votes first.
    for nn in node.sim_sorted_neighbors:
        votes_left = C - len(augmented_votes)
        if votes_left <= 0:
            break

        if nn.to_document_id not in all_sampled_votes:
            # Neighbor has no vote data.
            continue

        # TODO(andrei): Experiment with similarity thresholding here.

        nn_votes = all_sampled_votes[nn.to_document_id]
        # We ensure we don't merge too many votes.
        augmented_votes += nn_votes[:votes_left]

    return majority(augmented_votes)


def mev_aggregator_nx(
        topic_graph: NxDocumentGraph,
        doc_id: str,
        votes: Mapping[str, Sequence[JudgementRecord]],
        all_sampled_votes: Mapping[str, Sequence[JudgementRecord]],
        **kw
) -> bool:
    """'MergeEnoughVotes' aggregation for NetworkX graphs.

    Somewhat slower than the regular 'mev_aggregator_nx', since a node's edges
    aren't pre-sorted by similarity.
    """
    C = kw.get('C', 1)
    # node = topic_graph.get_node(doc_id)
    nxg = topic_graph.nx_graph

    neighbors = nxg[doc_id].items()

    if len(neighbors) == 0 or len(votes) >= C:
        return majority(votes)

    # $U_i$ = augmented_votes
    # Make sure we don't modify the original vote list.
    augmented_votes = list(votes)

    # Merge the most similar neighbor's votes first.
    # TODO(andrei): Pre-compute this.
    sim_sorted_neighbors = sorted(neighbors, key=lambda nb: -nb[1]['similarity'])
    for neighbor_doc_id, edge in sim_sorted_neighbors:
        votes_left = C - len(augmented_votes)
        if votes_left <= 0:
            break

        if neighbor_doc_id not in all_sampled_votes:
            # Neighbor has no vote data.
            continue

        nn_votes = all_sampled_votes[neighbor_doc_id]
        # We ensure we don't merge too many votes.
        augmented_votes += nn_votes[:votes_left]

    return majority(augmented_votes)


def aggregate_mev(topic_graph, all_sampled_votes, **kw) -> Mapping[str, bool]:
    """Performs 'MergeEnoughVotes'-style aggregation.

    Returns:
        A map which contains a boolean relevance for every document.
    """
    return {document_id: mev_aggregator(topic_graph, document_id,
                                        document_votes, all_sampled_votes,
                                        **kw)
            for (document_id, document_votes) in all_sampled_votes.items()}


def aggregate_mev_nx(topic_graph, all_sampled_votes, **kw) -> Mapping[str, bool]:
    """See: 'aggregate_mev'"""
    return {document_id: mev_aggregator_nx(
        topic_graph,
        document_id,
        document_votes,
        all_sampled_votes,
        **kw) for (document_id, document_votes) in all_sampled_votes.items()}


def classifier_aggregation_preprocess(
        topic_graph: NxDocumentGraph,
        all_sampled_votes: Mapping[str, Sequence[JudgementRecord]],
        **kw
) -> Tuple[sparse.spmatrix, np.ndarray, list, np.ndarray]:
    """Prepares the data for feeding into an advanced aggregation system.

    Among others, the gaussian process aggregator uses this.
    """
    # This should contain the tf--idf vector representation matrix for all the
    # documents in the topic.
    X_all = topic_graph.term_doc_matrix
    # An ordered list of the document ids of the rows in 'X_all'.
    doc_ids = [doc_id for doc_id, text in topic_graph.topic_corpus]
    # This should contain a map from document IDs to their tf--idf vector
    # representation.
    doc_repr = {doc_id: X_all[row] for row, doc_id in enumerate(doc_ids)}

    X_raw = []
    y_raw = []
    for (doc_id, judgements) in all_sampled_votes.items():
        # One training example per vote.
        for judgement in judgements:
            X_raw.append(doc_repr[doc_id])
            assert judgement.is_relevant is not None, \
                "Cannot work with useless votres. See 'JudgementRecord'."
            y_raw.append(int(judgement.is_relevant))

    # We are using sparse tf-idf vectors, so make sure that we are using the
    # right utilities, since e.g. stacking sparse vectors with dense numpy
    # operators such as `np.vstack` leads to weird errors inside sklearn.
    X = sparse.vstack(X_raw)
    y = np.array(y_raw)

    # Approach used in Martin's interop code:
    # y = np.array(labels, dtype=np.float64)[np.newaxis].T

    test_doc_ids = list(kw['ground_truth_docs'])
    X_test_raw = []
    for doc_id in test_doc_ids:
        X_test_raw.append(doc_repr[doc_id])

    X_test = sparse.vstack(X_test_raw)

    return X, y, test_doc_ids, X_test


def aggregate_lm(topic_graph: NxDocumentGraph,
                 all_sampled_votes: Mapping[str, Sequence[JudgementRecord]],
                 **kw):
    """Aggregates votes using a simple linear classifier.

    Highly experimental!
    """
    clf = SGDClassifier()
    X, y, test_doc_ids, X_test = classifier_aggregation_preprocess(
        topic_graph, all_sampled_votes, **kw)

    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        print("Too few labels in one of the classes. Defaulting to vanilla"
              " majority voting.")
        return aggregate_MV(topic_graph, all_sampled_votes)

    clf.fit(X, y)

    # Predict relevance for all documents.
    y_pred = clf.predict(X_test)

    doc_relevance = {}
    for idx, doc_id in enumerate(test_doc_ids):
        boolean_label = (y_pred[idx] >= 0.5)
        doc_relevance[doc_id] = boolean_label

    return doc_relevance


# Local scratch folder that gets deleted automatically when the job is done.
MATLAB_TEMP_DIR = '/tmp/scratch/'


# TODO(andrei): Extract this into dedicated Python MODULE. Document the shit
# out of it and make it easy to add support for pymatbridge or native Python
# GP code in the future!
def aggregate_gpml(topic_graph, all_sampled_votes, **kw):
    X, y, test_doc_ids, X_test = classifier_aggregation_preprocess(
        topic_graph, all_sampled_votes, **kw)

    # Massage the labels into what Matlab is expecting.
    y = [-1 if lbl == 0 else +1 for lbl in y]
    y = np.array(y, dtype=np.float64).reshape(-1, 1)

    random.seed()
    folder_id = random.randint(0, sys.maxsize)

    matlab_folder_name = MATLAB_TEMP_DIR + 'matlab_' + str(folder_id)
    shutil.copytree('matlab', matlab_folder_name)

    io.savemat(matlab_folder_name + '/train.mat', mdict={'x': X, 'y': y})
    io.savemat(matlab_folder_name + '/test.mat', mdict={'t': X_test})

    print("Test data shape: {0}".format(X_test.shape))

    print('Running MATLAB, started %s' % str(datetime.datetime.now()))
    code = subprocess.call(['matlab/run_in_dir.sh', matlab_folder_name])

    if code != 0:
        # TODO(andrei): More details.
        raise OSError('MATLAB code couldn\'t run')

    print('Finished %s' % str(datetime.datetime.now()))

    print('Getting the matrix')

    # Loads a `prob` vector
    prob_location = matlab_folder_name + '/prob.mat'
    print('Loading prob vector from %s' % prob_location)
    mat_objects = io.loadmat(prob_location)
    prob = mat_objects['prob']

    result = prob[:, 0]
    print("Result shape: {0}".format(result.shape))
    print(result)

    doc_relevance = {}
    for idx, doc_id in enumerate(test_doc_ids):
        boolean_label = (result[idx] >= 0.5)
        doc_relevance[doc_id] = boolean_label

    print("Will now quit.")
    exit(-1)

    return doc_relevance


def aggregate_gp(topic_graph, all_sampled_votes, **kw):
    """Performs Gaussian Process aggregation in Python.

    Practically speaking, whenever this gets called, it treats all existing
    sampled votes as training data, and tries to predict the relevance of
    all other nodes (documents).
    """
    raise RuntimeError("Not yet implemented. Still working on the Matlab version.")



