"""Vote aggregation algorithms."""

import random
import typing
from typing import Mapping, Sequence

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


def aggregate_MV_NN(topic_graph, all_sampled_votes, **kw):
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
                     **kw):
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
                          **kw):
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


def aggregate_MV(topic_graph, all_sampled_votes, **kw):
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


def mev_aggregator(topic_graph, doc_id, votes, all_sampled_votes, **kw):
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


def aggregate_mev(topic_graph, all_sampled_votes, **kw):
    """Performs 'MergeEnoughVotes'-style aggregation.

    Returns:
        A map which contains a boolean relevance for every document.
    """
    return {document_id: mev_aggregator(topic_graph, document_id,
                                        document_votes, all_sampled_votes,
                                        **kw)
            for (document_id, document_votes) in all_sampled_votes.items()}


def aggregate_mev_nx(topic_graph, all_sampled_votes, **kw):
    """See: 'aggregate_mev'"""
    return {document_id: mev_aggregator_nx(
        topic_graph,
        document_id,
        document_votes,
        all_sampled_votes,
        **kw) for (document_id, document_votes) in all_sampled_votes.items()}
