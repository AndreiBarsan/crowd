"""Code for document selection using information diffusion."""

from datetime import datetime
import logging
import heapq
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from crowd.aggregation import aggregate_mev_nx
from crowd.config import *
from crowd.graph import build_nx_document_graph, NxDocumentNode
from crowd.simulation import DocumentSampler, evaluate, LeastVotesSampler
from crowd.data import *
from crowd.topic import load_topic_metadata
from crowd.util import get_git_revision_hash


class Reachability(object):
    def compute_reachability(
        self,
        sampled_graph: nx.Graph,
        seed_set: Sequence[NxDocumentNode]
    ) -> Sequence[NxDocumentNode]:
        raise RuntimeError("Ghetto abstract method. Do not use.")


class DefaultReachability(Reachability):
    """Simple implementation for computing the reachability of a seed set."""

    def compute_reachability(self, sampled_graph: nx.Graph, seed_set: Sequence[NxDocumentNode]) -> Sequence[NxDocumentNode]:
        # Compute the reachability given the seed set and the nodes we sampled.
        all_reached = set()
        for seed_node in seed_set:
            # TODO(andrei): If we care about reachability, why not just use BFS?
            # TODO(andrei): Time this and see if vanilla BFS is faster!
            # nx.bfs_successors()
            # TODO(andrei): The next step would be to implement BFS in C++ or
            # at least Cython, since networkx doesn't do that. The idea is we
            # only need nodes and edges, and we don't care about content,
            # weights, or anything like that => could pass this raw data to
            # lower layers quite easily.
            reached = set(nx.shortest_path(sampled_graph, source=seed_node).keys())
            all_reached |= reached

        return all_reached


def sample_edges_ic(graph: nx.Graph, seed_set: Sequence[NxDocumentNode]) -> Sequence[NxDocumentNode]:
    """
    Samples every edge in the graph with the probability = similarity.

    Args:
        graph: The graph on which we should operate.
        seed_set: A set of nodes which are 'active' at the beginning.

    Returns:
        The set of reachable nodes given the seed set and the edges we sampled.
    """

    sampled = nx.Graph(graph)
    sampled.remove_edges_from(sampled.edges())

    for from_node, to_node, data in graph.edges(data=True):
        if random.random() <= data['similarity']:
            # Note: Batching this (i.e. only add all sampled at once) actually
            # slows things down under NetworkX...
            sampled.add_edge(from_node, to_node)

    all_reached = DefaultReachability().compute_reachability(sampled, seed_set)
    return all_reached


def sample_edges_lt(graph: nx.Graph, seed_set: Sequence[NxDocumentNode]) -> Sequence[NxDocumentNode]:
    """Linear Threshold Model."""

    # TODO(andrei): implement original LT model as described in Kempe et al.

    sampled = nx.Graph(graph)
    sampled.remove_edges_from(sampled.edges())

    # TODO(andrei): Consider doing preprocessing and e.g. probability
    # normalizations somewhere else, to avoid doing it every time. This may be
    # a serious bottleneck, so it could be useful to roll your own
    # implementation.

    # Each node select at most one incoming edge, with probability proportional
    # to its weight. Since our graph is not directed, there's no difference
    # between incoming and outgoing edges.
    for node in graph:
        # TODO(andrei): Numpy-ify!
        others = []
        sims = []
        for other, edge in graph[node].items():
            sim = edge['similarity']
            # edges.append(edge)
            others.append(other)
            sims.append(sim)

        if len(others) == 0:
            # No edge to sample.
            continue

        probs = np.array(sims)
        probs /= np.sum(probs)

        kept_other = np.random.choice(others, p=probs)
        sampled.add_edge(node, kept_other)

    all_reached = Reachability().compute_reachability(sampled, seed_set)
    return all_reached


def simulate_spread(graph, seed_set, iteration_count: int, sample_edges):
    """Simulates information spread in 'graph' using a particular model.

    Args:
        graph: The graph on which to operate.
        seed_set: The nodes which are active from the beginning.
        iteration_count: How many edges to sample before stopping.
        sample_edges: A function which performs edge sampling, such as
                      'sample_edges_ic'.

    Returns:
        The influence spread of the given seed set in the graph.
        This is a number between 0 and the size of the graph signifying
        the number of nodes we expect to reach from the seed set in the
        given graph, approximated using 'iteration_count' iterations.
    """
    reach_sum = 0.0

    for k in range(iteration_count):
        result = sample_edges(graph, seed_set)

        # TODO(andrei): Using naive lengths means nodes have no weights. One
        # can dynamically set node weights to fine-tune the algorithm. For
        # instance, a node's weight could be set proportional to the uncertainty
        # in the node, prioritizing nodes with absolutely no information, but
        # still giving a boost to nodes which have dissenting votes, as opposed
        # to nodes where all voters agree.
        reach_sum += len(result)

    expected_reach = reach_sum / iteration_count
    return expected_reach


# TODO(andrei): Use this to test lazy greedy implementation. Should be quite
# easy.
def pick_next_best(graph, current_seed_set, iteration_count):
    """Picks the node which best improves the information spread of the seed set.

    This is a greedy approach, which works because the influence spread function
    is submodular.
    """

    # TODO(andrei): Consider just deleting this function, since it's not used
    # any more, and the lazy greedy one should be preferred.

    # TODO(andrei): Parameterize this!
    edge_sampler = sample_edges_ic

    best_spread = -1
    best_node = None
    for node in graph.nodes():
        if node not in current_seed_set:
            expected_spread = simulate_spread(
                graph,
                # TODO(andrei): Is there a better way to do this?
                current_seed_set | {node},
                iteration_count,
                edge_sampler)

            if expected_spread > best_spread:
                best_spread = expected_spread
                best_node = node

    assert best_node is not None, "Must select a node to add to the seed set."
    return best_node, best_spread


def build_seed_set(graph, budget, iteration_count):
    seed_set = set()
    for index in range(budget):
        # TODO(andrei): Consider detecting when spread stops improving and quitting early. Does it ever happen?
        best_node, best_spread = pick_next_best(graph, seed_set,
                                                iteration_count)
        print("Budget: {}/{}, Spread: {:.2f}".format(index + 1, budget,
                                                     best_spread))
        seed_set.add(best_node)

    return seed_set


def compute_best_heap(graph, current_seed_set, prev_spread, iteration_count):
    """Similar to 'pick_next_best', but returns a priority queue of all
    candidates."""

    # TODO(andrei): Parameterize this!
    edge_sampler = sample_edges_ic

    spread_node_heap = []
    for node in graph.nodes():
        # Skip nodes which are not safe to sample because they have no real votes.
        #         if node.document_id not in safe_sample_set:
        #             continue

        if node not in current_seed_set:
            expected_benefit = simulate_spread(
                graph,
                current_seed_set | {node},
                iteration_count,
                edge_sampler)
            assert prev_spread <= 0
            node_marginal_benefit = expected_benefit - (-prev_spread)
            # By default, python's heap is a min-heap, so we need to invert this here sign.
            # TODO(andrei): Is the negative hack needed? Consider implementing a min/max-heap wrapper.
            # Implementing a dedicated max-heap will also make your code much clearer.
            heapq.heappush(spread_node_heap,
                           (-node_marginal_benefit, -expected_benefit, node))

    assert len(spread_node_heap) > 0, "Must find at least one candidate node."
    return spread_node_heap


# TODO(andrei): The algorithm in its early stages is slowed down by cliques
# in the graph => bad lazy greedy performance. Write about this in the
# report.
# The same happens in the later stages: there we are only sampling
# unconnected nodes (1-cliques).
def pick_next_best_lazy(graph, current_seed_set, iteration_count,
                        previous_best_heap, prev_spread, stats, **kw):
    """
    Args:
        graph: The graph from which to sample.
        current_seed_set: The documents which are "already picked" when the
                          algorithm starts.
        iteration_count: The number of times to repeat the sampling process
                         inside the graph.
        previous_best_heap: The heap computed at a previous step, or an empty
                            sequence.
        prev_spread: Information spread at the previous time step.
        stats: A dictionary to which various stats get written.

    Keyword Args:
        epsilon: Trick to make many more lazy greedy "hits" happen, at a
                 very small cost to the total accuracy. Applies this as a
                 "penalty" to the second-best score so that it's "easier" for
                 the lazy option to be picked.

    Returns:
        The new heap, as a Python list.
    """
    # TODO(andrei): Re-add safe sample set.
    # safe_sample_set: The IDs we are actually allowed to try out, since we can't pick
    # any node because many don't have any votes to sample from in our simulation!!!

    if len(previous_best_heap) < 2:
        # Heap not enough for any sensible lazy greediness.
        return compute_best_heap(graph, current_seed_set, 0, iteration_count)

    epsilon = kw.get('epsilon', 0.5)
    edge_sampler = kw.get('edge_sampler', sample_edges_ic)

    # The first value in the previous best heap has already been added to the
    # seed set.
    best, second_best = heapq.nsmallest(2, previous_best_heap)
    # prev_delta, prev_spread, prev_added_node = prev
    best_delta, best_spread, best_node = best
    second_best_delta, second_best_spread, second_best_node = second_best

    recomputed_best_score = simulate_spread(graph,
                                            current_seed_set | {best_node},
                                            iteration_count,
                                            edge_sampler)
    # Note: prev_spread is negative!
    recomputed_best_delta = -(recomputed_best_score + prev_spread)

    if recomputed_best_delta <= second_best_delta + epsilon:
        # print("Recomputed best delta <= {0}".format(second_best_delta + epsilon))
        # print("Next node should be: {0}".format(best_node))
        # print("Second after that:   {0}".format(second_best_node))

        stats['hit'] += 1
        # We succeeded in being lazy! Update the delta and score for the best
        # element. Note: heapreplace returns the smallest!
        _ = heapq.heapreplace(
            previous_best_heap,
            # The '- EPSILON' term ensures we actually pick this element when
            # popping from the heap.
            (recomputed_best_delta - epsilon, -recomputed_best_score, best_node))

        return previous_best_heap
    else:
        stats['miss'] += 1
        # Need to do a full recompute. Oh well.
        heap = compute_best_heap(graph, current_seed_set, prev_spread,
                                 iteration_count)
        return heap


def build_seed_set_lg(graph, budget, iteration_count):
    """Used for debugging graph sampling."""
    best_heap = []
    seed_set = set()
    best_spread_delta = 0
    best_spread = 0
    best_node = None
    stats = {'hit': 0, 'miss': 0}
    for index in range(budget):
        prev_spread = best_spread
        best_heap = pick_next_best_lazy(graph, seed_set, iteration_count,
                                        best_heap, prev_spread, stats)
        best_spread_delta, best_spread, best_node = heapq.heappop(best_heap)
        logging.debug("Budget: {}/{}, Spread: {:.2f}".format(index + 1,
                                                             budget,
                                                             best_spread))
        seed_set.add(best_node)
        print("New best node: {0}".format(best_node))
        print("Biggest delta: {0}".format(best_spread_delta))
        # print("Seed set: {0}".format(sorted([s.document_id[-5:] for s in seed_set])))
        print()

    return seed_set, stats, best_spread


class GraphSpreadSampler(DocumentSampler):
    """Tries to sample nodes which maximize information spread.

    Uses a greedy approach as the objective function we are trying
    to maximize under the information diffusion model is submodular.

    Note: When parallelizing computations, make sure that each worker
    gets its own sampler, as they are stateful. Failure to do so leads
    to race conditions.
    """

    def __init__(self, topic_graph, **kw):
        super().__init__()
        self.topic_graph = topic_graph
        self.seed_set = set()
        self.iteration_count = kw.get('iteration_count', 5)
        random.seed(0x789)

    def sample(self, existing_votes, available_topic_judgements):
        raise RuntimeError("DO NOT USE THIS. IT IS BUGGY.")
        best_node, best_spread = pick_next_best(
            self.topic_graph.nx_graph,
            self.seed_set,
            self.iteration_count)
        self.seed_set.add(best_node)
        return best_node.document_id


class LazyGreedyGraphSpreadSampler(DocumentSampler):
    """Same as 'GraphSpreadSampler' but in a lazy greedy fashion."""

    def __init__(self, topic_graph, **kw):
        super().__init__()
        self.topic_graph = topic_graph
        self.seed_set = set()
        self.iteration_count = kw.get('iteration_count', 5)
        self.best_heap = []
        self.best_spread_delta = 0
        self.best_spread = 0
        self.stats = {'hit': 0, 'miss': 0}

        # TODO(andrei): Better name for this.
        self.edge_sampler = kw.get('edge_sampler', sample_edges_ic)

        random.seed(0x789)

    def sample(self, existing_votes, available_topic_judgement):
        prev_spread = self.best_spread
        self.best_heap = pick_next_best_lazy(
            self.topic_graph.nx_graph,
            self.seed_set,
            self.iteration_count,
            self.best_heap,
            prev_spread,
            # Limit the nodes we're allowed to sample to the ones for which
            # we have votes, without getting rid of nodes with no info
            # altogether, as they may be the way towards other interesting
            # nodes in our information diffusion model.
            # TODO(andrei): Re-enable this once you speed the code up.
            # available_topic_judgement.keys(),
            self.stats,
            edge_sampler=self.edge_sampler)

        _, self.best_spread, self.best_node = heapq.heappop(self.best_heap)
        # print("Budget: {}/{}, Spread: {:.2f}".format(index + 1, budget, best_spread))
        self.seed_set.add(self.best_node)

        if self.best_node.document_id not in available_topic_judgement:
            print(available_topic_judgement)
            raise ValueError("Sanity check failed: tried to sample an"
                             " unavailable document ID.")

        return self.best_node.document_id


def lgss_graph_factory(iteration_count, **kw):
    """Proxy for ensuring that when computing things in parallel, every worker
    has its own sampler. This is necessary as these samplers are stateful."""
    def res(graph):
        return LazyGreedyGraphSpreadSampler(graph,
                                            iteration_count=iteration_count,
                                            **kw)
    return res


def compare_sampling(tid, sim_threshold, discard_empty_nodes=True):
    # TODO(andrei): Do we still need this function? It's only called from the
    # 'Graph Sampling' notebook, which may still be useful.

    id_topic_info = load_topic_metadata()
    judgements = read_useful_judgement_labels(JUDGEMENT_FILE)
    test_data = read_ground_truth()

    print(
        "Comparing least-votes and graph-based sampling for topic: {} (ID#{})".format(
            id_topic_info[tid].query,
            tid))
    nx_graph = build_nx_document_graph(
        id_topic_info[tid],
        test_data,
        get_topic_judgements_by_doc_id(tid, judgements),
        FULLTEXT_FOLDER,
        sim_threshold=sim_threshold,
        discard_empty=discard_empty_nodes)
    topic_judgements = get_topic_judgements_by_doc_id(tid, judgements)
    topic_ground_truth = {truth.document_id: truth for truth in test_data
                          if truth.topic_id == tid and truth.label >= 0}

    print("Judgements: {}".format(len(topic_judgements.keys())))
    print("Ground truths: {}".format(len(topic_ground_truth.keys())))

    graph_simulation_iterations = 5
    aggregation_iterations = 10
    print("Computing results for IC Information Spread Sampler...")

    ic_graph_result, ic_time_s = evaluate(
        nx_graph,
        topic_judgements,
        topic_ground_truth,
        lgss_graph_factory(graph_simulation_iterations),
        aggregate_mev_nx,
        budget=len(topic_judgements.keys()),
        iterations=aggregation_iterations)
    ic_sampling_info = "{0} info propagation simulation iterations; {1} graph empty nodes; {2:.4f} s".format(
        graph_simulation_iterations,
        ("Discarded" if discard_empty_nodes else "Not discarded"),
        ic_time_s)

    print("Computing results for LeastVotesSampler...")

    least_votes_result, lv_time_s = evaluate(
        nx_graph,
        topic_judgements,
        topic_ground_truth,
        LeastVotesSampler(),
        aggregate_mev_nx,
        budget=len(topic_judgements.keys()),
        iterations=aggregation_iterations)

    print("Plotting...")
    plt.rcParams['figure.figsize'] = (18, 10)

    # TODO(andrei): More info on plot.
    now = datetime.now()
    ic_graph_avg = np.mean(ic_graph_result, axis=0)
    least_votes_avg = np.mean(least_votes_result, axis=0)
    plt.plot(range(len(ic_graph_avg)), ic_graph_avg,
             label="MEV(1) and info spread sampling", marker='^')
    plt.plot(range(len(least_votes_avg)), least_votes_avg,
             label="MEV(1) and least-votes")
    title = "Sampling technique comparison; IC sampling: {}; Date: {}; Git: {}; Graph similarity threshold: {}; Iterations: {}".format(
        ic_sampling_info,
        now.strftime("%Y-%m-%d %H:%M"),
        get_git_revision_hash(),
        sim_threshold,
        aggregation_iterations)
    plt.title(title)
    plt.grid()
    plt.xlabel("Total sampled votes")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right', fontsize=16)

    # TODO(andrei): Also save plot!
    #     now = datetime.now()
    #     plotname = "sampling-comparison-{}-{}-{}-{}-{}".format(
    #         tid,
    #         now.strftime("%Y%m%dT%H%M"),
    #         get_git_revision_hash(),
    #         sim_threshold,
    #         aggregation_iterations)
    #     plt.savefig('../plots/{}.svg'.format(plotname))

