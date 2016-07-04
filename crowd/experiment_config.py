"""Helper wrappers for experiment configurations."""

from crowd.simulation import LeastVotesSampler


class ExperimentConfig(object):
    """Describes an experiment using a particular vote aggregator."""

    def __init__(self,
                 vote_aggregator,
                 name,
                 params,
                 nx_graph=False,
                 document_sampler=LeastVotesSampler(),
                 **kw):
        # Whether the underlying graph is an 'NxDocumentGraph' or a regular
        # 'DocumentGraph'.
        self.vote_aggregator = vote_aggregator
        self.name = name
        self.params = params
        self.nx_graph = nx_graph
        self.document_sampler = document_sampler

        self.graph_opts = kw.get('graph_opts', {})
