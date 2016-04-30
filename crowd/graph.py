"""Data holder classes and utility functions for the document similarity graph."""

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from file_util import *


class DocumentEdge(object):
    """ Represents an edge in the document similarity graph.

    """
    def __init__(self, from_document_id, to_document_id, similarity):
        self.from_document_id = from_document_id
        self.to_document_id = to_document_id
        self.similarity = similarity


class DocumentNode(object):
    """ Represents a node (a document) in the document similarity graph.

    Attributes:
        topic_id: The int ID of the topic to which the document belongs.
        document_id: The ID of the document, without any file extension.
        document_name: The full name of the document (usually its ID, plus
            its extension).
        neighbors: A list of 'DocumentEdge' objects, representing this
            node's neighbors.
    """
    def __init__(self, topic_id, document_id, document_name, neighbors):
        self.topic_id = topic_id
        self.document_id = document_id
        self.document_name = document_name
        self.neighbors = neighbors


class DocumentGraph(object):
    """ Represents a graph with documents as nodes, and similarities as
    edges.

    """

    def __init__(self, topic, nodes):
        self.nodes = nodes
        self.topic = topic
        self.topic_id = topic.number
        self.nodes_by_id = {node.document_id: node for node in nodes}

    def get_node(self, document_id):
        return self.nodes_by_id[document_id]

    def __len__(self):
        return len(self.nodes)


def build_document_graph(topic, fulltext_folder, sim_threshold=0.80):
    topic_id = topic.topic_id
    file_names = get_topic_file_names(fulltext_folder, str(topic_id))
    file_names_np = np.array(file_names)
    _, corpus = get_topic_files(fulltext_folder, str(topic_id))

    vectorizer = TfidfVectorizer(min_df=1)
    # Make sure we just pass document texts, and not (doc_id, text) tuples to
    # the tf-idf vectorizer.
    term_doc_matrix = vectorizer.fit_transform([text for doc_id, text in corpus])
    # TODO(andrei) This kernel is a popular choice for computing the
    # similarity of documents represented as tf-idf vectors.
    # cosine_similarity accepts scipy.sparse matrices. (Note that the tf-idf
    # functionality in sklearn.feature_extraction.text can produce normalized
    # vectors, in which case cosine_similarity is equivalent to linear_kernel,
    # only slower.)

    # Automagically computes ALL pairwise cosine similarities between
    # the documents in our corpus.
    similarities = cosine_similarity(term_doc_matrix)

    graph_nodes = []
    total_edges = 0

    # Whether we should print out larger clusters to facilitate manual inspection.
    print_large_clusters = False

    for row_index in range(len(similarities)):
        sims = similarities[row_index]
        doc_id, document = corpus[row_index]

        mask = sims > sim_threshold
        # Make sure we don't have an edge to ourselves, since we're always
        # 100% similar to ourselves.
        mask[row_index] = False
        relevant_sims = sims[mask]
        relevant_docs = file_names_np[mask]

        neighbors = []
        for sim, other_doc_name in zip(relevant_sims, relevant_docs):
            other_doc_id = other_doc_name[:other_doc_name.rfind('.')]
            neighbors.append(DocumentEdge(doc_id, other_doc_id, sim))

        node = DocumentNode(topic_id, doc_id, file_names_np[row_index], neighbors)
        total_edges += len(neighbors)
        graph_nodes.append(node)

        # Sanity check: every document must be 100% similar to itself.
        if not np.allclose(sims[row_index], 1.0):
            # TODO(andrei) Find out why this sometimes happens and report error
            # in a kinder fashion.
            print("WARNING: Found document not similar to itself while "
                  "building document graph; document ID: %s." % doc_id)

        # Explicitly print out larger clusters to facilitate manual inspection.
        if print_large_clusters and len(relevant_sims) > 15:
            print("Document %s has some similar friends!" % doc_id)
            print(list(zip(relevant_sims, relevant_docs)))

    # Note: this treats similarity edges as directed, even though they aren't.
    # Moreover, even though they should be, the edges aren't always 100% "undirected",
    # since (perhaps due to rounding errors) some similarity edges end up being only
    # one-way.
    # print("Built graph with %d total edges." % (total_edges / 2))

    return DocumentGraph(topic, graph_nodes)
