"""Data holder classes and utility functions for the project."""

import io
import logging

from typing import Mapping, Sequence, List, Tuple

from crowd.config import *


class JudgementRecord(object):
    """Judgement record submitted in the 2011 Crowdsourcing Track.

    Attributes:
        label_type: Additional label metadata (enum).
            0: default
            1. rejected label: where you would have filtered this
            label out before subsequent use
            2. automated label: label was produced by automation
            (.artificial artificial artificial intelligence.)
            3. training / quality-control label: used in training/evaluating
            worker, not for labeling test data
    """
    def __init__(self, table_row):
        attributes = table_row.split('\t')
        team_id, worker_id, _, topic_id, doc_id, _, relevance, _, _, _, label_type = attributes
        self.team_id = team_id
        self.worker_id = worker_id
        self.label_type = int(label_type)
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.relevance = relevance

        # Relevance can be a floating point number indicating the probability
        # of relevance.
        if not relevance == 'na':
            self.is_relevant = (float(relevance) >= 0.5)
        else:
            self.is_relevant = None

    def is_useful(self):
        """Whether this judgement is valid and can be used for aggregation."""
        return self.label_type == 0 and (self.is_relevant is not None)

    def __repr__(self):
        # TODO(andrei) Solve code duplication.
        if self.is_relevant is None:
            relevance = "n/A"
        elif self.is_relevant:
            relevance = "Relevant"
        else:
            relevance = "Not relevant"
        return "%s:%s:%s" % (self.topic_id, self.doc_id, relevance)


class WorkerLabel(object):
    from warnings import warn
    warn("Using deprecated old worker labels!", category=DeprecationWarning)

    def __init__(self, table_row):
        attributes = table_row.split()
        topic_id, hit_id, worker_id, document_id, nist_label, worker_label = attributes
        self.topic_id = topic_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.document_id = document_id
        self.nist_label = nist_label
        self.worker_label = worker_label


class ExpertLabel(object):
    def __init__(self, attributes):
        if len(attributes) == 3:
            topic_id, document_id, label = attributes
        elif len(attributes) == 4:
            # Also includes set column, which we ignore
            _, topic_id, document_id, label = attributes
        elif len(attributes) == 5:
            # Also includes team and set columns, which we ignore
            _, _, topic_id, document_id, label = attributes
        else:
            raise Exception("Unsupported expert label format: [%s]"
                            % str(attributes))

        self.topic_id = topic_id
        self.document_id = document_id
        # TODO(andrei) Does '-1' just mean 'missing'?
        # 0 (non-relevant), 1 (relevant) or 2 (highly relevant)
        self.label = int(label)

    def is_relevant(self):
        raise ValueError("Don't use this, it's borked (!is_relevant does not "
                         "imply explicit non-relevance, since it's not a binary "
                         "relation: it's ternary since docs can have "
                         "unestablished relevance).")
        return self.label > 0

    def __repr__(self):

        if self.label < 0:
            relevance = "Unknown"
        elif self.label == 0:
            relevance = "Non-relevant"
        elif self.label in [1, 2]:
            relevance = "Relevant"
        else:
            relevance = "Disallowed: [{0}]".format(self.label)
        return "%s:%s:%s" % (self.topic_id, self.document_id, relevance)


def read_judgement_labels(file_name: str) -> List[JudgementRecord]:
    with io.open(file_name, 'r') as f:
        return [JudgementRecord(line[:-1]) for line in f]


def read_useful_judgement_labels(file_name: str=JUDGEMENT_FILE) -> List[JudgementRecord]:
    return [l for l in read_judgement_labels(file_name) if l.is_useful()]


def read_expert_labels(file_name, header=False, sep=None) -> List[ExpertLabel]:
    with io.open(file_name, 'r') as f:
        if header:
            # Skip the header
            f.readline()
        return [ExpertLabel(line.split(sep)) for line in f]


def read_worker_labels(file_name: str) -> List[WorkerLabel]:
    logging.warning("Using old worker labels!")
    with io.open(file_name, 'r') as f:
        return [WorkerLabel(line) for line in f]


def read_ground_truth() -> List[ExpertLabel]:
    """Reads the 2011 test label data files, which are used as the ground truth
    in our evaluation."""
    return read_expert_labels(TEST_LABEL_FILE_SHARED, header=True, sep=',') + \
        read_expert_labels(TEST_LABEL_FILE_TEAMS, header=True, sep=',')


def get_known_labels(ground_truth_data: Sequence[ExpertLabel]) -> Tuple[Sequence[ExpertLabel], Sequence[ExpertLabel]]:
    """Returns all relevant and non-relevant documents in the ground truth data.

    Skips judgements in the ground truth where the label is negative, i.e.
    missing.
    """

    relevant_documents = {j.document_id for j in ground_truth_data if j.label > 0}
    non_relevant_documents = {j.document_id for j in ground_truth_data
                              if j.label == 0}

    return relevant_documents, non_relevant_documents


def get_relevant(topic_id: str, ground_truth_data: Sequence[ExpertLabel]) -> Tuple[Sequence[ExpertLabel], Sequence[ExpertLabel]]:
    """ Returns a set of relevant and a set of non-relevant expert labels from
    the specified topic.
    """
    topic_ground_truth_data = [j for j in ground_truth_data
                               if j.topic_id == topic_id]
    # TODO(andrei) Fix issue with 'is_relevant()' function for labels == -1.
    return get_known_labels(topic_ground_truth_data)


def get_all_judgements_by_doc_id(turk_judgements: Sequence[JudgementRecord]):
    judgements_by_doc_id = {}
    for j in turk_judgements:
        if j.doc_id not in judgements_by_doc_id:
            judgements_by_doc_id[j.doc_id] = []

        judgements_by_doc_id[j.doc_id].append(j)

    return judgements_by_doc_id


def get_topic_judgements_by_doc_id(topic_id, judgements) -> Mapping[str, Sequence[JudgementRecord]]:
    topic_judgements = [j for j in judgements if j.topic_id == topic_id]
    topic_judgements_by_doc_id = {}
    for j in topic_judgements:
        if j.doc_id not in topic_judgements_by_doc_id:
            topic_judgements_by_doc_id[j.doc_id] = []

        topic_judgements_by_doc_id[j.doc_id].append(j)

    return topic_judgements_by_doc_id
