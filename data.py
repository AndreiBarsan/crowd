"""Data holder classes for the project."""

import io

class JudgementRecord(object):
    """ Judgement record submitted in the 2011 Crowdsourcing Track.

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
        if not relevance == 'na':
            self.is_relevant = (float(relevance) >= 0.5)
        else:
            self.is_relevant = None

    def is_useful(self):
        """Whether this judgement is valid and can be used for aggregation."""
        return self.label_type == 0 and (self.is_relevant is not None)


class WorkerLabel(object):
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
        # 0 (non-relevant), 1 (relevant) or 2 (highly relevant)
        self.label = int(label)

    def is_relevant(self):
        return self.label > 0

    def __repr__(self):
        relevance = "Relevant" if self.is_relevant() else "Not relevant"
        return "%s:%s:%s" % (self.topic_id, self.document_id, relevance)

def read_judgement_labels(file_name):
    with io.open(file_name, 'r') as f:
        return [JudgementRecord(line[:-1]) for line in f]

def read_expert_labels(file_name, header=False, sep=None):
    with io.open(file_name, 'r') as f:
        if header:
            # Skip the header
            f.readline()
        return [ExpertLabel(line.split(sep)) for line in f]

def read_worker_labels(file_name):
    with io.open(file_name, 'r') as f:
        return [WorkerLabel(line) for line in f]
