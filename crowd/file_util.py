"""File handling utility functions."""

import io
import os


def get_topic_file_names(base_folder, topic_id):
    topic_folder = os.path.join(base_folder, str(topic_id))
    return [f for f in os.listdir(topic_folder)
            if os.path.isfile(os.path.join(topic_folder, f))
            and f.endswith(".txt")]


def read_file(name):
    with io.open(name, 'r', encoding='ISO-8859-1') as file:
        return file.read()


def get_topic_files(base_folder, topic_id):
    topic_folder = os.path.join(base_folder, str(topic_id))
    file_names = get_topic_file_names(base_folder, topic_id)

    # A map of file names (IDs) to their contents.
    doc_id_to_doc = {}
    corpus = []

    for f in file_names:
        doc_id = f[:f.rfind('.')]
        text = read_file(os.path.join(topic_folder, f))
        doc_id_to_doc[doc_id] = text
        corpus.append((doc_id, text))

    return doc_id_to_doc, corpus
