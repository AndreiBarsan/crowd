"""File handling utility functions."""

import io
import os


def get_topic_file_names(base_folder, topic_id):
    topic_folder = os.path.join(base_folder, str(topic_id))
    return [f for f in os.listdir(topic_folder)
            if os.path.isfile(os.path.join(topic_folder, f)) and
            f.endswith(".txt")]


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


def get_all_files(base_folder):
    topic_ids = os.listdir(base_folder)

    all_file_names = []
    for topic_id in topic_ids:
        # This can help skip dummy files which may be present in that folder.
        if os.path.isdir(os.path.join(base_folder, topic_id)):
            all_file_names += [os.path.join(str(topic_id), fname) for fname
                               in get_topic_file_names(base_folder, topic_id)]

    # A map of file names (IDs) to their contents.
    doc_id_to_doc = {}
    corpus = []

    for fname in all_file_names:
        doc_id = fname[fname.rfind('/') + 1:fname.rfind('.')]
        text = read_file(os.path.join(base_folder, fname))
        # A few documents appear in more than one topic. This prevents
        # duplicates and length discrepancies between 'doc_id_to_doc' and
        # 'corpus'.
        if doc_id not in doc_id_to_doc:
            doc_id_to_doc[doc_id] = text
            corpus.append((doc_id, text))

    return doc_id_to_doc, corpus

