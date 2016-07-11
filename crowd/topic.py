"""Utilities for managing topic metadata."""

import xml.etree.ElementTree


from .config import FULLTEXT_FOLDER, TOPIC_DESCRIPTION_FILE
from .file_util import get_topic_file_names


class Topic(object):
    required = ['number', 'query']

    def __init__(self, entries):
        for field in self.required:
            assert field in entries.keys()

        self.document_count = -1
        self.__dict__.update(entries)
        # For consistency with the rest of the code.
        self.topic_id = self.number

    def __repr__(self):
        return "%s:%s" % (self.number, self.query)


def load_topic_metadata(topic_file=TOPIC_DESCRIPTION_FILE):
    """Loads the XML topic metadata, containing information such as query descriptions.

    Returns:
        A map of topic_id -> Topic
    """
    xml_root = xml.etree.ElementTree.parse(topic_file).getroot()
    topic_info = [Topic({field.attrib['name'] : field.text for field in row})
                  for row in xml_root]
    id_topic_info = {topic.number : topic for topic in topic_info}

    for topic_id in id_topic_info:
        document_count = len(get_topic_file_names(FULLTEXT_FOLDER, topic_id))
        id_topic_info[topic_id].document_count = document_count

    return id_topic_info
