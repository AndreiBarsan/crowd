import os

# This should be the root folder containing the different judgement datasets,
# including the vote data, the expert judgements (ground truth), and the
# document text (the hard-to-get part of the dataset; sorry I can't share it
# directly, but it's disallowed...).
DATA_ROOT = 'data'

# This file contains exclusively the NIST expert judgements from 'stage1-dev'
# 'cat'ed together into a single file (from all the teams, as well as the
# common data).
EXPERT_GROUND_TRUTH_FILE = os.path.join(DATA_ROOT, 'ground_truth')

# This file contains the document labels computed by the Mechanical Turk
# workers.
# Unlike the NIST expert judgement file, this one is provided from the
# beginning as just one file (yay!). Contains the development data for the
# second part of the challenge (consensus).
WORKER_LABEL_FILE = os.path.join(DATA_ROOT, 'stage2-dev', 'stage2.dev')

# Mechanical Turk worker judgements for the 2011 Crowdsourcing Track.
JUDGEMENT_FILE = os.path.join(DATA_ROOT, 'all_judgements.tsv')

# Provided test data for the 1st stage of the TREC 2011 Crowdsourcing Track.
TEST_LABEL_FILE_SHARED = os.path.join(DATA_ROOT, 'test-set-Aug-8', 'trec-cs-2011-test-set-shared.csv')
TEST_LABEL_FILE_TEAMS = os.path.join(DATA_ROOT, 'test-set-Aug-8', 'trec-cs-2011-test-set-assigned-to-teams.csv')

TOPIC_DESCRIPTION_FILE = os.path.join(DATA_ROOT, 'topics.mq09.cs2011-test-topics.xml')
FULLTEXT_FOLDER = os.path.join(DATA_ROOT, 'url-header-html-txt')

# Constant dict key used when manipulating the MATLAB driver (i.e. the proxy
# which allows interoperability wiht MATLAB, as required by e.g. the
# Gaussian Processes code, which relies on the GPML library).
MATLAB_DRIVER_FACTORY_KEY = 'matlab_driver'
