"""Fabric deployment file for easy remote experiments.

Uses a Python 3 fork of Fabric (http://www.fabfile.org/).
Please install 'Fabric3' to use this, NOT the vanilla 'fabric'.

```bash
    pip install Fabric3
```

Make sure that 'env.hosts' points to wherever you want to train your model, and
that the remote host has the required dependencies installed.

Examples:
    TODO(andrei): Some examples.
"""

from __future__ import with_statement

import os

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync

env.use_ssh_config = True

# Hint: set your appropriate user and host for Euler in your '~/.ssh/config'!
@hosts('euler')
def euler(sub='run', label='euler'):
    """
    Submits the pipeline to Euler's batch job system.

    Arguments:
        sub: What action to perform. Can be 'run' for running the pipeline,
             'status' for seeing the job status on Euler, or 'fetch' to download
             the experiment results (experimental feature).
        label: An informative label for the job. MUST be a valid file name
               fragment, such as 'preprocess-v2-bob'. Does NOT get
               shell-escaped, so use special characters (e.g. spaces, $, etc.)
               at your own risk!
    """
    # To pass multiple arguments, to a fabric command, use:
    #  $ fab euler:run,some-label,foo,bar

    if sub == 'run':
        _run_euler(label)
    elif sub == 'status':
        run('bjobs')
    elif sub == 'fetch':
        raise ValueError("Not yet implemented.")
    else:
        raise ValueError("Unknown Euler action: {0}".format(sub))


def _run_euler(run_label):
    print("Will evaluate system on Euler.")
    print("Euler job label: {0}".format(run_label))


    # TODO(andrei): See if this is necessary!
    put(local_path='./remote/euler_voodoo.sh',
        remote_path=os.path.join('~/deploy', 'euler_voodoo.sh'))

    _sync_data_and_code()

    with cd('crowd'):
        # TODO(andrei): Run on scratch instead of in '~', since the user root
        # on Euler only has a quota of 20Gb but scratch is fuckhuge.
        # TODO(andrei): Warn when writing to scratch, since files in scratch get
        # cleared out every 15 days.
        # Creates a timestamped folder in which to run.
        ts = '$(date +%Y%m%dT%H%M%S)'
        # Hint: Replace the "heavy" 'train_model' call with 'tensor_hello' if
        # you just want to test things out.
        tf_command = ('t=' + ts + ' && mkdir $t && cd $t &&'
                      ' source ../euler_voodoo.sh &&'
                      ' bsub -n 48 -W 4:00'
                      # These flags tell 'bsub' to send an email to the
                      # submitter when the job starts, and when it finishes.
                      ' -B -N'
                      ' LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" "$HOME"/ext/lib/ld-2.23.so "$HOME"/.venv/bin/python3'
                      + _run_experiment(run_label))
        run(tf_command, shell_escape=False, shell=False)


def _run_experiment(run_label: str) -> str:
    """This is command for starting the accuracy evaluation pipeline.

    It is called inside a screen right away when running on AWS, and submitted
    to LFS using 'bsub' on Euler.
    """
    # TODO(andrei): Pass these all these parameters as arguments to fabric.
    return (' ../compute_learning_curves.py'
            ' --label "' + run_label + '"')


def _sync_data_and_code():
    # TODO(andrei): '--progress' flag for rsync or pipe through 'pv'.
    run('mkdir -p ~/deploy/data/preprocessing')

    # Ensure we have a trailing slash for rsync to work as intended.
    folder = os.path.join('data', 'preprocessing') + '/'
    # 'os.path.join' does no tilde expansion, and this is what we want.
    remote_folder = os.path.join('~/deploy', folder)

    # This syncs the data (needs to be preprocessed in advance).
    rsync(local_dir=folder, remote_dir=remote_folder, exclude=['*.txt'])

    put(local_path='./train_model.py',
        remote_path=os.path.join('~/deploy', 'train_model.py'))

    # This syncs the model code.
    rsync(local_dir='model', remote_dir='deploy')


# def latest_run_id():
#     """Returns the ID of the most recent TF run."""
#     # TODO(andrei): Nicer way of doing this?
#     return "ls -t ~/deploy/data/runs | cat | head -n1"


# def _download_results(prefix):
#     """Downloads all the TF output data from the remote host."""
#     local('mkdir -p data/runs/{0}'.format(prefix))
#
#     # TODO(andrei): Nicer folder structure.
#     # TODO(andrei): Random tmp folder for maximum Euler compatibility.
#     run('mkdir -p /tmp/last_tf_run')
#     run('cp -R ~/deploy/data/runs/$({})/ /tmp/last_tf_run'.format(latest_run_id()),
#         shell_escape=False, shell=False)
#     get(remote_path='/tmp/last_tf_run/*',
#         local_path='data/runs/{0}'.format(prefix))
#     print("Downloaded the pipeline results.")


def _in_screen(cmd, screen_name, **kw):
    """Runs the specified command inside a persistent screen.

    The screen persists into a regular 'bash' after the command completes.
    """
    screen = "screen -dmS {} bash -c '{} ; exec bash'".format(screen_name, cmd)
    print("Screen to run: [{0}]".format(screen))
    run(screen, pty=False, **kw)


@hosts('euler')
def host_type():
    """An example of a Fabric command."""

    # This runs on your machine.
    local('uname -a')

    # This runs on the remote host(s) specified by the -H flag. If none are
    # specified, this runs on all 'env.hosts'.
    run('uname -a && lsb_release -a')
    run('pwd')
    with cd('/tmp'):
        run('pwd')

