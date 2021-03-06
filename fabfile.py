"""Fabric deployment file for easy remote experiments.

Uses a Python 3 fork of Fabric (http://www.fabfile.org/).
Please install 'Fabric3' to use this, NOT the vanilla 'fabric'.

```bash
    pip install Fabric3
```

Make sure that 'env.hosts' points to wherever you want to train your model, and
that the remote host has the required dependencies installed.

Examples:
    `fab euler` Syncs your data to Euler and kicks off an experiment.
    `fab euler:status` Shows the status of your Euler jobs.
"""

from __future__ import with_statement

import os

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync
from fabric.state import env as fenv

from crowd.util import get_git_revision_hash

env.use_ssh_config = True


# Hint: set your appropriate user and host for Euler in your '~/.ssh/config'!
@hosts('euler')
def euler(sub='run', label='euler', topic_limit='-1'):
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
        topic_limit: How many topics to process. -1 means all.
    """
    # To pass multiple arguments, to a fabric command, use:
    #  $ fab euler:run,some-label,foo,bar

    if sub == 'run':
        _run_euler(label, topic_limit)
    elif sub == 'status':
        run('bjobs')
    elif sub == 'fetch':
        # The first arg after sub, normally 'label', is now the root folder
        # to sync.
        # TODO(andrei): Get rid of the "master" euler command. It's confusing.
        _download_results(label)
    else:
        raise ValueError("Unknown Euler action: {0}".format(sub))


@hosts('gce-crowd')
def gce(sub='run', label='gce'):
    if sub == 'run':
        print("Will train TF model remotely Google Compute Engine.")
        print("Yes, this MAY cost you real $$$.")
        _run_commodity(label)
    else:
        raise ValueError("Unknown GCE action: {0}".format(sub))


def _run_euler(run_label, topic_limit, aggregation_iterations=160):
    print("Will evaluate system on Euler.")
    print("Euler job label: {0}".format(run_label))
    print("Working in your scratch folder, files unused for 15 days are deleted"
          " automatically!")
    print("Euler (ETHZ) username: {0}".format(fenv['user']))

    work_dir = '/cluster/scratch/{0}/crowd'.format(fenv['user'])
    print("Will work in: {0}".format(work_dir))
    print("Topic limit: {0}".format(topic_limit))

    put(local_path='./remote/euler_voodoo.sh',
        remote_path=os.path.join(work_dir, 'euler_voodoo.sh'))

    _sync_data_and_code(work_dir)

    with cd(work_dir):
        command = ('source euler_voodoo.sh &&'
                   ' bsub -n 48 -W 00:30'
                   # Request 10Gb scratch space per processor to ensure that
                   # the MATLAB interop has enough space to work with.
                   ' -R "rusage[scratch=10000]"'
                   # These flags tell 'bsub' to send an email to the
                   # submitter when the job starts, and when it finishes.
                   ' -B -N'
                   ' "$HOME"/.venv/bin/python3 ' +
                   _run_experiment(run_label, topic_limit, aggregation_iterations))
        print("Will run: " + command)
        run(command, shell_escape=False, shell=False)

        experiment_dir = os.path.join(work_dir, 'experiments')
        fetch_command = "fab euler:fetch,{0}".format(experiment_dir)
        print("After completion, use the following command to sync the "
              "experiment results to your local machine:\n\t{0}\n".format(
                fetch_command))


def _run_commodity(run_label: str, topic_limit=-1, aggregation_iterations=160) -> None:
    """Runs the pipeline on commodity hardware with no LSF job queueing."""
    work_dir = "~/crowd"
    _sync_data_and_code(work_dir)

    with cd(work_dir):
        command = 'python3 ' + _run_experiment(run_label, topic_limit, aggregation_iterations)
        _in_screen(command, 'crowd_screen', shell_escape=False, shell=False)


def _run_experiment(run_label: str,
                    topic_limit: int,
                    aggregation_iterations: int,
                    git_hash=get_git_revision_hash()) -> str:
    """This is command for starting the accuracy evaluation pipeline.

    It is called inside a screen right away when running on AWS, and submitted
    to LFS using 'bsub' on Euler.
    """
    return ('compute_learning_curves.py'
            # This value is set so that it's large enough to achieve reasonable
            # statistical confidence, while also matching the upper CPU limit on Euler.
            ' --aggregation_iterations {0}'
            ' --label "{1}"'
            ' --topic_limit {2}'
            ' --git {3}').format(aggregation_iterations, run_label, topic_limit,
                                 git_hash)


def _sync_data_and_code(work_dir: str) -> None:
    # Ensure we have a trailing slash for rsync to work as intended.
    data_folder = 'data/'
    run('mkdir -p {0}/{1}'.format(work_dir, data_folder))

    # 'os.path.join' does no tilde expansion, and this is what we want.
    remote_folder = os.path.join(work_dir, data_folder)

    # This syncs the data. The custom flags let us see a global progress bar,
    # instead of per-file (spammy) progress.
    rsync(local_dir=data_folder, remote_dir=remote_folder,
          default_opts='-ptrz',
          extra_opts='--info=progress2')

    # This syncs the entry point script.
    put(local_path='./compute_learning_curves.py',
        remote_path=os.path.join(work_dir, 'compute_learning_curves.py'))

    # This syncs the core code.
    rsync(local_dir='crowd/', remote_dir=work_dir + '/crowd')

    # This syncs the matlab stuff.
    rsync(local_dir='matlab/', remote_dir=work_dir + '/matlab')


def _download_results(prefix):
    """Syncs all the output data from the remote host."""
    local('mkdir -p experiments')

    rsync(local_dir='experiments/',
          remote_dir='{0}'.format(prefix),
          default_opts='-ptrz',
          extra_opts='--info=progress2')

    print("Synced all experiment results.")


def _in_screen(cmd: str, screen_name: str, **kw) -> None:
    """Runs the specified command inside a persistent screen.

    The screen persists into a regular 'bash' after the command completes.
    """
    screen = "screen -dmS {} bash -c '{} ; exec bash'".format(screen_name, cmd)
    print("Screen to run: [{0}]".format(screen))
    run(screen, pty=False, **kw)


@hosts('gce-crowd')
def host_type() -> None:
    """An example of a Fabric command."""

    # This runs on your machine.
    local('uname -a')

    # This runs on the remote host(s) specified by the -H flag. If none are
    # specified, this runs on all 'env.hosts'.
    run('uname -a && lsb_release -a')
    run('pwd')
    with cd('/tmp'):
        run('pwd')

