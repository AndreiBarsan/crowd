"""Miscellaneous utilities."""

import re
import socket
import subprocess


def get_git_revision_hash() -> str:
    """Useful for experiment reproducibility."""
    # TODO(andrei): Remove this hack
    try:
        return (subprocess
                .check_output(['git', 'rev-parse', '--short', 'HEAD'])
                .decode("utf-8").strip())
    except subprocess.CalledProcessError:
        print("No git available.")
        return 'NO-GIT'


def on_euler() -> bool:
    """Ghetto heuristic to establish whether we're running on ETH's Euler.

    Assumes Euler hostnames start with 'e' and are followed by a numerical
    identifier, such as 'e3286'.
    """

    hostname = socket.gethostname()
    return re.match(r"e[0-9]+", hostname) is not None

