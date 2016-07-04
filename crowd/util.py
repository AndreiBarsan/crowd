"""Miscellaneous utilities."""

import subprocess


def get_git_revision_hash():
    """Useful for experiment reproducibility."""
    return (subprocess
            .check_output(['git', 'rev-parse', '--short', 'HEAD'])
            .decode("utf-8").strip())
