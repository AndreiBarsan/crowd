"""Miscellaneous utilities."""

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
