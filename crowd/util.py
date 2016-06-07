"""Miscellaneous utilities."""


def get_git_revision_hash():
    """Useful for reproducibility."""
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode("utf-8").strip()
