"""Logging utility: mirror stdout to both console and a file."""

import sys
from contextlib import contextmanager


@contextmanager
def redirect_stdout_to_file(file_path):
    """Mirror stdout to the console and a text file.

    Parameters
    ----------
    file_path : str
        Full path of the log file.
    """
    original_stdout = sys.stdout
    with open(file_path, "w", encoding="utf-8-sig") as f:

        class MultiWriter:
            def write(self, text):
                original_stdout.write(text)
                f.write(text)

            def flush(self):
                original_stdout.flush()
                f.flush()

        sys.stdout = MultiWriter()
        try:
            yield
        finally:
            sys.stdout = original_stdout
