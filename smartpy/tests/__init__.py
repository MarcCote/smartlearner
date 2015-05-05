from contextlib import contextmanager
import os
import shutil


@contextmanager
def tmp_folder(path):
    """ Context manager for creating a temporary folder. """
    os.mkdir(path)
    yield path
    shutil.rmtree(path)
