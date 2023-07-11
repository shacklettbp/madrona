import importlib.abc
import importlib.util
import sys

class Redirector(importlib.abc.MetaPathFinder):
    def __init__(self, redirections):
        self._map = redirections

    def find_spec(self, fullname, path = None, target = None):
        try:
            redir = self._map[fullname]
        except KeyError:
            return None

        return importlib.util.spec_from_file_location(fullname, redir)

def setup_redirection(redirections):
    sys.meta_path.insert(0, Redirector(redirections))

