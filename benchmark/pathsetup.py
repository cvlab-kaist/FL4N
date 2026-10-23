"""Import this to make the benchmark's runners/, common/ and datasets/ modules importable by bare name."""
import os
import sys


def add():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("runners", "common", "datasets"):
        p = os.path.join(here, d)
        if p not in sys.path:
            sys.path.insert(0, p)


add()
