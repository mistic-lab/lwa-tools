import setuptools

try:
    import numpy
except ModuleNotFoundError as e:
    raise ImportError("Due to quirks with LSL, numpy needs to be installed prior to installing lwatools. Please run pip install numpy and try again.")


setuptools.setup()
