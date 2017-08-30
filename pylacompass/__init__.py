"""Initialization file for the pylacompass module"""
from .pylacompass import read_data, read_input, data_dict, read_hdf5_file
from setuptools_scm import get_version as __get_version

__version__ = __get_version(root='..', relative_to=__file__)

__all__ = ['read_data', 'read_input', 'data_dict', 'read_hdf5_file']
