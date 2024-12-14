#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="dfdetect",
    version="0.1",
    description="Deepfake detection tool",
    author="Ismael Balafrej",
    author_email="ismael.balafrej@crim.ca",
    url="",
    packages=find_packages(".", include="dfdetect*"),
)
