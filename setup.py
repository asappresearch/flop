"""
setup.py
"""

from setuptools import setup, find_packages
from typing import Dict
import os


NAME = "flop"
AUTHOR = "ASAPP Inc."
EMAIL = "jeremy@asapp.com"
DESCRIPTION = "Pytorch based library for L0 based pruning."


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def required():
    with open('requirements.txt') as f:
        return f.read().splitlines()


# So that we don't import flambe.
VERSION: Dict[str, str] = {}
with open("flop/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setup(

    name=NAME,
    version=os.environ.get("TAG_VERSION", VERSION['VERSION']),

    description=DESCRIPTION,

    # Author information
    author=AUTHOR,
    author_email=EMAIL,

    # What is packaged here.
    packages=find_packages(),

    install_requires=required(),
    dependency_links=[
     "git+git://github.com/asappresearch/sru@custom-submodules#egg=sru",
    ],
    include_package_data=True,

    python_requires='>=3.6.1',
    zip_safe=True

)
