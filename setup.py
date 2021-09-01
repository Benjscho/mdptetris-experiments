import sys

from setuptools import setup
from distutils.core import setup

setup(name='mdptetris_experiments',
    version='0.2.0',
    install_requires=['gym', 'gym_mdptetris'],
    author="Ben Schofield",
    license='MIT',
    packages=['mdptetris_experiments',
            'mdptetris_experiments.agents'],
    zip_safe=False)
