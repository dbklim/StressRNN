#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O., MashaPo

from setuptools import setup
from typing import List


def readme() -> List[str]:
    with open('README.md', 'r', encoding='utf-8') as f_readme:
        return f_readme.read()


def requirements() -> List[str]:
    with open('requirements.txt', 'r', encoding='utf-8') as f_requirements:
        return f_requirements.read()


setup(name='stressrnn',
      version='0.2.2',
      description='Package that helps you to put lexical stress in russian text.',
      long_description=readme(),
      url='https://github.com/Desklop/StressRNN',
      classifiers=[
          'Natural Language :: Russian',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Text Processing :: Linguistic'
      ],
      author='Maria Ponomareva (RusStress), Kirill Milintsevich (RusStress), Vladislav Klim (StressRNN)',
      keywords='nlp russian stress accent emphasis linguistic rnn lstm bilstm',
      author_email='ponomarevamawa@gmail.com (RusStress), vladsklim@gmail.com (StressRNN)',
      packages=['stressrnn'],
      # Add py.typed to the package
      package_data={'stressrnn': ['py.typed']},
      install_requires=requirements(),
      include_package_data=True,
      zip_safe=False)

# python3 setup.py bdist_wheel