#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.6 or later
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
The placement of stress in text using a recurrent neural network BiLSTM. It's a modified
version of RusStress (https://github.com/MashaPo/russtress). Only Russian is supported.

Contains the 'StressRNN' and 'ExceptionDictWrapper' classes. Learn more in https://github.com/Desklop/StressRNN.

Dependences: numpy<1.19.0,>=1.16.0 scipy<=1.5.2 tensorflow<=2.3.1 pymorphy2[fast]<=0.9.2
'''

from .stressrnn import StressRNN
from .exception_dictionary_wrapper import ExceptionDictWrapper