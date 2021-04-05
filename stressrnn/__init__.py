#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
The placement of stress in text using a recurrent neural network BiLSTM. It's a modified
version of RusStress (https://github.com/MashaPo/russtress). Only Russian is supported.

Contains the 'StressRNN' and 'ExceptionDictWrapper' classes. Learn more in https://github.com/Desklop/StressRNN.

Dependences:
    - for ONNX Runtime: numpy>=1.16.0 pymorphy2[fast]<=0.9.2 onnxruntime<=1.7
    - for TensorFlow: numpy>=1.16.0 pymorphy2[fast]<=0.9.2 tensorflow<=2.3.1
'''

from .stressrnn import StressRNN
from .exception_dictionary_wrapper import ExceptionDictWrapper