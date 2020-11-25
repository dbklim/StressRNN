#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.6 or later
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Exception dictionary for correcting stress placement by a neural network.

Contains the 'ExceptionDictWrapper' class. Learn more in https://github.com/Desklop/StressRNN.

Dependences: pymorphy2[fast]<=0.9.2
'''

import pymorphy2

try:
    from .constants import *
except ImportError:
    from constants import *


class ExceptionDictWrapper:
    ''' Exception dictionary for correcting stress placement. Contains methods:
    - is_in_dict(): checking if a word is in the dictionary
    - put_stress(): placing stress in a word in accordance with the dictionary

    The exception dictionary must contain a list of words with stresses placed in them using the "'" or '+' symbol after the vowel
    (1 line = 1 word).

    The dictionary looks like this:
        дре+гер
        ивано+в
        ...

    The main dictionary comes with the package and is located in 'stressrnn/exception_dictionary.txt'. You can also add your own
    dictionary, the values from which will complement the main dictionary (and overwrite the same words, but with different stresses).

    1. f_name_add_exception_dict - name of additional .txt dictionary with exceptions '''

    def __init__(self, f_name_add_exception_dict: str = None) -> None:
        self.morph_analyzer = pymorphy2.MorphAnalyzer()
        self.exception_dict = {}

        self.__load_exception_dict(F_NAME_EXCEPTION_DICT)

        if f_name_add_exception_dict:
            self.__load_exception_dict(f_name_add_exception_dict, overwrite=True)


    def __load_exception_dict(self, f_name_exception_dict: str, overwrite: bool = False) -> None:
        ''' Loading a dictionary from a .txt file and creating pairs of the form 'word': [stress_position]. The file can contain
        several identical words with different stresses, they will all be added to the dictionary, and the stress positions will be
        specified in the order of reading the words. '''

        with open(f_name_exception_dict, 'r') as f_exception_dict:
            for word in f_exception_dict:
                word = word.strip('\n')
                if DEF_STRESS_SYMBOL in word:
                    unstressed_word = word.replace(DEF_STRESS_SYMBOL, '')
                    if not overwrite and unstressed_word in self.exception_dict:
                        self.exception_dict[unstressed_word].append(word.find(DEF_STRESS_SYMBOL))
                    else:
                        self.exception_dict[unstressed_word] = [word.find(DEF_STRESS_SYMBOL)]
                elif ADD_STRESS_SYMBOL in word:
                    unstressed_word = word.replace(ADD_STRESS_SYMBOL, '')
                    if not overwrite and unstressed_word in self.exception_dict:
                        self.exception_dict[unstressed_word].append(word.find(ADD_STRESS_SYMBOL))
                    else:
                        self.exception_dict[unstressed_word] = [word.find(ADD_STRESS_SYMBOL)]


    def is_in_dict(self, word: str, lemmatize_word: bool = False) -> bool:
        ''' Checking if the word is in the dictionary.

        1. word - string with the word of interest
        2. lemmatize_word - True: lemmatize (normalize) word before searching in dictionary
        3. returns True/False '''

        if word.lower() in self.exception_dict:
            return True
        elif word.lower().replace('ё', 'е') in self.exception_dict:
            return True
        elif lemmatize_word and self.morph_analyzer.parse(word)[0].normal_form in self.exception_dict:
            return True
        else:
            return False


    def put_stress(self, word: str, stress_symbol: str, lemmatize_word: bool = False) -> str:
        ''' Put stress in a word in accordance with the dictionary. Stress is indicated by stress_symbol after the stressed vowel.
        
        1. word - string with the word of interest
        2. stress_symbol - stress symbol
        3. lemmatize_word - True: lemmatize (normalize) word before searching in dictionary
        4. returns word with placed stress '''

        prepared_word = word.lower()
        if prepared_word in self.exception_dict:
            stress_index = self.exception_dict[prepared_word][0]
            return word[:stress_index] + stress_symbol + word[stress_index:]
        
        prepared_word = word.lower().replace('ё', 'е')
        if prepared_word in self.exception_dict:
            stress_index = self.exception_dict[prepared_word][0]
            return word[:stress_index] + stress_symbol + word[stress_index:]

        prepared_word = self.morph_analyzer.parse(word)[0].normal_form
        if lemmatize_word and prepared_word in self.exception_dict:
            stress_index = self.exception_dict[prepared_word][0]
            return word[:stress_index] + stress_symbol + word[stress_index:]
        
        return word
