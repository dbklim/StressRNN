#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.6 or later
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Supporting functions for the StressRNN.
'''

from typing import List

try:
    from .constants import *
except ImportError:
    from constants import *


def tokenize(text: str, categories: list = DEFAULT_TOKENIZER_CATEGORIES, replace_similar_symbols: bool = False) -> List[str]:
    ''' Splitting text into words according to categories. It also replaces similar latin symbols with cyrillic ones. '''

    # Latin symbols to cyrillic
    if replace_similar_symbols:
        for i, symbol in enumerate(text):
            if SAME_LETTERS_EN_RU.get(symbol):
                text = text[:i] + SAME_LETTERS_EN_RU.get(symbol) + text[i+1:]

    token = ''
    tokens = []
    category = None
    for symbol in text:
        if token:
            if category and symbol in category:
                token += symbol
            else:
                tokens.append(token)
                token = symbol
                category = None
                for cat in categories:
                    if symbol in cat:
                        category = cat
                        break
        else:
            category = None
            if not category:
                for cat in categories:
                    if symbol in cat:
                        category = cat
                        break
            token += symbol
    if token:
        tokens.append(token)
    return tokens


def prepare_text(text: str, replace_similar_symbols: bool = False) -> List[str]:
    ''' Text preparation: replacing similar latin symbols with cyrillic ones, marking to support endings (context),
    removing unsupported symbols and splitting into words by spaces. '''

    # Latin symbols to cyrillic
    if replace_similar_symbols:
        for i, symbol in enumerate(text):
            if SAME_LETTERS_EN_RU.get(symbol):
                text = text[:i] + SAME_LETTERS_EN_RU.get(symbol) + text[i+1:]

    text = MARKING_TEXT_RE.sub(' _ ', text).lower()  # mark beginning of clause
    text = CLEANING_TEXT_RE.sub(' ', text)
    words = text.split(' ')
    return words


def add_endings(words: List[str]) -> List[str]:
    ''' Adding the ending of the previous word (context) to each word, if appropriate. Works only with a list of words. '''

    words_with_endings = []
    for i, word in enumerate(words):
        if not SEARCH_TWO_VOWELS_RE.search(word):
            # Won't predict, just return (less then two syllables)
            words_with_endings.append(word)
        elif i == 0 or words[i-1] == '_':
            words_with_endings.append('_'+word)
        else:
            context = words[i-1].replace(DEF_STRESS_SYMBOL, '').replace(ADD_STRESS_SYMBOL, '')
            if len(context) < 3:
                ending = context
            else:
                ending = context[-3:]
            words_with_endings.append(ending+'_'+word)
    return words_with_endings


def del_endings(words: List[str]) -> List[str]:
    ''' Deleting the ending of the previous word (context) in words. Works both with one word and with a list of words. '''

    if isinstance(words, str):
        return words[words.index('_')+1:] if words.find('_') != -1 else words
    elif isinstance(words, list):
        return [word[word.index('_')+1:] if word.find('_') != -1 else word for word in words]
    else:
        return words


def count_number_of_vowels(word: str) -> int:
    ''' Counting the number of vowels in a word. '''

    number_of_vowels = 0
    for symbol in word:
        if symbol in VOWELS:
            number_of_vowels += 1
    return number_of_vowels


def find_vowel_indices(word: str) -> List[int]:
    ''' Search for indices of all vowels in a word. Returns a list of indices. '''

    return [i for i, symbol in enumerate(word) if symbol in VOWELS]
