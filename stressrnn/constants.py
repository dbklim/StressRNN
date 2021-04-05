#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O., MashaPo

from pathlib import Path
import re


DEF_STRESS_SYMBOL = "'"
ADD_STRESS_SYMBOL = '+'
VOWELS = 'аеиоуэюяыёАЕИОУЭЮЯЫЁ'

SEARCH_TWO_VOWELS_RE = re.compile('[{}].*[{}]'.format(VOWELS, VOWELS))
MARKING_TEXT_RE = re.compile(r'[…\:,\.\?!\-\n]')
CLEANING_TEXT_RE = re.compile(r"[^а-яё'_\+\s\-]")

DEFAULT_TOKENIZER_CATEGORIES = [
    '0123456789',
    ' ',
    ',.;:!?()\"[]@#$%^&*_-=«»',
    'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ+\''
]

SAME_LETTERS_EN_RU = {
    'A': 'А',
    'B': 'В',
    'C': 'С',
    'E': 'Е',
    'H': 'Н',
    'K': 'К',
    'M': 'М',
    'O': 'О',
    'P': 'Р',
    'T': 'Т',
    'X': 'Х',
    'a': 'а',
    'c': 'с',
    'e': 'е',
    'o': 'о',
    'p': 'р',
    'x': 'х'
}

MAX_INPUT_LEN = 40
CHARS = [
    DEF_STRESS_SYMBOL, '-', '_', 'а', 'б', 'в', 'г', 'д', 'е', 'ж',
    'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
    'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё'
]
CHAR_INDICES = {symbol:i for i, symbol in enumerate(CHARS)}

BASE_DIR = Path(__file__).resolve().parent
F_NAME_TF_MODEL = str(BASE_DIR/'models/model.json')
F_NAME_TF_WEIGHTS = str(BASE_DIR/'models/weights_96.hdf5')
F_NAME_ONNX_MODEL = str(BASE_DIR/'models/model.onnx')
F_NAME_EXCEPTION_DICT = str(BASE_DIR/'dicts/exception_dictionary.txt')
