#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Unpacking the "Grammatical dictionary" by A. A. Zaliznyak from .zip archive, converting it to the format of exception dictionary
and combining it with the current exception dictionary.

Zaliznyak's dictionary is taken from http://odict.ru/.
'''

import os
import sys
import time
import zipfile
import curses
import argparse


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


def count_number_of_vowels(word: str) -> int:
    ''' Counting the number of vowels in a word. '''

    number_of_vowels = 0
    for symbol in word:
        if symbol in 'аеиоуэюяыёАЕИОУЭЮЯЫЁ':
            number_of_vowels += 1
    return number_of_vowels


def main():
    curses.setupterm()

    f_name_zip_odict = 'odict.zip'
    f_name_odict = 'zaliznyak.txt'
    f_name_current_exception_dict = 'stressrnn/source_exception_dictionary.txt'
    f_name_new_exception_dict = 'stressrnn/exception_dictionary.txt'

    parser = argparse.ArgumentParser(description="Converting Zaliznyak's dictionary from http://odict.ru/ to the format of " + \
                                                 "exception dictionary and combining it with the current exception dictionary.")
    parser.add_argument('-iz', '--f_name_zip_odict', type=str, default=None,
                        help="Name of .zip archive with the Zaliznyak's dictionary")
    parser.add_argument('-i', '--f_name_odict', type=str, default=None,
                        help="Name of .txt file with the Zaliznyak's dictionary")
    parser.add_argument('-ic', '--f_name_current_exception_dict', type=str, default=None,
                        help="Name of .txt file with the current exception dictionary (it will be combined with the Zaliznyak's dictionary)")
    parser.add_argument('-o', '--f_name_new_exception_dict', type=str, default=None,
                        help="Name of .txt file to save the combined dictionary")
    args = parser.parse_args()


    if args.f_name_zip_odict and args.f_name_odict:
        print("[W] 'f_name_zip_odict' and 'f_name_odict' are set simultaneously — the value from 'f_name_odict' will be used.")
        f_name_zip_odict = None
        f_name_odict = args.f_name_odict

    elif args.f_name_zip_odict and not args.f_name_odict:
        f_name_zip_odict = args.f_name_zip_odict

    elif not args.f_name_zip_odict and args.f_name_odict:
        f_name_zip_odict = None
        f_name_odict = args.f_name_odict

    if args.f_name_current_exception_dict:
        f_name_current_exception_dict = args.f_name_current_exception_dict
    if args.f_name_new_exception_dict:
        f_name_new_exception_dict = args.f_name_new_exception_dict


    # Unpacking archive with the dictionary to the same folder, where the archive is located
    start_time = time.time()
    if f_name_zip_odict:
        print("[i] Unpacking '{}'...".format(f_name_zip_odict))
        with zipfile.ZipFile(f_name_zip_odict, 'r') as zip_odict:
            zip_odict.extractall(os.path.dirname(f_name_zip_odict))
            f_name_odict = zip_odict.namelist()[0]


    print("[i] Loading Zaliznyak's dictionary from '{}'...".format(f_name_odict))
    zaliznyak_dict = []
    with open(f_name_odict, 'r') as f_odict:
        zaliznyak_dict = f_odict.readlines()
    zaliznyak_dict[0] = zaliznyak_dict[0].replace('\ufeff', '')
    print('[i] Loaded {} values'.format(len(zaliznyak_dict)))


    print("[i] Converting Zaliznyak's dictionary to the format of exception dictionary...")
    for i, word in enumerate(zaliznyak_dict):
        word = word.replace('\n', '').lower().split(' ')
        if not word[0] or count_number_of_vowels(word[0]) == 0:
            zaliznyak_dict[i] = ''
            continue

        word, stress_index = [subword for subword in word if subword][:2]
        if stress_index.find(',') != -1 or stress_index.find('.') != -1:
            zaliznyak_dict[i] = ''
            continue

        if word[0] == '-':
            word = word[1:]
        j = 0
        while j < len(word):
            if SAME_LETTERS_EN_RU.get(word[j]):
                word = word.replace(word[j], SAME_LETTERS_EN_RU[word[j]])
            j += 1

        zaliznyak_dict[i] = word[:int(stress_index)] + '+' + word[int(stress_index):] + '\n'
    zaliznyak_dict = [word for word in zaliznyak_dict if word]
    print('[i] After converting, there are {} values ​​left'.format(len(zaliznyak_dict)))


    print("[i] Loading current exception dictionary from '{}'...".format(f_name_current_exception_dict))
    current_exception_dict = []
    with open(f_name_current_exception_dict, 'r') as f_exception_dict:
        current_exception_dict = f_exception_dict.readlines()
    current_exception_dict[-1] += '\n'
    print('[i] Loaded {} values'.format(len(current_exception_dict)))


    print('[i] Combining dictionaries... 0 of {}'.format(len(current_exception_dict)))
    zaliznyak_dict_without_stresses = [word.replace('+', '') for word in zaliznyak_dict]

    for i, word in enumerate(current_exception_dict):
        if i % 1000 == 0 or i == len(current_exception_dict) - 1:
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('[i] Combining dictionaries... {} of {}'.format(i, len(current_exception_dict)))

        if word not in zaliznyak_dict and word.replace('+', '') in zaliznyak_dict_without_stresses:
            zaliznyak_dict[zaliznyak_dict_without_stresses.index(word.replace('+', ''))] = word

    zaliznyak_dict = current_exception_dict + zaliznyak_dict
    zaliznyak_dict = sorted(list(set(zaliznyak_dict)))


    print("[i] Saving {} values in '{}'...".format(len(zaliznyak_dict), f_name_new_exception_dict))
    with open(f_name_new_exception_dict, 'w') as f_new_exception_dict:
        f_new_exception_dict.writelines(zaliznyak_dict)

    
    if f_name_zip_odict:
        os.remove(f_name_odict)
    print('[i] Done in {:.2f} second(-s)'.format(time.time()-start_time))


if __name__ == '__main__':
    main()
