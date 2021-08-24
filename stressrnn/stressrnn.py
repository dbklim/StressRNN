#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O., MashaPo

'''
The placement of stress in text using a recurrent neural network BiLSTM. It's a modified
version of RusStress (https://github.com/MashaPo/russtress). Only Russian is supported.

Contains the 'StressRNN' class. Learn more in https://github.com/Desklop/StressRNN.

Dependences:
    - for ONNX Runtime: numpy>=1.16.0 pymorphy2[fast]<=0.9.2 onnxruntime<=1.6
    - for TensorFlow: numpy>=1.16.0 pymorphy2[fast]<=0.9.2 tensorflow<=2.3.1
'''

import os
import time
from typing import List, Tuple
import numpy as np

try:
    from .exception_dictionary_wrapper import ExceptionDictWrapper
    from .constants import DEF_STRESS_SYMBOL, ADD_STRESS_SYMBOL, VOWELS, SEARCH_TWO_VOWELS_RE, MAX_INPUT_LEN, CHAR_INDICES
    from .constants import F_NAME_TF_MODEL, F_NAME_TF_WEIGHTS, F_NAME_ONNX_MODEL
    from .utils import tokenize, prepare_text, add_endings, del_endings, count_number_of_vowels, find_vowel_indices
except ImportError:
    from exception_dictionary_wrapper import ExceptionDictWrapper
    from constants import DEF_STRESS_SYMBOL, ADD_STRESS_SYMBOL, VOWELS, SEARCH_TWO_VOWELS_RE, MAX_INPUT_LEN, CHAR_INDICES
    from constants import F_NAME_TF_MODEL, F_NAME_TF_WEIGHTS, F_NAME_ONNX_MODEL
    from utils import tokenize, prepare_text, add_endings, del_endings, count_number_of_vowels, find_vowel_indices

if os.path.exists(F_NAME_ONNX_MODEL):
    import onnxruntime
else:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Activation

    # Disabling TensorFlow warning notifications (https://github.com/tensorflow/tensorflow/issues/27023#issuecomment-589673539)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


__version__ = '0.2.0'


class StressRNN(object):
    ''' The placement of stress in text using BiLSTM. Contains methods:
    - put_stress(): splitting text into words and placing stresses in them

    Additionally, the exception dictionary is supported, which must contain a list of words with stresses placed in them
    using the "'" or '+' symbol after the vowel (1 line = 1 word). The exception dictionary is used before the neural
    network (if a match was found, the neural network is not used).

    The dictionary looks like this:
        дре+гер
        ивано+в
        ...

    The main dictionary comes with the package and is located in 'stressrnn/exception_dictionary.txt'. You can also add your own
    dictionary, the values from which will complement the main dictionary (and overwrite the same words, but with different stresses).

    1. f_name_add_exception_dict - name of additional .txt dictionary with exceptions '''

    def __init__(self, f_name_add_exception_dict: str = None) -> None:
        self.__load_model()
        self.exception_dict_wrapper = ExceptionDictWrapper(f_name_add_exception_dict)


    def __load_model(self) -> None:
        ''' Load model using ONNX Runtime or TensorFlow (v1.X or v2.X) depending on the existence of 'F_NAME_ONNX_MODEL' file.
        The loaded model is saved to internal variable 'self.model'. '''

        # Load model with ONNX Runtime
        if os.path.exists(F_NAME_ONNX_MODEL):
            # Configuring ONNX Runtime session to work in 1 process
            session_options = onnxruntime.SessionOptions()
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1

            self.model = onnxruntime.InferenceSession(F_NAME_ONNX_MODEL, sess_options=session_options)

        # Load model with TensorFlow v1.X
        elif tf.__version__[0] == '1':
            with open(F_NAME_TF_MODEL, 'r') as f_model:
                model_json = f_model.read()

            tf.keras.backend.clear_session()
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.session = tf.Session(graph=self.graph)
                with self.session.as_default():
                    self.model = tf.keras.models.model_from_json(model_json)
                    self.model.load_weights(str(F_NAME_TF_WEIGHTS))

        # Load model with TensorFlow v2.X
        else:
            with open(F_NAME_TF_MODEL, 'r') as f_model:
                model_json = f_model.read()

            self.model = tf.keras.models.model_from_json(model_json)
            self.model.load_weights(str(F_NAME_TF_WEIGHTS))


    def __predict_wrapper(self, inputs: np.array) -> np.array:
        ''' Predict the stress position in list of prepared words.
        
        1. inputs - np.array with float32 representing words with context
        2. returns np.array with the stress position in each word (the position is reflected as a probability for each letter of the word) '''

        # Predict with ONNX Runtime
        if str(type(self.model)).find('onnxruntime') != -1:
            inputs = {model_input.name: inputs for model_input in self.model.get_inputs()}
            predictions = self.model.run(None, inputs)[0]

        # Predict with TensorFlow v1.X
        elif tf.__version__[0] == '1':
            with self.graph.as_default():
                with self.session.as_default():
                    predictions = self.model.predict(inputs, verbose=0)
        
        # Predict with TensorFlow v2.X
        else:
            predictions = self.model.predict(inputs, verbose=0)
        
        return predictions


    def __predict(self, words_with_ending: List[str], stress_symbol: str) -> Tuple[str, float]:
        ''' Predict the stress position in list of words or one word. Stress is indicated by stress_symbol after the stressed vowel.

        1. word_with_ending - list of words (or one word) with context (last 1-3 letters of the previous word), if there is one
        2. stress_symbol - stress symbol
        3. returns the list of tuples with source words without context with the stresses and prediction accuracies '''

        if not words_with_ending:
            return words_with_ending
        
        if isinstance(words_with_ending, str):
            words_with_ending = [words_with_ending]

        inputs = np.zeros((len(words_with_ending), MAX_INPUT_LEN, len(CHAR_INDICES)), dtype=np.float32)
        for i, word_with_ending in enumerate(words_with_ending):
            for j, symbol in enumerate(word_with_ending):
                pos = MAX_INPUT_LEN - len(word_with_ending) + j
                inputs[i, pos, CHAR_INDICES[symbol]] = 1

        predictions = self.__predict_wrapper(inputs)
 
        predictions = predictions.tolist()
        accuracities = [max(prediction) for prediction in predictions]
        stress_indexes = [prediction.index(accuracity) for prediction, accuracity in zip(predictions, accuracities)]

        words = [del_endings(word_with_ending) for word_with_ending in words_with_ending]
        stress_indexes = [len(word) - MAX_INPUT_LEN + stress_index for word, stress_index in zip(words, stress_indexes)]

        for i, (word, stress_index) in enumerate(zip(words, stress_indexes)):
            if stress_index > len(word) - 1:
                print("\n[W] No {}-th letter in '{}'!\n".format(stress_index+1, word))
                accuracities[i] = 0.0

        # Processing cases when the stress is placed on a consonant letter (stress index is shift to next vowel or to the previous one,
        # if there are no vowels after stress) (for example, 'дойдете', 'зряченюхослышащий') (these words were added to the exception dictionary)
        stressed_words = []
        for word, stress_index, accuracity in zip(words, stress_indexes, accuracities):
            if word[stress_index] not in VOWELS:
                vowel_indices = find_vowel_indices(word)
                for vowel_index in vowel_indices:
                    if stress_index < vowel_index:
                        stress_index = vowel_index
                        break
                if word[stress_index] not in VOWELS:
                    for vowel_index in reversed(vowel_indices):
                        if stress_index > vowel_index:
                            stress_index = vowel_index
                            break

            stressed_words.append([word[:stress_index+1]+stress_symbol+word[stress_index+1:], accuracity])

        return stressed_words


    def put_stress(self, text: str, stress_symbol:str = '+', accuracy_threshold: float = 0.75, replace_similar_symbols: bool = False,
                   lemmatize_words: bool = False, use_batch_mode: bool = True) -> str:
        ''' Split the text into words and place stress on them. The source text formatting is preserved. If some words already have an stress,
        it will be saved.

        The stress is indicated using the "'" or '+' symbol after the stressed vowel.

        The threshold for the accuracy of stress placement allows you to cut off stresses, the prediction accuracy of which is lower (<=)
        than specified. The 0.75 threshold reduces the number of incorrectly placed stresses, but increases the number of words that will
        not be stressed. The 0.0 threshold allows you to place stresses in absolutely all words, but not always correctly.
        
        1. text - string with text
        2. stress_symbol - stress symbol, only "'" and '+' are supported
        3. accuracy_threshold - threshold for the accuracy of stress placement (from 0.0 to 1.0)
        4. replace_similar_symbols - True: replacing similar latin symbols with cyrillic ones
        5. lemmatize_words - True: lemmatize (normalize) each word before searching in exception dictionary
        6. use_batch_mode - True: place stress on words for 1 call to the neural network (speeds up work by 1.5-2 times)
        7. returns text with placed stresses '''

        if stress_symbol != DEF_STRESS_SYMBOL and stress_symbol != ADD_STRESS_SYMBOL:
            raise ValueError("Unsupported stress symbol '{}'! Only \"{}\" and '{}' are supported.".format(
                                stress_symbol, DEF_STRESS_SYMBOL, ADD_STRESS_SYMBOL))

        words = prepare_text(text, replace_similar_symbols=replace_similar_symbols)
        tokens = tokenize(text, replace_similar_symbols=replace_similar_symbols)
        words_with_endings = add_endings(words)

        # Stress placement
        stressed_words = []
        batch_for_predict = []
        for word in words_with_endings:
            # When using the module after russian_g2p.Accentor, it is possible situation that one of the words passed to the input, contains
            # stress symbol after each letter (for example, 'почем' -> '+п+о+ч+е+м+')
            if word.count(DEF_STRESS_SYMBOL) > 2:
                word = word.replace(DEF_STRESS_SYMBOL, '')
            elif word.count(ADD_STRESS_SYMBOL) > 1:
                word = word.replace(ADD_STRESS_SYMBOL, '')

            if word.find(DEF_STRESS_SYMBOL) != -1 and word[word.find(DEF_STRESS_SYMBOL)-1] in VOWELS:
                continue
            
            elif word.find(ADD_STRESS_SYMBOL) != -1 and word[word.find(ADD_STRESS_SYMBOL)-1] in VOWELS:
                continue

            elif count_number_of_vowels(word) == 1:
                stressed_word = word[:find_vowel_indices(word)[-1]+1] + stress_symbol + word[find_vowel_indices(word)[-1]+1:]
                stressed_words.append(stressed_word)

            elif SEARCH_TWO_VOWELS_RE.search(word) and self.exception_dict_wrapper.is_in_dict(del_endings(word), lemmatize_words):
                stressed_word = self.exception_dict_wrapper.put_stress(del_endings(word), stress_symbol, lemmatize_words)
                stressed_words.append(stressed_word)

            elif use_batch_mode and SEARCH_TWO_VOWELS_RE.search(word):
                batch_for_predict.append(word)
                stressed_words.append(word)

            elif not use_batch_mode and SEARCH_TWO_VOWELS_RE.search(word):
                stressed_word, accuracity = self.__predict(word, stress_symbol)[0]
                if accuracity >= accuracy_threshold:
                    stressed_words.append(stressed_word)

        # Predict all words in 1 network call
        if use_batch_mode:
            batch_with_stressed_words = self.__predict(batch_for_predict, stress_symbol)

            if len(batch_for_predict) > 0:
                updated_stressed_words = []
                idx_in_batch = 0
                for stressed_word in stressed_words:
                    if idx_in_batch < len(batch_for_predict) and stressed_word == batch_for_predict[idx_in_batch] \
                                                             and batch_with_stressed_words[idx_in_batch][1] >= accuracy_threshold:
                        updated_stressed_words.append(batch_with_stressed_words[idx_in_batch][0])
                        idx_in_batch += 1
                    elif idx_in_batch < len(batch_for_predict) and stressed_word == batch_for_predict[idx_in_batch]:
                        idx_in_batch += 1
                    else:
                        updated_stressed_words.append(stressed_word)
                stressed_words = updated_stressed_words

        # Transferring stresses to the source text
        stressed_text = []
        for token in tokens:
            if count_number_of_vowels(token) == 0:
                stressed_text.append(token)
            else:
                try:
                    unstressed_word = stressed_words[0].replace(stress_symbol, '')
                except IndexError:
                    unstressed_word = ''
                if unstressed_word == token.lower():
                    stress_position = stressed_words[0].find(stress_symbol)
                    stressed_token = token[:stress_position] + stress_symbol + token[stress_position:]
                    stressed_text.append(stressed_token)
                    stressed_words = stressed_words[1:]
                else:
                    stressed_text.append(token)
        stressed_text = ''.join(stressed_text)

        return stressed_text


# Добавлено/изменено:
# - добавлен символ ударения '+'
# - ограничен список поддерживаемых символов ударений до "'" и '+' (добавляются всегда после ударной гласной)
# - добавлена фильтрация ответов нейронной сети по точности (позволяет отсечь ударения, точность предсказания которых меньше (<=) указанной)
# - добавлена постановка ударения в словах с 1 гласной
# - улучшена работа со словами, записанными через дефис (раньше ударение ставилось только на первую часть слова, теперь ставится на обе части)
# - реализован пропуск слов с заранее поставленными ударениями (раньше это приводило к неопределённому поведению)
# - исправлено поведение с некоторыми словами, при котором ударение ставилось не после, а перед гласной (например, слово "зряченюхослышащий")
# - добавлена замена похожих латинских символов на такие же кириллические (например, `A` -> `А`, `e` -> `е`) (полный список доступен в словаре замен)
# - добавлен (дописан) словарь исключений и загрузка своего словаря исключений при создании объекта класса
# - скомпилированы все регулярные выражения для повышения производительности
# - выполнен рефакторинг кода
# - добавлена поддержка TensorFlow v2.X
# - оптимизация скорости работы: добавлена работа с нейронной сетью в batch режиме (ускоряет работу в 1.5-2 раза)
# - оптимизация скорости работы: добавлен predict в ONNX Runtime (ускоряет работу в 10-20 раз на CPU)


def main():
    test_text = 'ПривЕТ, зряченюхослышащий Иванов-Лапнюкoв Григо\'рий Фе+ликсович! Как ВAши Дела-Делишки тут, мм? Дойдете/зайдешь сами/сам? ' + \
                'Купили каких-нибудь экзотермов вчера или решили набрать нифуртов у Ельпетифоровых? Один рубль или два рубля?'

    stress_rnn = StressRNN(f_name_add_exception_dict='add_exception_dictionary.txt')


    start_time = time.time()
    stressed_text = stress_rnn.put_stress(test_text, stress_symbol='+', accuracy_threshold=0.75, replace_similar_symbols=True)
    elapsed_time = time.time() - start_time
    print("\n[i] Source text:   '{}',\n    Stressed text: '{}',\n    Elapsed time {:.6f} s".format(test_text, stressed_text, elapsed_time))

    start_time = time.time()
    stressed_text = stress_rnn.put_stress(test_text, stress_symbol='+', accuracy_threshold=0.0, replace_similar_symbols=True)
    elapsed_time = time.time() - start_time
    print("\n[i] Source text:   '{}',\n    Stressed text: '{}',\n    Elapsed time {:.6f} s".format(test_text, stressed_text, elapsed_time))


    print("\n[i] Interactive mode (enter 'e', 'exit', 'q', 'quit' to exit)")
    while True:
        text = input('\n[i] Enter text: ')
        if text in ['e', 'exit', 'q', 'quit']:
            break

        start_time = time.time()
        stressed_text = stress_rnn.put_stress(text, stress_symbol='+', accuracy_threshold=0.0)
        elapsed_time = time.time() - start_time
        print("[i] Source text:   '{}',\n    Stressed text: '{}',\n    Elapsed time {:.6f} s".format(text, stressed_text, elapsed_time))


if __name__ == '__main__':
    main()
