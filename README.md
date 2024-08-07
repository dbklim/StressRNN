# StressRNN

This tool based on LSTM predicts stress position in each word in russian text depending on the word context.
For more details about the tool see [«Automated Word Stress Detection in Russian»](http://www.aclweb.org/anthology/W/W17/W17-4104.pdf), EMNLP-2017, Copenhagen, Denmark.

This is a modified version of the [RusStress](https://github.com/MashaPo/russtress) (many thanks to its authors!). The modification fixes some bugs, improves the quality of stress placement and the usability of the project. The neural network and the principles of working with it remained unchanged.

## Modification features

The modification has the following differences from the original RusStress:

- added stress symbol `'+'`
- the list of supported stress symbols is limited to `"'"` and `'+'` (always added after a stressed vowel)
- added filtering of neural network responses by accuracy (allows you to cut off stresses, the prediction accuracy of which is lower (<=) than specified)
- added stress placing in words with 1 vowel
- improved work with hyphenated words (earlier, stress was placed only on the first part of the word, now it's placed on both parts)
- implemented skipping words with pre-set stresses (it used to lead to undefined behavior)
- added replacement of similar Latin symbols with the same Cyrillic ones (for example, `A` -> `А`, `e` -> `е`) (the full list is available in [SAME_LETTERS_EN_RU](https://github.com/Desklop/StressRNN/blob/master/stressrnn/constants.py#L27))
- fixed bug with some words, in which the stress was placed on a consonant letter, not a vowel (stress index is shift to next vowel or to the previous one, if there are no vowels after stress) (for example, "дойдете", "зряченюхослышащий") (these words were added to the [exception dictionary](https://github.com/Desklop/StressRNN/blob/master/stressrnn/source_exception_dictionary.txt))
- added an exception dictionary and the ability to load your own additional exception dictionary when creating a class object
- added a [default exception dictionary](https://github.com/Desklop/StressRNN/blob/master/stressrnn/exception_dictionary.txt) based on the "Grammatical dictionary" by A. A. Zaliznyak from http://odict.ru/ (supplied with the package)
- compiled all regular expressions to improve performance
- code refactoring was performed
- added support for TensorFlow v2.X
- optimization of work speed: added work with neural network in batch mode (speeds up work by 1.5-2 times)
- optimization of work speed: added predict using ONNX Runtime (increases work speed of model by 10-30 times on the CPU)

These changes allow you to use this package in projects such as speech synthesis, without any changes, "out of the box".

## Installation

Simple installation with pip (python 3.6-3.10 supported):

```bash
pip3 install git+https://github.com/Desklop/StressRNN
```

Or installing from sources:

```bash
git clone https://github.com/Desklop/StressRNN && cd StressRNN
virtualenv env_base && source env_base/bin/activate
pip install -r requirements.txt
python3 setup.py bdist_wheel
```

Then copy .whl package from `/dist` to your project and run:

```bash
pip install stressrnn-0.2.*.whl
```

For development, you need to additionally install TensorFlow:

```bash
git clone https://github.com/Desklop/StressRNN && cd StressRNN
virtualenv env_base && source env_base/bin/activate
pip install -r requirements_dev.txt
```

**Full dependencies:** `numpy>=1.16.0 pymorphy2[fast]<=0.9.2 tensorflow<=2.2.2 onnxruntime<=1.15.1 keras2onnx<=1.7` (listed in [requirements_dev.txt](https://github.com/Desklop/StressRNN/blob/master/requirements_dev.txt), only Python 3.6-3.8).

**Dependencies for package use only:** `numpy>=1.16.0 pymorphy2[fast]<=0.9.2 onnxruntime<=1.15.1` (listed in [requirements.txt](https://github.com/Desklop/StressRNN/blob/master/requirements.txt)).

**Note 1:** TensorFlow v2.X is supported, but the work speed using TensorFlow v2.X is about 3-5 times lower than with TensorFlow v1.X.

**Note 2:** ONNX Runtime is used by default. This increases the work speed of package by 10-30 times on the CPU (on Intel i7-10510U, with TensorFlow v1.X test phrase processing time is about 150 ms, with ONNX Runtime — about 1-3 ms).

## Usage

Example of stress placement in your text:

```python
from stressrnn import StressRNN

stress_rnn = StressRNN()

text = 'Проставь, пожалуйста, ударения'
stressed_text = stress_rnn.put_stress(text, stress_symbol='+', accuracy_threshold=0.75, replace_similar_symbols=True)
print(stressed_text)  # 'Проста+вь, пожа+луйста, ударе+ния'
```

The package contains 2 classes: [StressRNN](https://github.com/Desklop/StressRNN/blob/master/stressrnn/stressrnn.py#L40) and [ExceptionDictWrapper](https://github.com/Desklop/StressRNN/blob/master/stressrnn/exception_dictionary_wrapper.py#L24).

---

### [StressRNN](https://github.com/Desklop/StressRNN/blob/master/stressrnn/stressrnn.py#L40) class

The placement of stress in text using BiLSTM.

Additionally, the exception dictionary is supported, which must contain a list of words with stresses placed in them using the `"'"` or `'+'` symbol after the vowel (1 line = 1 word). The exception dictionary is used before the neural network (if a match was found, the neural network is not used).

The dictionary looks like this:

```text
    дре+гер
    ивано+в
    ...
```

The main dictionary comes with the package and is located in [`stressrnn/exception_dictionary.txt`](https://github.com/Desklop/StressRNN/blob/master/stressrnn/exception_dictionary.txt). You can also add your own dictionary, the values from which will complement the main dictionary (and overwrite the same words, but with different stresses).

Accepts arguments:

1. `f_name_add_exception_dict` - name of additional .txt dictionary with exceptions

**Contains methods:**

```python
put_stress(self, text: str, stress_symbol:str = '+', accuracy_threshold: float = 0.75, replace_similar_symbols: bool = False, lemmatize_words: bool = False) -> str:
```

Split the text into words and place stress on them. The source text formatting is preserved. If some words already have an stress, it will be saved.

The stress is indicated using the `"'"` or `'+'` symbol after the stressed vowel.

The threshold for the accuracy of stress placement allows you to cut off stresses, the prediction accuracy of which is lower (<=) than specified. The 0.75 threshold reduces the number of incorrectly placed stresses, but increases the number of words that will not be stressed. The 0.0 threshold allows you to place stresses in absolutely all words, but not always correctly.

1. `text` - string with text
2. `stress_symbol` - stress symbol, only `"'"` and `'+'` are supported
3. `accuracy_threshold` - threshold for the accuracy of stress placement (from `0.0` to `1.0`)
4. `replace_similar_symbols` - `True`: replacing similar latin symbols with cyrillic ones
5. `lemmatize_words` - `True`: lemmatize (normalize) each word before searching in exception dictionary
6. returns text with placed stresses

---

### [ExceptionDictWrapper](https://github.com/Desklop/StressRNN/blob/master/stressrnn/exception_dictionary_wrapper.py#L24) class

Exception dictionary for correcting stress placement.

The exception dictionary must contain a list of words with stresses placed in them using the `"'"` or `'+'` symbol after the vowel (1 line = 1 word).

The dictionary looks like this:

```text
    дре+гер
    ивано+в
    ...
```

The main dictionary comes with the package and is located in [`stressrnn/exception_dictionary.txt`](https://github.com/Desklop/StressRNN/blob/master/stressrnn/exception_dictionary.txt). You can also add your own dictionary, the values from which will complement the main dictionary (and overwrite the same words, but with different stresses).

Accepts arguments:

1. `f_name_add_exception_dict` - name of additional .txt dictionary with exceptions

**Contains methods:**

```python
is_in_dict(self, word: str, lemmatize_word: bool = False) -> bool:
```

Checking if the word is in the dictionary.

1. `word` - string with the word of interest
2. `lemmatize_word` - `True`: lemmatize (normalize) word before searching in dictionary
3. returns `True`/`False`

```python
put_stress(self, word: str, stress_symbol: str, lemmatize_word: bool = False) -> str:
```

Put stress in a word in accordance with the dictionary. Stress is indicated by stress_symbol after the stressed vowel.

1. `word` - string with the word of interest
2. `stress_symbol` - stress symbol
3. `lemmatize_word` - `True`: lemmatize (normalize) word before searching in dictionary
4. returns word with placed stress

## Updating the default exception dictionary

The default exception dictionary is based on the "Grammatical Dictionary" by A. A. Zaliznyak. To update it, you need to download the latest version from http://odict.ru/ (you need a dictionary without word forms) and run:

```python
python3 convert_zaliznyak_to_exception_dictionary.py -iz odict.zip -o stressrnn/exception_dictionary.txt
```

## Datasets

The repo [contains](https://github.com/Desklop/StressRNN/tree/master/datasets) samples from UD treebanks annotated with word stress for the Russian, Ukranian and Belorusian languages. For more details about the tool see VarDial paper (coming soon).

## Authors

- [Maria Ponomareva](https://github.com/MashaPo) ([RusStress](https://github.com/MashaPo/russtress), ponomarevamawa@gmail.com)
- [Kirill Milintsevich](https://github.com/501Good) ([RusStress](https://github.com/MashaPo/russtress))
- [Vladislav Klim](https://github.com/Desklop) ([StressRNN](https://github.com/Desklop/StressRNN), vladsklim@gmail.com, [LinkedIn](https://www.linkedin.com/in/vladklim/))
