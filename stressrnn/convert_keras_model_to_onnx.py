#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Converting trained Keras model to ONNX Runtime format. Increases work speed of model by 10-30 times on the CPU.

Dependences: numpy>=1.16.0 tensorflow<=2.2.2 onnxruntime<=1.15.1 keras2onnx<=1.7
'''

import tensorflow as tf
import keras2onnx
import onnxruntime

try:
    from .constants import F_NAME_TF_MODEL, F_NAME_TF_WEIGHTS, F_NAME_ONNX_MODEL
except ImportError:
    from constants import F_NAME_TF_MODEL, F_NAME_TF_WEIGHTS, F_NAME_ONNX_MODEL


def convert_keras_model_to_onnx(f_name_keras_model: str, f_name_model_weights: str, f_name_onnx_model: str) -> None:
    ''' Convert trained Keras model to ONNX Runtime format. Increases work speed of model by 10-30 times on the CPU (on Intel i7-10510U, with
    TensorFlow v1.X test phrase processing time is about 150 ms, with ONNX Runtime — about 5 ms).

    Only TensorFlow <= v2.2.2 is supported (source: https://github.com/onnx/keras-onnx)!
    
    1. f_name_keras_model - name of .json file with the keras model
    2. f_name_model_weights - name of .hdf5 file with keras model weights
    3. f_name_onnx_model - name of .onnx file to save onnx model
    4. returns None '''

    print("[i] Loading keras model and its weights from '{}' and '{}'".format(f_name_keras_model, f_name_model_weights))
    with open(f_name_keras_model, 'r') as f_model:
        model_json = f_model.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(str(f_name_model_weights))

    print('[i] Converting keras model to onnx...')
    onnx_model = keras2onnx.convert_keras(model, model.name)

    print("[i] Saving onnx model to '{}'".format(f_name_onnx_model))
    keras2onnx.save_model(onnx_model, f_name_onnx_model)


# TensorFlow v2.4.1: "AttributeError: 'KerasTensor' object has no attribute 'graph'"


def main():
    f_name_keras_model = F_NAME_TF_MODEL
    f_name_model_weights = F_NAME_TF_WEIGHTS
    f_name_onnx_model = F_NAME_ONNX_MODEL

    convert_keras_model_to_onnx(f_name_keras_model, f_name_model_weights, f_name_onnx_model)


if __name__ == '__main__':
    main()
