# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras

from . import metrics  # pylint: disable=relative-beyond-top-level
from . import embedding_layer


class EncoderStack(object):
  pass


class DecoderStack(object):
  pass


class Transformer(keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, params, name):
    super(Transformer, self).__init__(name=name)
    self.params = params

    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
      params["vocab_size"], params["hidden_size"])

    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)

  def call(self,inputs,training):
    pass

  def encode(self):
    pass

  def decode(self):
    pass

  def get_config(self):
    pass


def create_model(params, is_train):
  with tf.name_scope("model"):
    if is_train:  # pylint: disable=no-else-return
      inputs = keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = keras.layers.Input((None,), dtype="int64", name="targets")
      internal_model = Transformer(params, name="transformer")
      logits = internal_model([inputs, targets], training=is_train)
      vocab_size = params["vocab_size"]
      label_smoothing = params["label_smoothing"]
      # logits = metrics.MetricLayer(vocab_size)([logits, targets])
      logits = metrics.LossLayer(vocab_size, label_smoothing)([logits, targets])
      logits = keras.layers.Lambda(lambda x: x, name='logits')(logits)
      return keras.Model([inputs, targets], logits)

    else:
      inputs = keras.layers.Input((None,), dtype="int64", name="inputs")
      internal_model = Transformer(params, name="transformer")
      _return = internal_model([inputs], training=is_train)
      outputs, scores = _return['outputs'], _return['scores']
      return keras.Model(inputs, [outputs, scores])
