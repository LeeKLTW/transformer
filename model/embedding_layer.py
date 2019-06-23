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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, **kwarg):
    """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super(EmbeddingSharedWeights, self).__init__(**kwarg)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    # in version 1 matmul is considered,
    # since it works better on TPU than gather(),
    # on CPU, GPU, vice versa

  def build(self, input_shape):
    with tf.name_scope("embedding_and_softmax"):
      self.shared_weights = \
        self.add_weight(name='weights',
                        shape=[self.vocab_size, self.hidden_size],
                        dtype='float32',
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stdev=self.hidden_size ** -0.5)
                        )
    super(EmbeddingSharedWeights, self).build(input_shape)

  def call(self, inputs, mode="embedding"):
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == 'linear':  # this is default in version 1
      return self._linear(inputs)
    else:
      raise ValueError(f"mode {str(mode)} is not valid")

  def _embedding(self, inputs):
    """Applies embedding based on inputs tensor."""
    with tf.name_scope("embedding"):
      mask = tf.cast(tf.math.not_equal(inputs, 0), tf.float32)
      embeddings = tf.gather(params=self.shared_weights, indices=inputs, axis=0)
      embeddings = embeddings * tf.expand_dims(mask, axis=-1)
      # scale
      embeddings = embeddings * self.hidden_size ** 0.5
      return embeddings

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.
    Args:
      inputs: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(inputs)[0]
      length = tf.shape(inputs)[1]

      x = tf.reshape(inputs, shape=(-1, self.hidden_size))
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)
      logits = tf.reshape(logits, [batch_size, length, self.vocab_size])
      return logits

  def get_config(self):
    return {
      "vocab_size": self.vocab_size,
      "hidden_size": self.hidden_size,
    }
