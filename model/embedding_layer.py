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
  def __init__(self, vocab_size, hidden_size, **kwarg):
    super(EmbeddingSharedWeights, self).__init__(**kwarg)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

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
    pass

  def get_config(self):
    return {
      "vocab_size": self.vocab_size,
      "hidden_size": self.hidden_size,
    }
