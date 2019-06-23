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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class FeedForwardNetwork(keras.layers.Layer):
  def __init__(self, hidden_size, filter_size, relu_dropout, **kwargs):
    super(FeedForwardNetwork, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

  def build(self, input_shape):
    self.filter_dense_layer = \
      keras.layers.Dense(units=self.filter_size, use_bias=True,
                         activation=tf.nn.relu, name='filter_layer')
    self.output_dense_layer = \
      keras.layers.Dense(units=self.hidden_size, use_bias=True,
                         activation=tf.nn.relu, name='output_layer')
    super(FeedForwardNetwork, self).build(input_shape)

  def call(self, x, training):
    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, keep_prob=1.0 - self.relu_dropout)
    output = self.output(output)
    return output

  def get_config(self):
    return {"hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout}
