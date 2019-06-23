# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class Attention(keras.layers.Layer):
  def __init__(self, hidden_size, num_heads, attention_dropout, **kwargs):
    if hidden_size % num_heads != 0:
      raise ValueError(
        "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

  def build(self, input_shape):
    self.q_dense_layer = \
      keras.layers.Dense(units=self.hidden_size, use_bias=False, name='W_Q')

    self.k_dense_layer = \
      keras.layers.Dense(units=self.hidden_size, use_bias=False, name='W_K')

    self.v_dense_layer = \
      keras.layers.Dense(units=self.hidden_size, use_bias=False, name='W_V')

    self.output_dense_layer = \
      keras.layers.Dense(units=self.hidden_size, use_bias=False, name='W_O')

    super(Attention, self).build(input_shape)


  # special call
  def call(self, x, bias, training, cache=None): # pylint: disable=unused-argument
    pass

  def get_config(self):
    pass


class SelfAttention(Attention):
  def call(self, x, bias, training, cache=None):
    return super(SelfAttention, self).call(x, x, bias, training, cache)
