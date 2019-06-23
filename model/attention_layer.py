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

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]

    """
    with tf.name_scope('split_heads'):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      depth = (self.hidden_size // self.num_heads)  # the dim per head

      x = tf.reshape(x, (batch_size, length, depth))
      """The tensor is transposed to insure the inner dimensions hold the 
      correctvalues during the matrix multiplication.
      """
      x = tf.transpose(x, [0, 2, 1, 3])
      return x

  def combine_heads(self, x):
    pass

  # special call
  def call(self, x, bias, training, cache=None): # pylint: disable=unused-argument
    pass

  def get_config(self):
    pass


class SelfAttention(Attention):
  def call(self, x, bias, training, cache=None):
    return super(SelfAttention, self).call(x, x, bias, training, cache)
