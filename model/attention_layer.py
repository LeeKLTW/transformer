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
        "Hidden size ({}) must be divisible by the number of heads ({}).".format(
          hidden_size, num_heads))

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
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope('combine_heads'):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])
      x = tf.reshape(x, [batch_size, length, self.hidden_size])
      return x

  def call(self, x, y, bias, training, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """

    # Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.q_dense_layer(y)
    v = self.q_dense_layer(y)

    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    depth = (self.hidden_size // self.num_heads)
    scale = depth ** -0.5
    q *= scale
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias

    weights = tf.nn.softmax(logits)

    if training:
      weights = tf.nn.dropout(weights, keep_prob=1.0 - self.attention_dropout)

    attention_output = tf.matmul(weights, v, transpose_b=True)

    attention_output = self.output_dense_layer(attention_output)
    return attention_output

  def get_config(self):
    pass


class SelfAttention(Attention):
  def call(self, x, bias, training, cache=None):
    return super(SelfAttention, self).call(x, x, bias, training, cache)
