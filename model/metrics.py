# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
from tensorflow import keras


def _pad_tensors_to_same_length(x, y):
  with tf.name_scope('pad_to_same_length'):
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]
    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
  return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding."""
  with tf.name_scope("loss"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)

    with tf.name_scope("smoothing_cross_entropy"):  # smoothing=epsilon
      confidence = 1.0 - smoothing
      # uniform distribution
      low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)

      soft_targets = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                depth=int(vocab_size), on_value=confidence,
                                off_value=low_confidence)

      # Note: labels=soft_targets
      xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=soft_targets)
  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

  return xentropy * weights, weights


def padded_neg_log_perplexity(logits, labels, vocab_size):
  num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
  return -num, den


def padded_accuracy(logits, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  with tf.name_scope("padded_accuracy"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    return tf.cast(tf.equal(outputs, padded_labels), tf.float32), weights


def padded_sequence_accuracy():
  pass


def padded_accuracy_top5():
  pass


class MetricLayer(keras.layers.Layer):
  """Custom a layer of metrics for Transformer model."""

  def __init__(self, vocab_size):
    super(MetricLayer, self).__init__()
    self.vocab_size = vocab_size
    self.metrics_fn = []

  def build(self, input_shape):
    neg_log_perplexity = partial(padded_neg_log_perplexity,
                                 vocab_size=self.vocab_size)
    self.metrics_fn = []