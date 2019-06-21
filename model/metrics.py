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

    x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])  # rank of x = 3
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
  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)  # that is not padding

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


def padded_sequence_accuracy(logits, labels):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  with tf.name_scope("padded_sequence_accuracy"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    not_correct = tf.cast(tf.not_equal(outputs, padded_labels),
                          tf.float32) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)


def padded_accuracy_topk(logits, labels, k):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.name_scope("padded_accuracy_topk"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    effective_k = tf.minimum(k, tf.shape(logits)[-1])
    _, outputs = tf.nn.top_k(logits, k=effective_k)
    outputs = tf.cast(outputs, tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.cast(tf.equal(outputs, padded_labels), tf.float32)
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(logits, labels):
  return padded_accuracy_topk(logits, labels, 5)


class MetricLayer(keras.layers.Layer):
  """Custom a layer of metrics for Transformer model."""

  def __init__(self, vocab_size, **kwargs):
    super(MetricLayer, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.metrics_fn = [padded_cross_entropy_loss, padded_neg_log_perplexity]

  def build(self, input_shape):
    neg_log_perplexity = partial(padded_neg_log_perplexity,
                                 vocab_size=self.vocab_size)
    self.metric_mean_fns = [
      (tf.metrics.mean, padded_accuracy, "accuracy"),
      (tf.metrics.mean, padded_accuracy_top5, "accuracy_top5"),
      (tf.metrics.mean, padded_sequence_accuracy, "accuracy_per_sequence"),
      (tf.metrics.mean, neg_log_perplexity, "neg_log_perplexity"),
    ]
    super(MetricLayer, self).build(input_shape)

  def get_config(self):
    return {"vocab_size": self.vocab_size}

  def call(self, inputs, **kwargs):
    logits, targets = inputs[0], inputs[1]
    for mean, fn, name in self.metric_mean_fns:
      m = mean(*fn(logits, targets), name=name)
      # todo: how to add metrics
    return logits


def transformer_loss(logits, labels, smoothing, vocab_size):
  """Calculates total loss containing cross entropy with padding ignored.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary

  Returns:
    A scalar float tensor for loss.
  """
  xentropy, weights = padded_cross_entropy_loss(logits, labels, smoothing,
                                                vocab_size)

  return tf.reduce_sum(xentropy) / tf.reduce_sum(weights)


class LossLayer(keras.layers.Layer):
  def __init__(self, vocab_size, label_smoothing):
    super(LossLayer, self).__init__()
    self.vocab_size = vocab_size
    self.label_smoothing = label_smoothing

  def call(self, inputs, **kwargs):
    logits, targets = inputs[0], inputs[1]
    losses = transformer_loss(logits, targets, self.label_smoothing,
                              self.vocab_size)
    self.add_loss(losses)

  def get_config(self):
    return {'vocab_size': self.vocab_size,
            'label_smoothing': self.label_smoothing}
