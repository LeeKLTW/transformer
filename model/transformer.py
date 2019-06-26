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
from . import attention_layer
from . import ffn_layer
from utils import preprocessing  # sys.path.extend or export in command-line


class PrePostProcessingWrapper(keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, **kwargs):
    super(PrePostProcessingWrapper, self).__init__(**kwargs)
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params.get("layer_postprocess_dropout")

  def build(self, input_shape):
    self.layer_norm = LayerNormalization(self.params.get("hidden_size"))
    super(PrePostProcessingWrapper, self).build(input_shape)

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)
    y = self.layer(y, *args, **kwargs)

    # ADD Norm
    training = kwargs.get("training")
    if training:
      y = tf.nn.dropout(y, keep_prob=1.0 - self.postprocess_dropout)
    y = x + y  # residual
    return y

  def get_config(self):
    return {"params": self.params}  # self.postprocess_dropout is included


class LayerNormalization(keras.layers.Layer):
  """Applies layer normalization.
  J.Ba, J.Kiros, G.Hinton, Layer Normalization, 2016 Equation (3)
  """

  def __init__(self, hidden_size, **kwargs):
    super(LayerNormalization, self).__init__(**kwargs)
    self.hidden_size = hidden_size

  def build(self, input_shape):
    self.scale = \
      self.add_weight(name='layer_norm_scale', shape=input_shape[-1],
                      dtype='float32', initializer=tf.initializers.ones())

    self.bias = \
      self.add_weight(name='layer_norm_bias', shape=input_shape[-1],
                      dtype='float32', initializer=tf.initializers.zeros())
    super(LayerNormalization, self).build(input_shape)

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(mean - x), axis=[-1], keepdims=True)
    variance = tf.sqrt(variance)

    norm_x = x - mean
    norm_x = tf.divide(norm_x, variance)

    norm_x = norm_x * self.scale + self.bias

    return norm_x

  def get_config(self):
    return {"hidden_size": self.hidden_size}


class EncoderStack(keras.layers.Layer):
  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
        params["hidden_size"], params["num_heads"], params["attention_dropout"])

      feed_forward_network = ffn_layer.FeedForwardNetwork(
        params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append(
        [PrePostProcessingWrapper(self_attention_layer, params),
         PrePostProcessingWrapper(feed_forward_network, params)]
      )

    self.output_normalization = LayerNormalization(params["hidden_size"])
    super(EncoderStack, self).build(input_shape)

  def call(self, encoder_inputs, attention_bias, training):

    for idx, layers in enumerate(self.layers):
      self_attention_layer = layers[0]
      feed_forward_layer = layers[1]

      with tf.name_scope('encoder_layer{}'.format(idx)):
        with tf.name_scope('self_attention'):
          y = self_attention_layer(encoder_inputs,
                                   bias=attention_bias, training=training)
        with tf.name_scope('ffn'):
          y = feed_forward_layer(y, training=training)

    y = self.output_normalization(y)
    return y

  def get_config(self):
    return {
      "params": self.params,
    }


class DecoderStack(keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, params):
    super(DecoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
        params["hidden_size"], params["num_heads"], params["attention_dropout"])
      encdec_attention_layer = attention_layer.Attention(
        params["hidden_size"], params["num_heads"], params["attention_dropout"])
      feed_forward_layer = ffn_layer.FeedForwardNetwork(params["hidden_size"],
                                                        params["filter_size"],
                                                        params["relu_dropout"])

      self.layers.append([
        PrePostProcessingWrapper(self_attention_layer,params),
        PrePostProcessingWrapper(encdec_attention_layer,params),
        PrePostProcessingWrapper(feed_forward_layer,params),
      ])
    self.output_normalization = LayerNormalization(params['hidden_size'])

    super(DecoderStack, self).build(input_shape)

  def call(self, decoder_inputs, encoder_output, decoder_self_attention_bias,
           training):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
      encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: bias for decoder self-attention layer. [1, 1,
        target_len, target_length]
      attention_bias: bias for encoder-decoder attention layer. [batch_size, 1,
        1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for idx, layers in enumerate(self.layers):
      self_attention_layer = layers[0]
      encdec_attention_layer = layers[1]
      feed_forward_layer = layers[2]

      with tf.name_scope('decoder_layer{}'.format(idx)):
        with tf.name_scope('self_attention'):
          y = self_attention_layer(decoder_inputs,
                                   bias=decoder_self_attention_bias,
                                   training=training)
        with tf.name_scope('encdec_attention'):
          y = encdec_attention_layer(y, encoder_output,
                                     bias=decoder_self_attention_bias,
                                     training=training)

        with tf.name_scope('feed_forward_layer'):
          y = feed_forward_layer(y, training=training)

    y = self.output_normalization(y)
    return y

  def get_config(self):
    return {
      "params": self.params,
    }


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

  def call(self, inputs, training, **kwargs):
    if len(inputs) == 2:
      inputs, targets = inputs[0], inputs[1]
    elif len(inputs) == 1:
      inputs, targets = inputs[0], None
    else:
      raise ValueError(f"Length of inputs should be 2 or 1, got {len(inputs)}")

    with tf.name_scope("Transformer"):
      attention_bias = preprocessing.get_padding_bias(inputs)

      encoder_outputs = self.encode(inputs, attention_bias, training)

      if targets is None:
        self.predict(encoder_outputs, attention_bias, training)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias, training)
        return logits

  def encode(self, inputs, attention_bias, training):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      inputs_embedded = self.embedding_softmax_layer(inputs)
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(inputs_embedded)[1]
        pos_encoding = preprocessing.get_position_encoding(length, self.params[
          "hidden_size"])
        encoder_inputs = inputs_embedded + pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(encoder_inputs,
                                       keep_prob=1.0 - self.params[
                                         "layer_postprocess_dropout"])

      encoder_outputs = self.encoder_stack(encoder_inputs,
                                           attention_bias=attention_bias,
                                           training=training)
      return encoder_outputs

  def decode(self, targets, encoder_outputs, attention_bias, training):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope('decode'):
      decoder_inputs = self.embedding_softmax_layer(targets)
      with tf.name_scope('shift_targets'):
        # Shift targets to the right, and remove the last element
        decoder_inputs = \
          tf.pad(decoder_inputs, paddings=[[0, 0], [0, 1], [0, 0]])[:, :-1, :]
      with tf.name_scope('add_pos_encoding'):
        length = tf.shape(decoder_inputs)[1]
        position_encoding = preprocessing.get_position_encoding(
          length, self.params["hidden_size"])
        decoder_inputs += position_encoding

      if training:
        decoder_inputs = tf.nn.dropout(decoder_inputs,
                                       keep_prob=1.0 - self.params[
                                         "layer_postprocess_dropout"])

    decoder_self_attention_bias = \
      preprocessing.get_decoder_self_attention_bias(length)

    outputs = self.decoder_stack(decoder_inputs, encoder_outputs,
                                 decoder_self_attention_bias, attention_bias,
                                 training)

    logits = self.embedding_softmax_layer(outputs)

    return logits

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
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
