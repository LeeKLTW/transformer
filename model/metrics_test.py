# -*- coding: utf-8 -*-
import tensorflow as tf
from . import metrics


class LossLayerTest(tf.test.TestCase):
  def test_init(self):
    params = dict(vocab_size=10, label_smoothing=0.1)
    test_layer = metrics.LossLayer(**params)
    for key, value in params.items():
      self.assertEqual(test_layer.__dict__.get(key, None), value,
                       f"Expect:{value}.Get:{test_layer.__dict__.get(key)}")

  # todo fix ValueError: slice index 1 of dimension 0 out of bounds. for 'loss_layer/loss/pad_to_same_length/strided_slice'
  def test_call(self):
    params = dict(vocab_size=10, label_smoothing=0.1)
    test_layer = metrics.LossLayer(**params)
    # test_layer([tf.constant([1,1]),tf.constant([1,2])])
    test_layer([[1, 1], [1, 2]])
    print(test_layer.losses)
