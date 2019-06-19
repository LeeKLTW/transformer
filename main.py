# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from absl import app

from utils.flags import define_transformer_flags


class TransFormerTask(object):
  def __init__(self, flags_obj):
    self.flags_obj = flags_obj


def main(_):
  pass


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  define_transformer_flags()
  app.run(main)

