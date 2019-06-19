# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from absl import app
from absl import flags

from utils.flags._transformer import define_transformer_flags
from utils.logs import logger


class TransformerTask(object):
  def __init__(self, flags_obj):
    self.flags_obj = flags_obj

  # todo
  def train(self):
    pass


def main(_):
  flags_obj = flags.FLAGS
  # todo logger
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    if flags_obj.mode == "train":
      # todo class task
      task.train()

    elif flags_obj.mode == "predict":
      pass

    elif flags_obj.mode == "eval":
      pass

    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  define_transformer_flags()
  app.run(main)
