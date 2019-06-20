# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

from absl import app
from absl import flags
from model import transformer

from utils.flags._transformer import define_transformer_flags, get_model_params
from utils.flags import core
from utils.logs import logger


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class TransformerTask(object):
  def __init__(self, flags_obj):
    self.flags_obj = flags_obj
    self.predict_model = None
    self.params = params = get_model_params(flags_obj.param_set)

    params["num_gpus"] = 0
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["num_parallel_calls"] = flags_obj.num_parallel_calls
    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None

  def train(self):
    params, flags_obj, is_train = self.params, self.flags_obj, True

    _ensure_dir(flags_obj.model_dir)

    model = transformer.create_model(params, is_train) #todo
    optimizer = self._create_optimizer()
    model.compile(optimizer)
    model.summary()

    pass

  # todo
  def predict(self):
    pass

  # todo
  def eval(self):
    pass

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    optimizer = tf.keras.optimizers.Adam(lr=params["learning_rate"],
                                   beta_1=params["optimizer_adam_beta1"],
                                   beta_2=params["optimizer_adam_beta2"],
                                   epsilon=params["optimizer_adam_epsilon"])
    return optimizer


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    if flags_obj.mode == "train":
      task.train()

    elif flags_obj.mode == "predict":
      task.predict()

    elif flags_obj.mode == "eval":
      task.eval()

    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  define_transformer_flags()
  app.run(main)
