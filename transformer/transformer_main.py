# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from transformer.utils import misc
from transformer.utils.logs import logger
from transformer.utils.misc import keras_utils
from transformer.utils.misc import distribution_utils
from transformer.utils.flags import core as flags_core

INF = int(1e9)
BLEU_DUR = "bleu"
_SINGLE_SAMPLE = 1


class TransformerTask(object):
  """Main entry of Transformer model."""
  def __init__(self, flags_obj):
    self.flags_obj = flags_obj
    self.predict_model = None
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_max_length"] = flags_obj.padded_decode
    params["num_parallel_calls"] = (flags_obj.num_parallel_calls or
                                    tf.data.experimental.AUTOTUNE)    #FIXME
    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals

    self.distribution_strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu or "")

    if self.use_tpu:
      params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
      if not params["static_batch"]:
        raise ValueError("TPU requies static batch for input data.")
    else:
      logging.info("Running transformer with num_gpus = %d", num_gpus)

    # TODO: continue

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy,
                        tf.distribute.experimental.TPUStrategy)


  pass


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)

    if flags_obj.tf_gpu_thread_mode:
      keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)

    if flags_obj.mode == "train":
      task.train()
    elif flags_obj.mode == "predict":
      task.predict()
    elif flags_obj.mode == "eval":
      task.eval()
    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  misc.define_transformer_flags()
  app.run(main)


