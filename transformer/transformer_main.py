# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from transformer import misc
from transformer.utils.logs import logger
from transformer.utils.misc import keras_utils
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
    # TODO: continue

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


