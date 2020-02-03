# -*- coding: utf-8 -*-
from absl import flags


from transformer import model_params
from transformer.utils.flags import core as flags_core

PARAMS_MAP = {
  'tiny': model_params.TINY_PARAMS,
  'base': model_params.BASE_PARAMS,
  'big': model_params.BIG_PARAMS,
}

FLAGS = flags.FLAGS

def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  flags_core.define_base(num_gpu=True, distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=True,
      loss_scale=True,
      all_reduce_alg=True,
      num_packs=True,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      enable_xla=True,
      force_v2_in_keras_compile=True,
      fp16_implementation=True
  )
  flags_core.define_benchmark()
  flags_core.define_device(tpu=True)
  # TODO:continue

  pass


def get_model_params(param_set, num_gpus):
  """Gets predefined model params."""

  if num_gpus > 1:
    if param_set == 'big':
      return model_params.BIG_MULTI_GPU_PARAMS.copy()
    elif param_set == 'base':
      return model_params.BASE_MULTI_GPU_PARAMS.copy()
    else:
      raise ValueError("Not valid params: param_set={} num_gpus={}".format(
        param_set, num_gpus))
  return PARAMS_MAP[param_set].copy()
