# -*- coding: utf-8 -*-
from transformer import model_params
PARAMS_MAP = {
  'tiny': model_params.TINY_PARAMS,
  'base': model_params.BASE_PARAMS,
  'big': model_params.BIG_PARAMS,
}
#TODO
def define_transformer_flags():
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