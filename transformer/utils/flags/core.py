# -*- coding: utf-8 -*-
"""Public interface for flag definition.
"""
from transformer.utils.flags import _conventions

help_wrap = _conventions.help_wrap


def get_num_gpus(flags_obj):
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type =="GPU"])