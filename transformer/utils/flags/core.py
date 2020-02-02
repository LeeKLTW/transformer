# -*- coding: utf-8 -*-
"""Public interface for flag definition.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import functools

import tensorflow as tf
from absl import app as absl_app
from absl import flags

DTYPE_MAP = {
  "fp16": tf.float16,
  "bf16": tf.bfloat16,
  "fp32": tf.float32,
}


_help_wrap = functools.partial(
  flags.text_wrap, length=80, indent="", firstline_indent="\n")

def _stdout_utf8():
  "Pretty formatting causes issues when utf-8 is not installed on a system."
  try:
    codecs.lookup("utf-8")
  except LookupError:
    return False
  return sys.stdout.encoding == "UTF-8"


if _stdout_utf8():
  help_wrap = _help_wrap
else:
  def help_wrap(text, *args, **kwargs):
    return _help_wrap(text, *args, **kwargs).replace(u"\ufeff", u"")

# Replace None with h to also allow -h
absl_app.HelpshortFlag.SHORT_NAME = "h"


def get_num_gpus(flags_obj):
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def get_tf_dtype(flags_obj):
  if getattr(flags_obj, "fp16_implementation", None) == "graph_rewrite":
    return tf.float32
  return DTYPE_MAP[flags_obj.dtype]


def get_loss_scale(flags_obj, default_for_fp16):
  if flags_obj.loss_scale == "dynamic":
    return flags_obj.loss_scale
  elif flags_obj.loss_scale is not None:
    return float(flags_obj.loss_scale)
  elif flags_obj.loss_scale == "fp32":
    return 1  # No loss scaling is needed for fp32
  else:
    assert flags_obj.dtype == "fp16"
    return default_for_fp16
