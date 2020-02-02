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

from transformer.utils.logs import hooks_helper

DTYPE_MAP = {
  "fp16": tf.float16,
  "bf16": tf.bfloat16,
  "fp32": tf.float32,
}


def register_key_flags_in_core(f):
  """Defines a function in core.py, and registers its key flags.

  absl uses the location of a flags.declare_key_flag() to determine the context
  in which a flag is key. By making all declares in core, this allows model
  main functions to call flags.adopt_module_key_flags() on core and correctly
  chain key flags.

  Args:
    f:  The function to be wrapped

  Returns:
    The "core-defined" version of the input function.
  """

  def core_fn(*args, **kwargs):
    key_flags = f(*args, **kwargs)
    [flags.declare_key_flag(flag) for flag in key_flags]

  return core_fn


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



  if num_gpu:
    flags.DEFINE_integer(
      name="num_gpu", short_name="ng", default=1,
      help=help_wrap("How many GPUs to use at each worker with the "
                     "DistributionStrategies API. The default is 1."))

  if run_eagerly:
    flags.DEFINE_boolean(
      name="run_eagerly", default=False,
      help=help_wrap("Run the model op by op without building model function."))

  if hooks:
    hook_list_str = (u"\ufeff Hook:\n" + u"\n".join(
      [u"\ufeee    {}".format(key) for key in hooks_helper.HOOKS]))
    flags.DEFINE_string(
      name="hooks", short_name="hk", default="LoggingTensorHook",
      help=help_wrap(u"A list of (case insensitive) strings to specify the name "
                     u" of training hooks. \n{}\n\ufeff Example `--hooks "
                     u"ProfileHook, ExamplePerSecondHook`. \n "
                     u"See transformer.utils.logs.hooks_helper for detail.".format(hook_list_str)))
    key_flags.append("hooks")

  if export_dir:
    flags.DEFINE_string(
      name="export_dir", short_name="ed", default=None,
      help=help_wrap("If set, a SavedModel serialization of the model will be "
                     "export to this directory at the end of training. See README "
                     "for more details and relevant links."))
    key_flags.append("export_dir")

  if distribution_strategy:
    flags.DEFINE_string(
      name="distribution_strategy", short_name="ds", default="mirrored",
      help=help_wrap("The Distribution Strategy to use for traing. Accepted "
                     "values are 'off', 'one_device', 'mirrred',"
                     "'parameter_server', 'collective', case insensitive. 'off' "
                     "means not to use Ditribution Strategy; 'default' means to "
                     "choose from 'MirroredStrategy' or 'OneDevicesStrategy' "
                     "according to the number of GPUs."))
    key_flags.append("distribution_strategy")
  return key_flags
