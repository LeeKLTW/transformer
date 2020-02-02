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


def define_base(data_dir=True, model_dir=True, clean=False, train_epochs=False,
                epochs_between_evals=False, stop_threshold=False,
                batch_size=True, num_gpu=False, hooks=False, export_dir=False,
                distribution_strategy=False, run_eagerly=False):
  """Register base flags.

  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    clean: Create a flag for removing the model_dir.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    num_gpu: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.
    distribution_strategy: Create a flag to specify which Distribution Strategy
      to use.
    run_eagerly: Create a flag to specify to run eagerly op by op.
  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []
  if data_dir:
    flags.DEFINE_string(name="data_dir", short_name="dd", default="/tmp",
                        help=help_wrap("The location of the input data"))
    key_flags.append("data_dir")

  if model_dir:
    flags.DEFINE_string(
      name="model_dir", short_name="md", default="/tmp",
      help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

  if clean:
    flags.DEFINE_boolean(
      name="clean", default=False,
      help=help_wrap("If set, model_dir will be removed if it exists."))
    key_flags.append("clean")

  if train_epochs:
    flags.DEFINE_integer(
      name="train_epochs", short_name="te", default=1,
      help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

  if epochs_between_evals:
    flags.DEFINE_integer(
      name="epochs_between_evals", short_name="ebe", default=1,
      help=help_wrap(
        "The number of training epochs to run between evaluations."))
    key_flags.append("epochs_between_evals")

  if stop_threshold:
    flags.DEFINE_float(
      name="stop_threshold", short_name="st", default=None,
      help=help_wrap("If passed, training will stop at the earlier of training_"
                     "epochs and when the evaluation metric is greater than or "
                     "equal to stop_threshold"))
  if batch_size:
    flags.DEFINE_integer(
      name="batch_size", short_name="bs", default=32,
      help=help_wrap("Batch size for training and evaluation. When using "
                     "multiple GPUs, this is the global batch size for all "
                     "devices. For example, if the batch size is 32 and there "
                     "are 4 GPUs, each GPU will get 8 examples on each step."))
    key_flags.append("batch_size")

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
