# -*- coding: utf-8 -*-
"""Public interface for flag definition.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import functools
import multiprocessing

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
      help=help_wrap(
        u"A list of (case insensitive) strings to specify the name "
        u" of training hooks. \n{}\n\ufeff Example `--hooks "
        u"ProfileHook, ExamplePerSecondHook`. \n "
        u"See transformer.utils.logs.hooks_helper for detail.".format(
          hook_list_str)))
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


def define_performance(num_parallel_calls=False, inter_op=False, intra_op=False,
                       synthetic_data=False, max_train_steps=False, dtype=False,
                       all_reduce_alg=False, num_packs=False,
                       tf_gpu_thread_mode=False,
                       datasets_num_private_threads=False,
                       datasets_num_parallel_batches=False,
                       dynamic_loss_scale=False, fp16_implementation=False,
                       loss_scale=False,
                       tf_data_experimental_slack=False, enable_xla=False,
                       force_v2_in_keras_compile=False,
                       training_dataset_cache=False):
  """Register flags for specifying performance tuning arguments.

  Args:
    num_parallel_calls: Create a flag to specify parallelism of data loading.
    inter_op: Create a flag to allow specification of inter op threads.
    intra_op: Create a flag to allow specification of intra op threads.
    synthetic_data: Create a flag to allow the use of synthetic data.
    max_train_steps: Create a flags to allow specification of maximum number
      of training steps
    dtype: Create flags for specifying dtype.
    all_reduce_alg: If set forces a specific algorithm for multi-gpu.
    num_packs: If set provides number of packs for MirroredStrategy's cross
      device ops.
    tf_gpu_thread_mode: gpu_private triggers us of private thread pool.
    datasets_num_private_threads: Number of private threads for datasets.
    datasets_num_parallel_batches: Determines how many batches to process in
    parallel when using map and batch from tf.data.
    dynamic_loss_scale: Allow the "loss_scale" flag to take on the value
      "dynamic". Only valid if `dtype` is True.
    fp16_implementation: Create fp16_implementation flag.
    loss_scale: Controls the loss scaling, normally for mixed-precision
      training. Can only be turned on if dtype is also True.
    tf_data_experimental_slack: Determines whether to enable tf.data's
      `experimental_slack` option.
    enable_xla: Determines if XLA (auto clustering) is turned on.
    force_v2_in_keras_compile: Forces the use of run_distribued path even if not
      using a `strategy`. This is not the same as
      `tf.distribute.OneDeviceStrategy`
    training_dataset_cache: Whether to cache the training dataset on workers.
       Typically used to improve training performance when training data is in
       remote storage and can fit into worker memory.

  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []
  if num_parallel_calls:
    flags.DEFINE_integer(
      name="num_parallel_calls", short_name="npc",
      default=multiprocessing.cpu_count(),
      help=help_wrap("The number of records that are processed in parallel "
                     "during input processing. This can be optimized per data "
                     "set but for generally homogeneous data sets, should be "
                     "approximately the number of available CPU cores."
                     "(default behavior)"))

  if inter_op:
    flags.DEFINE_integer(
      name="inter_op_parallelism_threads", short_name="inter", default=0,
      help=help_wrap("Number of inter_op_parallelism_threads to use for CPU."
                     "See TensorFlow config.protos for details."))

  if intra_op:
    flags.DEFINE_integer(
      name="intra_op_parallelism_threads", short_name="intra", default=0,
      help=help_wrap("Number of intra_op_parallelism_threads to use for CPU."
                     "See TensorFlow config.protos for details."))

  if synthetic_data:
    flags.DEFINE_boolean(
      name="use_synthetic_data", short_name="synth", default=False,
      help=help_wrap("If set, use fake data(zeros) instead of a real dataset."
                     "This mode is useful for performance debugging, as it "
                     "removes input processing steps, but will not learn "
                     "anything."))

  if max_train_steps:
    flags.DEFINE_integer(
      name="max_train_steps", short_name="mts", default=None,
      help=help_wrap("The model will stop training if global_step reaches this "
                     "value. If not set, training will run until the specified "
                     "number of epochs have run as usual. It is generally "
                     "recommended to set --train_epochs=1 when using this flag."))

  if dtype:
    flags.DEFINE_enum(
      name="dtype", short_name="dt", default="fp32",
      enum_values=DTYPE_MAP.keys(),
      help=help_wrap("The TensorFlow datatype used for calculations."
                     "Variables may be cast to a higher precision on a "
                     "case-by-case basis for numerical stability."))
    loss_scale_help_text = (
      "The amount to scale the loss by when the model is run. {}. Before "
      "gradients are computed, the loss is multiplied by the loss scale, "
      "making all gradients loss_scale times larger. To adjust for this, "
      "gradients are divided by the loss scale before being applied to "
      "variables. This is mathematically equivalent to training without a loss "
      "scale, but the loss scale helps avoid some intermediate gradients from "
      "underflowing to zero. If not provided default for fp16 is 128 and 1 for "
      "other dtypes. {}"
    )
    if dynamic_loss_scale:
      loss_scale_help_text = loss_scale_help_text.format(
        "This can be an int/float or the string 'dynamic'",
        "The string 'dynamic' can be used to dynamically determine the optimal "
        "loss scale during training, but currently this significantly slows down "
        "performance")

      loss_scale_validation_msg = ("loss_scale should be positive int/float or "
                                   "the string 'dynamic'.")
    else:
      loss_scale_help_text = loss_scale_help_text.format(
        "This must be an int/float", "")
      loss_scale_validation_msg = "loss_scale should be a positive int/float."

  if loss_scale:
    flags.DEFINE_string(
      name="loss_scale:", short_name="ls",
      default=None,
      help=help_wrap(loss_scale_help_text))

    @flags.validator(flag_name="ls", message=loss_scale_validation_msg)
    def _check_loss_scale(loss_scale):
      """Validator to check the loss scale flag is valid."""
      if loss_scale is None:
        return True

      if loss_scale == "dynamic" and dynamic_loss_scale:
        return True

      try:
        loss_scale = float(loss_scale)
      except ValueError:
        return False

      return loss_scale > 0

  if fp16_implementation:
    flags.DEFINE_enum(
      name="fp16_implementation", default="keras",
      enum_values=("keras", "graph_rewrite"),
      help=help_wrap(
        "When --dtype=fp16, how fp16 should be implemented. This has no impact "
        "on correctness. 'keras' use the tf.keras.mixed_precision API."
        "'graph_rewrite' uses the "
        "tf.train.experimental.enable_precision_graph_rewrite API."))

    @flags.multi_flags_validator(flag_names=
                                 ["fp16_implementation", "dtype", "loss_scale"])
    def _check_fp16_implementation(flags_dict):
      """Validator to check fp16_implementation flag is valid."""
      if (flags_dict["fp16_implementation"] == "graph_rewrite" and
              flags_dict["dtype"] != "fp16"):
        raise flags.ValidationError("--fp16_implementation should not be "
                                    "specified unless --dtype==fp16")
      return True

  if all_reduce_alg:
    flags.DEFINE_string(
      name="all_reduce_alg", short_name="ara",default=None,
      help=help_wrap(
        "Define algorithm to use for performing all-reduce.When specified with "
        "MirroredStrategy for single worker, this controls "
        "tf.contrib.distribute.AllReduceCrossTowerOps. When specified with "
        "MultiWorkerMirroredStrategy, this controls "
        "tf.distribute.experimental.CollectiveCommunication; valid options are "
        "`ring` and `nccl`"))

  if num_packs:
    flags.DEFINE_integer(
      name="num_packs",default=1,
      help=help_wrap(
        "Sets `num_packs` in the cross device pos used in MirroredStrategy. "
        "For details, see tf.distribute.NcclAllReduce."))

  if tf_gpu_thread_mode:
    flags.DEFINE_string(
      name="tf_gpu_thread_mode", short_name="gt_mode",default=None,
      help=help_wrap(
        "Whether and how the GPU device uses its own threadpool."))

    flags.DEFINE_integer(
      name="per_gpu_thread_count", short_name="pgtc",default=0,
      help=help_wrap(
        "The number of threads to use for GPU. Only valid when "
        "tf_gpu_thread_mode is not global."))

  if datasets_num_parallel_batches:
    flags.DEFINE_integer(
      name="datasets_num_parallel_batches",default=None,
      help=help_wrap(
        "Determines how many batches to process in parallel when using map and "
        "batch from tf.data."))

  if training_dataset_cache:
    flags.DEFINE_boolean(
      name="training_dataset_cache",default=False,
      help=help_wrap(
        "Determines whether to cache the training dataset on workers."
        "Typically used to improve training performance when training data is in "
        "remote storage and can fit into worker memory."))

  if tf_data_experimental_slack:
    flags.DEFINE_boolean(
      name="tf_data_experimental_slack",default=False,
      help=help_wrap(
        "Whether to enable tf.data's `experimental_slack` option."))


  if enable_xla:
    flags.DEFINE_boolean(
      name="enable_xla",default=False,
      help=help_wrap(
        "Whether to enable XLA auto jit compilation"))

  if force_v2_in_keras_compile:
    flags.DEFINE_integer(
      name="force_v2_in_keras_compile",default=None,
      help=help_wrap(
        "Forces the use of run_distributed path even if not using a `strategy`."
        "This is not the same as `tf.distribute.OneDeviceStrategy`."))
  return key_flags


# TODO
def define_benchmark():
  pass


# TODO
def define_device(tpu):
  pass

#FIXME: fix flags
define_benchmark = register_key_flags_in_core(define_benchmark)
define_benchmark = register_key_flags_in_core(define_benchmark)
define_benchmark = register_key_flags_in_core(define_benchmark)
define_benchmark = register_key_flags_in_core(define_benchmark)
