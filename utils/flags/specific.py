# -*- coding: utf-8 -*-
from absl import flags
from .core import define_base, set_defaults, define_performance, \
  define_benchmark, define_device
from ._conventions import help_wrap

from model import model_params

PARAMS_MAP = {
  'tiny': model_params.TINY_PARAMS,
  'base': model_params.BASE_PARAMS,
  'big': model_params.BIG_PARAMS,
}


def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, train_epochs, etc.).
  define_base()
  define_performance(
    num_parallel_calls=True,
    inter_op=False,
    intra_op=False,
    synthetic_data=True,
    max_train_steps=False,
    dtype=False,
    all_reduce_alg=True
  )
  define_benchmark()
  define_device(tpu=True)

  flags.DEFINE_integer(
    name='train_steps', short_name='ts', default=300000,
    help=help_wrap('The number of steps used to train.'))
  flags.DEFINE_integer(
    name='steps_between_evals', short_name='sbe', default=1000,
    help=help_wrap(
      'The Number of training steps to run between evaluations. This is '
      'used if --train_steps is defined.'))
  flags.DEFINE_boolean(
    name='enable_time_history', default=True,
    help='Whether to enable TimeHistory callback.')
  flags.DEFINE_boolean(
    name='enable_tensorboard', default=False,
    help='Whether to enable Tensorboard callback.')
  flags.DEFINE_string(
    name='profile_steps', default=None,
    help='Save profiling data to model dir at given range of steps. The '
         'value must be a comma separated pair of positive integers, specifying '
         'the first and last step to profile. For example, "--profile_steps=2,4" '
         'triggers the profiler to process 3 steps, starting from the 2nd step. '
         'Note that profiler has a non-trivial performance overhead, and the '
         'output file can be gigantic if profiling many steps.')

  # Add transformer-specific flags
  flags.DEFINE_enum(
    name='param_set', short_name='mp', default='big',
    enum_values=PARAMS_MAP.keys(),
    help=help_wrap(
      'Parameter set to use when creating and training the model. The '
      'parameters define the input shape (batch size and max length), '
      'model configuration (size of embedding, # of hidden layers, etc.), '
      'and various other settings. The big parameter set increases the '
      'default batch size, embedding/hidden size, and filter size. For a '
      'complete list of parameters, please see model/model_params.py.'))

  flags.DEFINE_bool(
    name='static_batch', short_name='sb', default=False,
    help=help_wrap(
      'Whether the batches in the dataset should have static shapes. In '
      'general, this setting should be False. Dynamic shapes allow the '
      'inputs to be grouped so that the number of padding tokens is '
      'minimized, and helps model training. In cases where the input shape '
      'must be static (e.g. running on TPU), this setting will be ignored '
      'and static batching will always be used.'))
  flags.DEFINE_integer(
    name='max_length', short_name='ml', default=256,
    help=help_wrap(
      'Max sentence length for Transformer. Default is 256. Note: Usually '
      'it is more effective to use a smaller max length if static_batch is '
      'enabled, e.g. 64.'))

  # Flags for training with steps (may be used for debugging)
  flags.DEFINE_integer(
    name='validation_steps', short_name='vs', default=64,
    help=help_wrap('The number of steps used in validation.'))

  # BLEU score computation
  flags.DEFINE_string(
    name='bleu_source', short_name='bls', default=None,
    help=help_wrap(
      'Path to source file containing text translate when calculating the '
      'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
      'Use the flag --stop_threshold to stop the script based on the '
      'uncased BLEU score.'))
  flags.DEFINE_string(
    name='bleu_ref', short_name='blr', default=None,
    help=help_wrap(
      'Path to source file containing text translate when calculating the '
      'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
      'Use the flag --stop_threshold to stop the script based on the '
      'uncased BLEU score.'))
  flags.DEFINE_string(
    name='vocab_file', short_name='vf', default=None,
    help=help_wrap(
      'Path to subtoken vocabulary file. If data_download.py was used to '
      'download and encode the training data, look in the data_dir to find '
      'the vocab file.'))
  flags.DEFINE_string(
    name='mode', default='train',
    help=help_wrap('mode: train, eval, or predict'))

  set_defaults(data_dir='/tmp/translate_ende',
               model_dir='/tmp/transformer_model',
               batch_size=None,
               train_epochs=10)

  # pylint: disable=unused-variable
  @flags.multi_flags_validator(
    ['mode', 'train_epochs'],
    message='--train_epochs must be defined in train mode')
  def _check_train_limits(flag_dict):
    if flag_dict['mode'] == 'train':
      return flag_dict['train_epochs'] is not None
    return True

  @flags.multi_flags_validator(
    ['bleu_source', 'bleu_ref'],
    message='Both or neither --bleu_source and --bleu_ref must be defined.')
  def _check_bleu_files(flags_dict):
    return (flags_dict['bleu_source'] is None) == (
            flags_dict['bleu_ref'] is None)

  @flags.multi_flags_validator(
    ['bleu_source', 'bleu_ref', 'vocab_file'],
    message='--vocab_file must be defined if --bleu_source and --bleu_ref '
            'are defined.')
  def _check_bleu_vocab_file(flags_dict):
    if flags_dict['bleu_source'] and flags_dict['bleu_ref']:
      return flags_dict['vocab_file'] is not None
    return True

  @flags.multi_flags_validator(
    ['export_dir', 'vocab_file'],
    message='--vocab_file must be defined if --export_dir is set.')
  def _check_export_vocab_file(flags_dict):
    if flags_dict['export_dir']:
      return flags_dict['vocab_file'] is not None
    return True