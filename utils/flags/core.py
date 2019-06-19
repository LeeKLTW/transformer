# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ._base import define_base


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
    [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned

  return core_fn


define_base = register_key_flags_in_core(define_base)

#todo
def define_performance():
  pass

#todo
def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)

#todo
def define_benchmark():
  pass

#todo
def define_device():
  pass


def help_wrap():
  pass


def require_cloud_storage():
  pass
