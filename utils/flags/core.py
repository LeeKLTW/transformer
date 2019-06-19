# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ._base import define_base  # pylint: disable=relative-beyond-top-level
from ._performance import define_performance  # pylint: disable=relative-beyond-top-level
from ._benchmark import define_benchmark  # pylint: disable=relative-beyond-top-level
from ._device import define_device  # pylint: disable=relative-beyond-top-level


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
    [flags.declare_key_flag(fl) for fl in
     key_flags]  # pylint: disable=expression-not-assigned

  return core_fn


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


define_base = register_key_flags_in_core(define_base)
define_performance = register_key_flags_in_core(define_performance)
define_benchmark = register_key_flags_in_core(define_benchmark)
define_device = register_key_flags_in_core(define_device)
