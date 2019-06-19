# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl import flags
import tensorflow as tf

from . import core # pylint: disable=relative-beyond-top-level


def define_flags():
  core.define_base()
  core.define_performance()
  core.define_benchmark()


class BaseTester(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    super(BaseTester, cls).setUpClass()
    define_flags()

  def test_default_setting(self):
    defaults = dict(
      data_dir="dfgasf",
      model_dir="dfsdkjgbs",
      train_epochs=534,
      epochs_between_evals=15,
      batch_size=256,
      hooks=["LoggingTensorHook"],
      num_parallel_calls=18,
      inter_op_parallelism_threads=5,
      intra_op_parallelism_threads=10,
    )

    core.set_defaults(**defaults)
    core.parse_flags()

    for key, value in defaults.items():
      self.assertEqual(flags.FLAGS.get_flag_value(key, default=None), value)

  def test_benchmark_setting(self):
    defaults = dict(
      hooks=["LoggingMetricHook"],
      benchmark_log_dir="/tmp/12345",
      gcp_project="project_abc",
    )

    core.set_defaults(**defaults)
    core.parse_flags()

    for key, value in defaults.items():
      self.assertEqual(flags.FLAGS.get_flag_value(name=key, default=None),
                       value)

  def test_booleans(self):
    pass

  def test_parse_dtype_info(self):
    pass

  def test_get_nondefault_flags_as_str(self):
    pass
