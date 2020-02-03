# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hook that counts examples per second every N steps or seconds."""

import tensorflow as tf
from . import logger


class ExamplePerSecondHook(tf.estimator.SessionRunHook):
  """Hook to print out examples per second.

  Total time is tracked and then divided by the total number of steps
  to get the average step time and then batch_size is used to determine
  the running average of examples per second. The examples per second for the
  most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=None,
               every_n_secs=None,
               warm_steps=0,
               metric_loger=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
        "Exactly one of every_n_steps and every_n_secs should be provided.")

    self._logger = metric_loger or logger.BaseBenchmarkLogger()
    self._timer = tf.estimator.SecondOrStepTimer(every_n_steps=every_n_steps,
                                                 every_n_secs=every_n_secs)
    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._warm_steps = warm_steps
    self.current_examples_per_sec_list = []

  # TODO: continue
  def begin(self):
    """Called once before using the session to check global step."""
    # self._globel_step_tensor =
    pass

  #TODO
  def before_run(self):
    """Called before each call to run().

    Args:
      run_context: A SessionRunContext object.

    Returns:
      A SessionRunArgs object or None if never triggered.
    """
    pass

  #TODO
  def after_run(self):
    """Called after each call to run().

    Args:
      run_context: A SessionRunContext object.
      run_values: A SessionRunValues object.
    """
    pass
