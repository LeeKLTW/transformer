# -*- coding: utf-8 -*-
"""Hooks helper to return a list of TensorFlow hooks for training by name.

More hooks can be added to this set. To add a new hook, 1) add the new hook to
the registry in HOOKS, 2) add a corresponding function that parses out necessary
parameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])


def get_logging_tensor_hook(every_n_iter=100, tensor_to_log=None, **kwargs):
  """Function to get LoggingTensorHook.

  Args:
    every_n_iter: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
    tensors_to_log: List of tensor names or dictionary mapping labels to tensor
      names. If not set, log _TENSORS_TO_LOG by default.
    **kwargs: a dictionary of arguments to LoggingTensorHook.

  Returns:
    Returns a LoggingTensorHook with a standard set of tensors that will be
    printed to stdout.
  """
  if tensor_to_log is None:
    tensor_to_log = _TENSORS_TO_LOG

  return tf.estimator.LoggingTensorHook(
    tensors=tensor_to_log,
    every_n_iter=every_n_iter)


def get_profiler_hook(model_dir, save_steps=1000, **kwargs):
  """Function to get ProfilerHook.

  Args:
    model_dir: The directory to save the profile traces to.
    save_steps: `int`, print profile traces every N steps.
    **kwargs: a dictionary of arguments to ProfilerHook.

  Returns:
    Returns a ProfilerHook that writes out timelines that can be loaded into
    profiling tools like chrome://tracing.
  """
  return tf.estimator.ProfilerHook(save_steps=save_steps, output_dir=model_dir)


# TODO
def get_example_per_second_hook():
  pass


# TODO
def get_logging_metric_hook():
  pass


# TODO
def get_step_counter_hook():
  pass


HOOKS = {'loggingtensorhook': get_logging_tensor_hook,
         'profilerhook': get_profiler_hook,
         'examplespersecondhook': get_example_per_second_hook,
         'loggingmetrichook': get_logging_metric_hook,
         'stepcounterhook': get_step_counter_hook,
         }
