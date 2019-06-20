# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Logging utilities for benchmark.

For collecting local environment metrics like CPU and memory, certain python
packages need be installed. See README for details.
"""
import os
import threading
import multiprocessing
from absl import flags
import tensorflow as tf
import numbers  # (ABCs) for numbers, according to PEP 3141.
import datetime
import json

from tensorflow.python.client import device_lib

METRIC_LOG_FILE_NAME = "metric.log"
RUN_STATUS_SUCCESS = "success"
RUN_STATUS_FAILURE = "failure"
RUN_STATUS_RUNNING = "running"
_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"

FLAGS = flags.FLAGS

# Use get_benchmark_logger to access a logger.
_benchmark_logger = None
_logger_lock = threading.Lock()



class BaseBenchmarkLogger(object):
  """Class to log the benchmark information to STDOUT."""

  def log_evaluation_result(self, eval_results):
    """Log the evaluation result.

    The evaluate result is a dictionary that contains metrics defined in
    model_fn. It also contains a entry for global_step which contains the value
    of the global step when evaluation was performed.

    Args:
      eval_results: dict, the result of evaluate.
    """
    if not isinstance(eval_results, dict):
      tf.logging.warning(
        "eval_results should be dictionary for logging. Got %s",
        type(eval_results))
      return
    global_step = eval_results[tf.GraphKeys.GLOBAL_STEP]
    for key in sorted(eval_results):
      if key != tf.GraphKeys.GLOBAL_STEP:
        self.log_metric(key, eval_results[key], global_step=global_step)

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to local file.

    Currently the logging is done in a synchronized way. This should be updated
    to log asynchronously.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    metric = _process_metric_to_json(name, value, unit, global_step, extras)
    if metric:
      tf.logging.info("Benchmark metric: %s", metric)

  def log_run_info(self, model_name, dataset_name, run_params, test_id=None):
    tf.logging.info(
      "Benchmark run: %s", _gather_run_info(model_name, dataset_name,
                                            run_params, test_id))

  def on_finish(self, status):
    pass


class BenchmarkFileLogger(BaseBenchmarkLogger):
  """Class to log the benchmark information to local disk."""

  def __init__(self, logging_dir):
    super(BenchmarkFileLogger, self).__init__()
    self._logging_dir = logging_dir
    if not tf.gfile.IsDirectory(self._logging_dir):
      tf.gfile.MakeDirs(self._logging_dir)
    self._metric_file_handler = tf.gfile.GFile(
      os.path.join(self._logging_dir, METRIC_LOG_FILE_NAME), "a")

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to local file.

    Currently the logging is done in a synchronized way. This should be updated
    to log asynchronously.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    metric = _process_metric_to_json(name, value, unit, global_step, extras)
    if metric:
      try:
        json.dump(metric, self._metric_file_handler)
        self._metric_file_handler.write("\n")
        self._metric_file_handler.flush()
      except (TypeError, ValueError) as e:
        tf.logging.warning(
          "Failed to dump metric to log file: name %s, value %s, error %s",
          name, value, e)


def config_benchmark_logger(flag_obj=None):
  _logger_lock.acquire()
  try:
    global _benchmark_logger # pylint: disable=global-statement
    if not flag_obj:
      flag_obj = FLAGS
    if (not hasattr(flag_obj, "benchmark_logger_type") or
            flag_obj.benchmark_logger_type == "BaseBenchmarkLogger"):
      _benchmark_logger = BaseBenchmarkLogger()
    elif flag_obj.benchmark_logger_type == "BenchmarkFileLogger":
      _benchmark_logger = BenchmarkFileLogger(flag_obj.benchmark_log_dir)
    else:
      raise ValueError("Unrecognized benchmark_logger_type: %s"
                       % flag_obj.benchmark_logger_type)
  finally:
    _logger_lock.release()
  return _benchmark_logger


def benchmark_context(flag_obj):
  """Context of benchmark, which will update status of the run accordingly."""
  benchmark_logger = config_benchmark_logger(flag_obj)
  try:
    yield benchmark_logger.on_finish(RUN_STATUS_SUCCESS)
  except Exception:  # pylint: disable=broad-except
    # Catch all the exception, update the run status to be failure, and re-raise
    benchmark_logger.on_finish(RUN_STATUS_FAILURE)
    raise

def _convert_to_json_dict(input_dict):
  if input_dict:
    return [{"name": k, "value": v} for k, v in sorted(input_dict.items())]
  else:
    return []

def _process_metric_to_json(name, value, unit=None, global_step=None,
                            extras=None):
  """Validate the metric data and generate JSON for insert."""
  if not isinstance(value, numbers.Number):
    tf.logging.warning(
      "Metric value to log should be a number. Got %s", type(value))
    return None

  extras = _convert_to_json_dict(extras)
  return {
    "name": name,
    "value": float(value),
    "unit": unit,
    "global_step": global_step,
    "timestamp": datetime.datetime.utcnow().strftime(
      _DATE_TIME_FORMAT_PATTERN),
    "extras": extras}


def _collect_tensorflow_info(run_info):
  run_info["tensorflow_version"] = {
      "version": tf.__version__, "git_hash": tf.GIT_VERSION}


def _collect_run_params(run_info, run_params):
  """Log the parameter information for the benchmark run."""
  def process_param(name, value):
    type_check = {
        str: {"name": name, "string_value": value},
        int: {"name": name, "long_value": value},
        bool: {"name": name, "bool_value": str(value)},
        float: {"name": name, "float_value": value},
    }
    return type_check.get(type(value),
                          {"name": name, "string_value": str(value)})
  if run_params:
    run_info["run_parameters"] = [
        process_param(k, v) for k, v in sorted(run_params.items())]


def _collect_tensorflow_environment_variables(run_info):
  run_info["tensorflow_environment_variables"] = [
      {"name": k, "value": v}
      for k, v in sorted(os.environ.items()) if k.startswith("TF_")]


# The following code is mirrored from tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
  """Collect the CPU information for the local environment."""
  cpu_info = {}

  cpu_info["num_cores"] = multiprocessing.cpu_count()

  try:
    # Note: cpuinfo is not installed in the TensorFlow OSS tree.
    # It is installable via pip.
    import cpuinfo    # pylint: disable=bad-option-value

    info = cpuinfo.get_cpu_info()
    cpu_info["cpu_info"] = info["brand"]
    cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6

    run_info["machine_config"]["cpu_info"] = cpu_info
  except ImportError:
    tf.logging.warn(
        "'cpuinfo' not imported. CPU info will not be logged.")


def _collect_gpu_info(run_info, session_config=None):
  """Collect local GPU information by TF device library."""
  gpu_info = {}
  local_device_protos = device_lib.list_local_devices(session_config)

  gpu_info["count"] = len([d for d in local_device_protos
                           if d.device_type == "GPU"])
  # The device description usually is a JSON string, which contains the GPU
  # model info, eg:
  # "device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0"
  for d in local_device_protos:
    if d.device_type == "GPU":
      gpu_info["model"] = _parse_gpu_model(d.physical_device_desc)
      # Assume all the GPU connected are same model
      break
  run_info["machine_config"]["gpu_info"] = gpu_info


def _collect_memory_info(run_info):
  try:
    # Note: psutil is not installed in the TensorFlow OSS tree.
    # It is installable via pip.
    import psutil   # pylint: disable=bad-option-value
    vmem = psutil.virtual_memory()
    run_info["machine_config"]["memory_total"] = vmem.total
    run_info["machine_config"]["memory_available"] = vmem.available
  except ImportError:
    tf.logging.warn(
        "'psutil' not imported. Memory info will not be logged.")


def _collect_test_environment(run_info):#pylint: disable=unused-argument
  pass

def _parse_gpu_model(physical_device_desc):
  # Assume all the GPU connected are same model
  for kv in physical_device_desc.split(","):
    k, _, v = kv.partition(":")
    if k.strip() == "name":
      return v.strip()
  return None

def _gather_run_info(model_name, dataset_name, run_params, test_id):
  """Collect the benchmark run information for the local environment."""
  run_info = {
    "model_name": model_name,
    "dataset":{"name": dataset_name},
    "machine_config": {},
    "test_id": test_id,
    "run_date": datetime.datetime.utcnow().strftime(_DATE_TIME_FORMAT_PATTERN)
  }

  session_config = None
  if "session_config" in run_params:
    session_config = run_params["session_config"]
  _collect_tensorflow_info(run_info)
  _collect_tensorflow_environment_variables(run_info)
  _collect_run_params(run_info, run_params)
  _collect_cpu_info(run_info)
  _collect_gpu_info(run_info, session_config)
  _collect_memory_info(run_info)
  _collect_test_environment(run_info)
  return run_info



