# -*- coding: utf-8 -*-
import multiprocessing
import os
from absl import logging

def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                  datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info("Local CPU cores: %s", cpu_count)
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ["TF_GPU_THREAD_MODE"] = gpu_thread_mode
  os.environ["TF_GPU_THREAD_COUNT"] = str(per_gpu_thread_count)
  logging.info("TF_GPU_THREAD_MODE: %s", os.environ["TF_GPU_THREAD_MODE"])
  logging.info("TF_GPU_THREAD_COUNT: %s", os.environ["TF_GPU_THREAD_COUNT"])
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(
      cpu_count - total_gpu_thread_count - num_runtime_threads,
      num_gpus * 8)
    logging.info("Set datasets_num_private_threads to %s",
                 datasets_num_private_threads)