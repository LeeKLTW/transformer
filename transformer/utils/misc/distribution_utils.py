# -*- coding: utf-8 -*-

def get_distribution_strategy(distribution_strategy="mirrored",
                              num_gpus=0,
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None):
  """Return a DistributionStrategy for running the model.

  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are 'off', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case insensitive.
      'off' means not to use Distribution Strategy; 'tpu' means to use
      TPUStrategy using `tpu_address`.
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
      "ring" and "nccl".  If None, DistributionStrategy will choose based on
      device topology.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
    tpu_address: Optional. String that represents TPU to connect to. Must not
      be None if `distribution_strategy` is set to `tpu`.
  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative or if
      `distribution_strategy` is `tpu` but `tpu_address` is not specified.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpu` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  # TODO: continue the distribution_strategy cases

  pass