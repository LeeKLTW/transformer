# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_examples_per_second_hook():
  pass



def get_logging_metric_hook():
  pass


HOOKS = {
  'examplespersecondhook': get_examples_per_second_hook,
  'loggingmetrichook': get_logging_metric_hook,
}
