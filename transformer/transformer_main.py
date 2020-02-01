# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from transformer import misc
from utils.logs import logger

INF = int(1e9)
BLEU_DUR = "bleu"
_SINGLE_SAMPLE = 1

#TODO
class TransformerTask(object):
  pass


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    #TODO: continue
  pass

if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  misc.define_transformer_flags()
  app.run(main)


