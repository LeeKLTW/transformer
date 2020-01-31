# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import app

from transformer import misc

INF = int(1e9)
BLEU_DUR = "bleu"
_SINGLE_SAMPLE = 1

def main(_):
  #TODO: continue
  pass

if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  misc.define_transformer_flags()
  app.run(main)


