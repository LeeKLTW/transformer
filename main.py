# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf

from model import data_pipeline
from model import optimizer
from model import transformer
from model import translate

from utils import compute_bleu
from utils import tokenizer
from utils import misc
from utils.logs import logger

class TransFormerTask(object):
  def __int__(self,flags_obj):
    self.flags_obj = flags_obj

def main(_):
  flags_obj = flags.FLAGS