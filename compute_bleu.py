# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app as absl_app
from absl import flags

from utils.flags import core as flags_core
from utils import tokenizer
from utils import metrics


def bleu_tokenize(x):
  pass


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
  """Compute BLEU for two files (reference and hypothesis translation)."""
  ref_lines = tokenizer.native_to_unicode(
    tf.io.gfile.GFile(ref_filename).read()
  ).strip().splitlines()

  hyp_lines = tokenizer.native_to_unicode(
    tf.io.gfile.GFile(hyp_filename).read()
  ).strip().splitlines()

  if len(ref_lines) != len(hyp_lines):
    raise ValueError("Reference and translation files have different number of "
                     "lines. If training only a few steps (100-200), the "
                     "translation may be empty.")
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]

  return metrics.compute_bleu(ref_tokens, hyp_tokens) * 100
  pass


def main(unused_argv):
  if FLAGS.bleu_variant in ("both", "uncased"):
    score = bleu_wrapper(FLAGS.reference, FLAGS.translation, False)
    tf.logging.info("Case-insensitive result: %f" % score)
  if FLAGS.bleu_variant in ("both", "cased"):
    score = bleu_wrapper(FLAGS.reference, FLAGS.translation, True)
    tf.logging.info("Case-sensitive result: %f" % score)


def define_compute_bleu_flags():
  """Add flags for computing BLEU score."""
  flags.DEFINE_string(
    name="translation", default=None,
    help=flags_core.help_wrap("File containing translated text."))
  flags.mark_flag_as_required("translation")

  flags.DEFINE_string(
    name="reference", default=None,
    help=flags_core.help_wrap("File containing reference translation."))
  flags.mark_flag_as_required("reference")

  flags.DEFINE_enum(
    name="bleu_variant", short_name="bv", default="both",
    enum_values=["both", "uncased", "cased"], case_sensitive=False,
    help=flags_core.help_wrap(
      "Specify one or more BLEU variants to calculate. Variants: \"cased\""
      ", \"uncased\", or \"both\"."))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  define_compute_bleu_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
