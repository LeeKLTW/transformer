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

#TODO
class UnicodeRegex(object):
  def __init__(self):
    pass
  def nondigit_punct_re(self):
    pass


urgex = UnicodeRegex()

def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.
  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).
  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.
  Args:
    string: the input string
  Returns:
    a list of tokens
  """
  string = urgex.nondigit_punct_re.sub(r"\1 \2", string)
  string = urgex.punct_nondigit_re.sub(r" \1 \2", string)
  string = urgex.symbol_re.sub(r" \1 ",string)
  return string.split()


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
