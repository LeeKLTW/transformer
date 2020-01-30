# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def _get_ngrams_with_counter():
  pass


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict(
      (ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in
      ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]

    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
        ngram]

  precisions = [0] * max_order
  smooth = 1.0

  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0

  bleu = geo_mean * bp
  return np.float32(bleu)
