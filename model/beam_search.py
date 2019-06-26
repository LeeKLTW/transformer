# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class _StateKeys:
  """Keys to dictionary storing the state of the beam search loop."""

  # Variable storing the loop index.
  CUR_INDEX = "CUR_INDEX"

  # Top sequences that are alive for each batch item. Alive sequences are ones
  # that have not generated an EOS token. Sequences that reach EOS are marked as
  # finished and moved to the FINISHED_SEQ tensor.
  # Has shape [batch_size, beam_size, CUR_INDEX + 1]
  ALIVE_SEQ = "ALIVE_SEQ"

  # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
  ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"

  # Dictionary of cached values for each alive sequence. The cache stores
  # the encoder output, attention bias, and the decoder attention output from
  # the previous iteration.
  ALIVE_CACHE = "ALIVE_CACHE"

  # Top finished sequences for each batch item.
  # Has shape [batch_size, beam_size, CUR_INDEX + 1]. Sequences that are
  # shorter than CUR_INDEX + 1 are padded with 0s.
  FINISHED_SEQ = "FINISHED_SEQ"

  # Scores for each finished sequence. Score = log probability / length norm
  # Shape [batch_size, beam_size]
  FINISHED_SCORES = "FINISHED_SCORES"

  # Flags indicating which sequences in the finished sequences are finished.
  # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
  # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
  FINISHED_FLAGS = "FINISHED_FLAGS"


#todo
def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  pass


class SequenceBeamSearch(object):
  def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
               beam_size, alpha, max_decode_length, eos_id):

    self.symbols_to_logits_fn = symbols_to_logits_fn
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.beam_size = beam_size
    self.alpha = alpha
    self.max_decode_length = max_decode_length
    self.eos_id = eos_id

  def search(self, initial_ids, initial_cache):
    """Beam search for sequences with highest scores."""

    state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

    finished_state = tf.while_loop(
        self._continue_search, self._search_step, loop_vars=[state],
        shape_invariants=[state_shapes], parallel_iterations=1, back_prop=False)

    finished_state = finished_state[0]

    alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
    alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
    finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
    finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
    finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

    # Account for corner case where there are no finished sequences for a
    # particular batch item. In that case, return alive sequences for that batch
    # item.
    finished_seq = tf.where(
        tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)

    finished_scores = tf.where(
        tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)

    return finished_seq, finished_scores

  #todo
  def _create_initial_state(self, initial_ids, initial_cache):
    """Return initial state dictionary and its shape invariants.

    Args:
      initial_ids: initial ids to pass into the symbols_to_logits_fn.
        int tensor with shape [batch_size, 1]
      initial_cache: dictionary storing values to be passed into the
        symbols_to_logits_fn.

    Returns:
        state and shape invariant dictionaries with keys from _StateKeys
    """
    pass

  #todo
  def _continue_search(self, state):
    """Return whether to continue the search loop.

    The loops should terminate when
      1) when decode length has been reached, or
      2) when the worst score in the finished sequences is better than the best
         score in the alive sequences (i.e. the finished sequences are provably
         unchanging)

    Args:
      state: A dictionary with the current loop state.

    Returns:
      Bool tensor with value True if loop should continue, False if loop should
      terminate.
    """
    pass

  def _search_step(self, state):
    """Beam search loop body.

    Grow alive sequences by a single ID. Sequences that have reached the EOS
    token are marked as finished. The alive and finished sequences with the
    highest log probabilities and scores are returned.

    A sequence's finished score is calculating by dividing the log probability
    by the length normalization factor. Without length normalization, the
    search is more likely to return shorter sequences.

    Args:
      state: A dictionary with the current loop state.

    Returns:
      new state dictionary.
    """
    pass




def sequence_beam_search(symbols_to_logits_fn, initial_ids, initial_cache,
                         vocab_size, beam_size, alpha, max_decode_length,
                         eos_id):
  """Search for sequence of subtoken ids with the largest probability.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> [batch_size * beam_size, index]
        index -> [] (scalar)
        cache -> nested dictionary of tensors [batch_size * beam_size, ...]
      The function must return logits and new cache.
        logits -> [batch * beam_size, vocab_size]
        new cache -> same shape/structure as inputted cache
    initial_ids: Starting ids for each batch item.
      int32 tensor with shape [batch_size]
    initial_cache: dict containing starting decoder variables information
    vocab_size: int size of tokens
    beam_size: int number of beams
    alpha: float defining the strength of length normalization
    max_decode_length: maximum length to decoded sequence
    eos_id: int id of eos token, used to determine when a sequence has finished

  Returns:
    Top decoded sequences [batch_size, beam_size, max_decode_length]
    sequence scores [batch_size, beam_size]
  """
  batch_size = tf.shape(initial_ids)[0]
  sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                           beam_size, alpha, max_decode_length, eos_id)
  return sbs.search(initial_ids, initial_cache)
