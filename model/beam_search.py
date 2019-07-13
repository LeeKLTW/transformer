# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

# Default value for INF
INF = 1. * 1e7


class _StateKeys(object):
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


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)

def _get_shape_keep_last_dim(tensor):
  pass


def _length_normalization(alpha, length):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.cast(length, tf.float32)) / 6.), alpha)



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

  #todo continue review the paper
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
    # Current loop index (starts at 0)
    cur_index = tf.constant(0)

    # Create alive sequence with shape [batch_size, beam_size, 1]
    alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
    alive_seq = tf.expand_dims(alive_seq, axis=2)

    # Create tensor for storing initial log probabilities.
    # Assume initial_ids are prob 1.0
    initial_log_probs = tf.constant(
        [[0.] + [-float("inf")] * (self.beam_size - 1)])# [[0.0, -inf, -inf ...]
    alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])

    # Expand all values stored in the dictionary to the beam size, so that each
    # beam has a separate cache.
    alive_cache = nest.map_structure(
        lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)
    # Initialize tensor storing finished sequences with filler values.
    finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)

    # Set scores of the initial finished seqs to negative infinity.
    finished_scores = tf.ones([self.batch_size, self.beam_size]) * -INF

    # Initialize finished flags with all False values.
    finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

    # Create state dictionary
    state = {
        _StateKeys.CUR_INDEX: cur_index,
        _StateKeys.ALIVE_SEQ: alive_seq,
        _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
        _StateKeys.ALIVE_CACHE: alive_cache,
        _StateKeys.FINISHED_SEQ: finished_seq,
        _StateKeys.FINISHED_SCORES: finished_scores,
        _StateKeys.FINISHED_FLAGS: finished_flags
    }

    # Create state invariants for each value in the state dictionary. Each
    # dimension must be a constant or None. A None dimension means either:
    #   1) the dimension's value is a tensor that remains the same but may
    #      depend on the input sequence to the model (e.g. batch size).
    #   2) the dimension may have different values on different iterations.
    state_shape_invariants = {
        _StateKeys.CUR_INDEX: tf.TensorShape([]),
        _StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
        _StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
        _StateKeys.ALIVE_CACHE: nest.map_structure(
            _get_shape_keep_last_dim, alive_cache),
        _StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
        _StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size]),
        _StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size])
    }

    return state, state_shape_invariants

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
    i = state[_StateKeys.CUR_INDEX]
    alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
    finished_scores = state[_StateKeys.FINISHED_SCORES]
    finished_flags = state[_StateKeys.FINISHED_FLAGS]

    not_at_max_decode_length = tf.less(i, self.max_decode_length)

    # Calculate largest length penalty (the larger penalty, the better score).
    max_length_norm = _length_normalization(self.alpha, self.max_decode_length)
    # Get the best possible scores from alive sequences.
    best_alive_scores = alive_log_probs[:, 0] / max_length_norm



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

