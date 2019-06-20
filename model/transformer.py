# -*- coding: utf-8 -*-
def create_model(params, is_train):
  with tf.name_scope("model"):
    if is_train:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
      internal_model = Transformer(params,name="transformer")
      logits = internal_model([inputs, targets], training=is_train)
      vocab_size = params["vocab_size"]
      label_smoothing = params["label_smoothing"]
      logits = metrics.MetricLayer(vocab_size)([logits, targets])

