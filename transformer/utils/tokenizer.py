# -*- coding: utf-8 -*-
def native_to_unicode(s):
  """Convert string to unicode (required in Python 2)."""
  try:               # Python 2
    return s if isinstance(s, unicode) else s.decode("utf-8")
  except NameError:  # Python 3
    return s