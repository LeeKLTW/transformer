# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Central location for shared arparse convention definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import functools

from absl import app
from absl import flags

# flags.text_wrap: Wraps a given text to a maximum line length and returns it.
_help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                               firstline_indent="\n")

try:
  # Looks up a codec in the Python codec registry and returns a CodecInfo obj.
  codecs.lookup("utf-8")
  help_wrap = _help_wrap
except LookupError:
  def help_wrap(text, *args, **kwargs):
    return _help_wrap(text, *args, **kwargs).replace("\ufeff", "")

app.HelpshortFlag.SHORT_NAME = "h"
