# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utitlity functions for environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _fix15_to_rgba(buf):
  """Converts buffer from a 15-bit fixed-point representation into uint8 RGBA.

  Taken verbatim from the C code for libmypaint.

  Args:
    buf: 15-bit fixed-point buffer represented in `uint16`.

  Returns:
    A `uint8` buffer with RGBA channels.
  """
  rgb, alpha = np.split(buf, [3], axis=2)
  rgb = rgb.astype(np.uint32)
  mask = alpha[:, 0] == 0
  rgb[mask] = 0
  rgb[~mask] = ((rgb[~mask] << 15) + alpha[~mask] // 2) // alpha[~mask]
  rgba = np.concatenate((rgb, alpha), axis=2)
  rgba = (255 * rgba + (1 << 15) // 2) // (1 << 15)
  return rgba.astype(np.uint8)


def _fix15_to_hsva(buf):
    def rgb_to_hsv_vectorized(img):  # img with BGR format
        maxc = img.max(-1)
        minc = img.min(-1)

        out = np.zeros(img.shape)
        out[:, :, 2] = maxc
        out[:, :, 1] = (maxc - minc) / maxc

        divs = (maxc[..., None] - img) / ((maxc - minc)[..., None])
        cond1 = divs[..., 0] - divs[..., 1]
        cond2 = 2.0 + divs[..., 2] - divs[..., 0]
        h = 4.0 + divs[..., 1] - divs[..., 2]
        h[img[..., 2] == maxc] = cond1[img[..., 2] == maxc]
        h[img[..., 1] == maxc] = cond2[img[..., 1] == maxc]
        out[:, :, 0] = (h / 6.0) % 1.0

        out[minc == maxc, :2] = 0
        return out

    rgb, alpha = np.split(_fix15_to_rgba(buf), [3], axis=2)
    hsv = rgb_to_hsv_vectorized((rgb[..., ::-1] * 255).copy())
    return np.concatenate((hsv, alpha), axis=2)


def quadratic_bezier(p_s, p_c, p_e, n):
  t = np.linspace(0., 1., n)
  t = t.reshape((1, n, 1))
  p_s, p_c, p_e = [np.expand_dims(p, axis=1) for p in [p_s, p_c, p_e]]
  p = (1 - t) * (1 - t) * p_s + 2 * (1 - t) * t * p_c + t * t * p_e
  return p


def rgb_to_hsv(red, green, blue):
  """Converts RGB to HSV."""
  hue = 0.0

  red = np.clip(red, 0.0, 1.0)
  green = np.clip(green, 0.0, 1.0)
  blue = np.clip(blue, 0.0, 1.0)

  max_value = np.max([red, green, blue])
  min_value = np.min([red, green, blue])

  value = max_value
  delta = max_value - min_value

  if delta > 0.0001:
    saturation = delta / max_value

    if red == max_value:
      hue = (green - blue) / delta
      if hue < 0.0:
        hue += 6.0
    elif green == max_value:
      hue = 2.0 + (blue - red) / delta
    elif blue == max_value:
      hue = 4.0 + (red - green) / delta

    hue /= 6.0
  else:
    saturation = 0.0
    hue = 0.0

  return hue, saturation, value
