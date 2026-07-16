# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def generate_perlin_noise_2d(shape: Sequence[int], res: Sequence[int]) -> np.ndarray:
    """Generate 2D Perlin gradient noise.

    Args:
        shape: Output array dimensions (rows, cols).
        res: Number of noise grid cells along each axis. Must evenly divide ``shape``.

    Returns:
        Noise array of shape ``(shape[0], shape[1])`` with values in ``[-1, 1]``.

    Raises:
        ValueError: If ``shape`` is not evenly divisible by ``res`` along each axis.
    """
    if shape[0] % res[0] != 0 or shape[1] % res[1] != 0:
        raise ValueError(f"Shape {shape} must be evenly divisible by resolution {res}.")

    def fade(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    d = (shape[0] // res[0], shape[1] // res[1])
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    shape: Sequence[int],
    res: Sequence[int],
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    z_scale: float = 1.0,
    centering: bool = True,
) -> np.ndarray:
    """Generate 2D fractal Brownian motion (fBm) noise by stacking Perlin noise octaves.

    Args:
        shape: Output array dimensions (rows, cols).
        res: Base noise grid resolution. Must divide ``shape`` for the first octave.
            Subsequent octaves multiply ``res`` by ``lacunarity``; if the result no
            longer divides ``shape`` evenly, that octave is skipped.
        octaves: Number of noise layers to stack.
        lacunarity: Frequency multiplier between successive octaves.
        gain: Amplitude multiplier between successive octaves (``gain < 1`` = higher
            octaves contribute less).
        z_scale: Vertical scaling applied to the final noise (in meters).
        centering: If True, subtract the mean so the terrain is centered around zero.

    Returns:
        Noise array of shape ``(shape[0], shape[1])``, dtype ``float32``, scaled by
        ``z_scale``.
    """
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    current_res = [int(res[0]), int(res[1])]
    for i in range(octaves):
        if shape[0] % current_res[0] != 0 or shape[1] % current_res[1] != 0:
            if i == 0:
                raise ValueError(f"Shape {shape} must be divisible by base resolution {res}.")
            break
        noise += amplitude * generate_perlin_noise_2d(shape, current_res).astype(np.float32)
        amplitude *= gain
        current_res = [int(current_res[0] * lacunarity), int(current_res[1] * lacunarity)]
    noise *= z_scale
    if centering:
        noise -= np.mean(noise)
    return noise
