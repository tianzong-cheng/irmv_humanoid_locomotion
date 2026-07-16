# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class FractalNoiseTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a fractal noise (fBm) height field terrain."""

    function = hf_terrains.fractal_noise_terrain

    amplitude: float = 0.05
    """Peak noise height scale (in m). Applied as z_scale to the fBm output."""

    noise_frequency: int = 10
    """Base noise grid resolution before lacunarity. Higher = more, smaller bumps."""

    octaves: int = 2
    """Number of Perlin noise layers stacked for the fractal."""

    lacunarity: float = 2.0
    """Frequency multiplier between successive octaves."""

    gain: float = 0.5
    """Amplitude multiplier between successive octaves (gain < 1 = higher octaves smaller)."""

    centering: bool = True
    """If True, subtract the mean so the terrain is centered around zero."""
