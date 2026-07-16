# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from irmv_humanoid_locomotion.utils.fractal_noise import generate_fractal_noise_2d

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def fractal_noise_terrain(difficulty: float, cfg: hf_terrains_cfg.FractalNoiseTerrainCfg) -> np.ndarray:
    """Generate a fractal noise height field terrain.

    The ``difficulty`` parameter is accepted for Isaac Lab API compatibility but unused
    (this terrain uses fixed parameters).

    Args:
        difficulty: Terrain difficulty in [0, 1] (unused).
        cfg: The terrain configuration.

    Returns:
        Height field array (int16) suitable for Isaac Lab terrain import.
    """
    raw_shape = (
        int(cfg.size[0] / cfg.horizontal_scale),
        int(cfg.size[1] / cfg.horizontal_scale),
    )
    res = cfg.noise_frequency
    noise_shape = tuple(s - s % res for s in raw_shape)
    noise = generate_fractal_noise_2d(
        shape=noise_shape,
        res=(res, res),
        octaves=cfg.octaves,
        lacunarity=cfg.lacunarity,
        gain=cfg.gain,
        z_scale=cfg.amplitude,
        centering=cfg.centering,
    )
    padded = np.pad(
        noise,
        ((0, raw_shape[0] - noise_shape[0]), (0, raw_shape[1] - noise_shape[1])),
        mode="edge",
    )
    return (padded / cfg.vertical_scale).astype(np.int16)
