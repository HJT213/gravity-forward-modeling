"""Build a 64-cubed density contrast field and its analytic surface response."""
import math

import numpy as np


SIZE = 64
BOUNDS = (20000.0, 20000.0, 7000.0)
MAX_PRISMS = 16
PARAMETERS = ("x", "y", "depth", "length", "width", "height", "contrast")
COORDINATES = [(np.arange(SIZE, dtype=np.float64) + .5) * (bound / SIZE) for bound in BOUNDS]
GRID_X, GRID_Y = np.meshgrid(*COORDINATES[:2], indexing="ij")


def analytic_prism(lower, upper, contrast):
    # Same eight-corner sum and down-positive convention as the thesis generator.
    total = np.zeros_like(GRID_X)
    for ix, edge_x in enumerate((lower[0], upper[0])):
        x = edge_x - GRID_X
        for iy, edge_y in enumerate((lower[1], upper[1])):
            y = edge_y - GRID_Y
            for iz, z in enumerate((lower[2], upper[2])):
                radius = np.sqrt(x*x + y*y + z*z)
                angle = np.arctan2(x*y, z*radius)
                term = x * np.log(np.maximum(y + radius, 1e-12))
                term += y * np.log(np.maximum(x + radius, 1e-12))
                term += z * angle
                total += (1 if (ix + iy + iz) % 2 == 0 else -1) * term
    return total * (6.67e-11 * contrast * 1e5)


def build_model(prisms):
    if not isinstance(prisms, list) or not 1 <= len(prisms) <= MAX_PRISMS:
        raise ValueError("须填写 1–16 个棱柱。")
    density = np.zeros((SIZE,)*3, dtype=np.float64)
    truth = np.zeros((SIZE, SIZE), dtype=np.float64)
    for i, prism in enumerate(prisms, 1):
        if not isinstance(prism, dict) or set(prism) != set(PARAMETERS):
            raise ValueError("棱柱 {} 的字段不完整。".format(i))
        values = [prism[key] for key in PARAMETERS]
        if any(type(v) not in (int, float) or not -20000 <= v <= 20000 or not math.isfinite(v) for v in values):
            raise ValueError("棱柱 {} 的参数必须是有效数值。".format(i))
        center = np.asarray(values[:3]) * 1000
        extents = np.asarray(values[3:6]) * 1000
        contrast = values[6]
        if np.any(extents <= 0):
            raise ValueError("棱柱 {} 的长、宽、高必须大于 0。".format(i))
        if contrast == 0 or not -2000 <= contrast <= 2500:
            raise ValueError("棱柱 {} 的密度差须在 −2000 至 2500 kg/m³ 之间，且不能为 0。".format(i))
        lower = center - extents / 2
        upper = center + extents / 2
        tolerance = 1e-8
        if np.any(lower < -tolerance) or np.any(upper > np.asarray(BOUNDS) + tolerance):
            raise ValueError("棱柱 {} 超出模型域（X、Y 为 0–20 km，深度为 0–7 km）。".format(i))
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, BOUNDS)
        masks = [(axis >= lo - tolerance) & (axis <= hi + tolerance)
                 for axis, lo, hi in zip(COORDINATES, lower, upper)]
        if not all(mask.any() for mask in masks):
            raise ValueError("棱柱 {} 未覆盖任何网格中心，请调整位置或增大尺寸。".format(i))
        density[np.ix_(*masks)] += contrast
        truth += analytic_prism(lower, upper, contrast)

    density = density.astype(np.float32)
    if not np.any(density):
        raise ValueError("叠加后所有网格的密度差均为 0，请调整棱柱参数。")
    return {
        "density": density,
        "truth": truth.astype(np.float32),
        "bounds": BOUNDS,
        "defaultLayer": int(np.argmax(np.count_nonzero(density, axis=(0, 1)))),
        "densityLimit": float(np.max(np.abs(density))),
        "outsideDensityRange": bool(np.min(density) < -2000 or np.max(density) > 2500),
    }
