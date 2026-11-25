import unittest

import numpy as np

from prisms import build_model


BASE = dict(x=10, y=10, depth=2.5, length=3, width=2.5, height=1.5, contrast=600)


class PrismTests(unittest.TestCase):
    def test_sign_symmetry_and_grid(self):
        positive = build_model([BASE])
        negative = build_model([dict(BASE, contrast=-600)])
        self.assertEqual(positive["density"].shape, (64, 64, 64))
        self.assertEqual(positive["truth"].shape, (64, 64))
        self.assertGreater(positive["truth"][32, 32], 0)
        np.testing.assert_array_equal(positive["density"], -negative["density"])
        np.testing.assert_allclose(positive["truth"], -negative["truth"])
        np.testing.assert_allclose(positive["truth"], positive["truth"][::-1, :])
        self.assertEqual(set(np.unique(positive["density"])), {0, 600})

    def test_superposition_and_overlap(self):
        other = dict(BASE, contrast=-150, x=11)
        a, b = (build_model([p]) for p in (BASE, other))
        merged = build_model([BASE, other])
        np.testing.assert_array_equal(merged["density"], a["density"] + b["density"])
        np.testing.assert_allclose(merged["truth"], a["truth"] + b["truth"], atol=1e-5)
        self.assertIn(450, np.unique(merged["density"]))
        self.assertTrue(build_model([dict(BASE, contrast=2000)] * 2)["outsideDensityRange"])

    def test_boundary_and_bad_inputs(self):
        near_surface = build_model([dict(BASE, depth=.75)])
        self.assertTrue(np.all(np.isfinite(near_surface["truth"])))
        for invalid in (
            [], [BASE] * 17, [dict(BASE, x=0)], [dict(BASE, length=float("inf"))],
            [dict(BASE, height=0)], [dict(BASE, height=.00001, depth=2.51)],
            [dict(BASE, contrast=2501)], [BASE, dict(BASE, contrast=-600)],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_model(invalid)


if __name__ == "__main__":
    unittest.main()
