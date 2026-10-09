"""Checks against the worked example in the design note (200 nodules, m = 10).

Run from the repo root:  python -m unittest discover tests
"""

import unittest

import numpy as np
import pandas as pd

import conformal_triage as ct

M = 10
BENIGN_AT_V = [60, 30, 16, 10, 6, 4, 2, 2, 0, 0, 0]
MALIGNANT_AT_V = [0, 0, 2, 2, 2, 4, 4, 6, 10, 16, 24]


def toy_calibration():
    votes = np.concatenate([np.repeat(np.arange(M + 1), BENIGN_AT_V),
                            np.repeat(np.arange(M + 1), MALIGNANT_AT_V)])
    y = np.concatenate([np.zeros(sum(BENIGN_AT_V), int), np.ones(sum(MALIGNANT_AT_V), int)])
    return votes, y


class TestLTT(unittest.TestCase):
    def setUp(self):
        self.votes, self.y = toy_calibration()
        t_grid, u_grid = ct.vote_candidates(M)
        self.rule = ct.calibrate_triage(self.votes, self.y, t_grid, u_grid,
                                        alpha_benign=0.10, alpha_malignant=0.20, delta=0.05)

    def test_candidate_grid(self):
        self.assertEqual(ct.vote_candidates(M), ([0, 1, 2, 3, 4], [10, 9, 8, 7, 6]))

    def test_benign_table(self):
        tab = self.rule.benign_table
        self.assertEqual(tab["n"].tolist(), [60, 90, 108, 120, 128])
        self.assertEqual(tab["errors"].tolist(), [0, 0, 2, 4, 6])
        np.testing.assert_allclose(tab["p_value"].round(3).tolist()[2:],
                                   [0.001, 0.006, 0.024], atol=1e-3)
        self.assertAlmostEqual(tab["p_value"].iloc[0], 0.9 ** 60)
        self.assertTrue(tab["certified"].all())

    def test_malignant_table(self):
        tab = self.rule.malignant_table
        self.assertEqual(tab["n"].tolist(), [24, 40, 50, 58, 64])
        self.assertEqual(tab["errors"].tolist(), [0, 0, 0, 2, 4])
        self.assertAlmostEqual(tab["p_value"].iloc[0], 0.8 ** 24)
        self.assertAlmostEqual(round(tab["p_value"].iloc[-1], 3), 0.002)
        self.assertTrue(tab["certified"].all())

    def test_locked_rule(self):
        self.assertEqual((self.rule.t, self.rule.u), (4, 6))
        decisions = self.rule.decide(self.votes)
        self.assertEqual(int((decisions == ct.REFER).sum()), 8)
        self.assertEqual(self.rule.decide(np.array([1, 5, 7])).tolist(),
                         [ct.BENIGN, ct.REFER, ct.MALIGNANT])

    def test_t5_would_fail(self):
        # 136 nodules with v <= 5, 10 malignant: observed 7.4% but p = 0.19.
        p = ct.binomial_pvalue(136, 10, 0.10)
        self.assertAlmostEqual(round(p, 2), 0.19)

    def test_fixed_sequence_stops_at_first_failure(self):
        scores = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        y = np.array([0, 0, 1, 1, 0, 0, 0, 0])
        tab = ct.fixed_sequence_test(scores, y, [0, 1, 2], "benign", alpha=0.5, delta=0.3)
        # t=1 fails, so t=2 is not certified even if its own p-value were small.
        self.assertEqual(tab["certified"].tolist(), [True, False, False])

    def test_nothing_certified(self):
        rule = ct.calibrate_triage(np.array([5, 5]), np.array([0, 1]), [0, 1], [10, 9],
                                   0.05, 0.05, 0.05)
        self.assertIsNone(rule.t)
        self.assertIsNone(rule.u)
        self.assertTrue((rule.decide(np.array([0, 10])) == ct.REFER).all())

    def test_rejects_overlapping_grids(self):
        with self.assertRaises(ValueError):
            ct.calibrate_triage(self.votes, self.y, [0, 6], [10, 5], 0.1, 0.2, 0.05)


class TestSplitCP(unittest.TestCase):
    def setUp(self):
        self.votes, self.y = toy_calibration()
        self.scores = ct.vote_scores(self.votes, M)

    def test_marginal_quantile_and_sets(self):
        q = ct.calibrate_marginal(self.scores, self.y, alpha=0.10)
        self.assertEqual(q.tolist(), [4.0, 4.0])
        sets = ct.prediction_sets(ct.vote_scores(np.array([1, 5, 7]), M), q)
        self.assertEqual(sets.tolist(), [[True, False], [False, False], [False, True]])

    def test_same_partition_as_ltt(self):
        q = ct.calibrate_marginal(self.scores, self.y, alpha=0.10)
        cp_decisions = ct.sets_to_decisions(ct.prediction_sets(self.scores, q))
        rule = ct.TriageRule(4, 6, None, None)
        np.testing.assert_array_equal(cp_decisions, rule.decide(self.votes))

    def test_quantile_infinite_when_too_few(self):
        self.assertEqual(ct.conformal_quantile(np.array([0.1, 0.2]), alpha=0.1), float("inf"))

    def test_mondrian_per_class(self):
        q = ct.calibrate_mondrian(self.scores, self.y, alpha=0.10)
        self.assertEqual(len(q), 2)
        sets = ct.prediction_sets(self.scores, q)
        rep = ct.set_report(sets, self.y)
        self.assertGreaterEqual(rep["coverage_benign"], 0.9)
        self.assertGreaterEqual(rep["coverage_malignant"], 0.9)


class TestPrompts(unittest.TestCase):
    names = ["original_shape2D_MajorAxisLength", "original_firstorder_Range",
             "original_glcm_ClusterShade"]
    values = np.array([0.5, -1.25, 2.0])

    def variants(self):
        return ct.make_variants(self.names, self.values, np.random.default_rng(0),
                                n_jitter=4, jitter_sds=[0.1, 0.2])

    def test_base_matches_training_prompt(self):
        expected = ("Classify this thyroid nodule as benign or malignant based on the following "
                    "standardized radiomic features. The 'Major Axis Length' feature is measured "
                    "at 0.500. The 'Range' feature is measured at -1.250. The 'Cluster Shade' "
                    "feature is measured at 2.000.")
        self.assertEqual(self.variants()[0]["prompt"], expected)

    def test_layout(self):
        v = self.variants()
        self.assertEqual([x["kind"] for x in v], ["base"] + ["jitter"] * 8 + ["drop"] * 3)
        self.assertEqual([x["query"] for x in v], list(range(12)))
        self.assertEqual([x["noise_sd"] for x in v[1:9]], [0.1] * 4 + [0.2] * 4)

    def test_jitter_keeps_format(self):
        for x in self.variants()[1:9]:
            self.assertTrue(x["prompt"].startswith("Classify this thyroid nodule"))
            self.assertEqual(x["prompt"].count("feature is measured at"), 3)
            self.assertNotEqual(x["prompt"], self.variants()[0]["prompt"])

    def test_readable_names(self):
        from conformal_triage.prompts import readable_feature_name as r
        self.assertEqual(r("original_glszm_LargeAreaHighGrayLevelEmphasis"),
                         "Large Area High Gray Level Emphasis")
        self.assertEqual(r("original_shape2D_MajorAxisLength"), "Major Axis Length")
        self.assertEqual(r("original_glcm_MCC"), "MCC")
        self.assertEqual(r("original_firstorder_10Percentile"), "10 Percentile")

    def test_names_from_filtered_images(self):
        from conformal_triage.prompts import readable_feature_name as r
        self.assertEqual(r("log-sigma-2-0-mm-3D_firstorder_Mean"), "LoG 2.0 Mean")
        self.assertEqual(r("wavelet-LH_glcm_Contrast"), "Wavelet LH Contrast")

    def test_shared_names_get_family(self):
        names = ["original_glszm_GrayLevelNonUniformity", "original_glrlm_GrayLevelNonUniformity",
                 "original_shape2D_Elongation"]
        text = ct.render_prompt(names, np.array([0.1, 0.2, 0.3]))
        self.assertIn("'Gray Level Non Uniformity (GLSZM)'", text)
        self.assertIn("'Gray Level Non Uniformity (GLRLM)'", text)
        self.assertIn("'Elongation' feature", text)

    def test_value_formats(self):
        z = ct.render_prompt(self.names, self.values, value_format="zscore")
        self.assertIn("'Range' feature has a z-score of -1.25 relative to the training nodules.", z)
        pct = ct.render_prompt(self.names, np.array([12.4, 50.0, 97.6]), value_format="percentile")
        self.assertIn("'Major Axis Length' feature is at percentile 12 of the training nodules.", pct)
        self.assertEqual(ct.render_prompt(self.names, self.values),
                         ct.render_prompt(self.names, self.values, value_format="decimal"))

    def test_drop_removes_only_that_feature(self):
        drops = self.variants()[9:]
        self.assertNotIn("Range", drops[1]["prompt"])
        self.assertEqual(drops[1]["prompt"],
                         ct.render_prompt(self.names, self.values, dropped=1))
        self.assertIn("'Cluster Shade' feature is measured at 2.000.", drops[1]["prompt"])

    def test_perturbed_table_is_deterministic_per_nodule(self):
        table = pd.DataFrame({"sample_id": ["a", "b"], "label": [0, 1],
                              **{n: [0.1, 0.2] for n in self.names}})
        kw = dict(n_jitter=3, jitter_sds=[0.1], seed=42)
        long1 = ct.build_perturbed_table(table, self.names, **kw)
        long2 = ct.build_perturbed_table(table.iloc[::-1], self.names, **kw)
        key = ["sample_id", "query"]
        pd.testing.assert_frame_equal(long1.sort_values(key).reset_index(drop=True),
                                      long2.sort_values(key).reset_index(drop=True))
        self.assertEqual(len(long1), 2 * (1 + 3 + 3))


class TestSignals(unittest.TestCase):
    def setUp(self):
        names = ["f_A", "f_B"]
        table = pd.DataFrame({"sample_id": ["a", "b", "c"], "label": [0, 1, 1],
                              "f_A": [0.0, 1.0, 2.0], "f_B": [0.0, -1.0, 1.0]})
        self.long = ct.build_perturbed_table(table, names, n_jitter=2, jitter_sds=[0.1, 0.3],
                                             seed=0)
        self.long["prob_malignant"] = np.linspace(0.05, 0.95, len(self.long))

    def test_signal_matrix_shapes(self):
        p, y, keys = ct.signal_matrix(self.long, "jitter", noise_sd=0.3)
        self.assertEqual(p.shape, (3, 3))
        self.assertEqual(keys, ["a", "b", "c"])
        self.assertEqual(y.tolist(), [0, 1, 1])
        base = self.long.query("kind == 'base'")["prob_malignant"].to_numpy()
        np.testing.assert_allclose(p[:, 0], base)
        p_drop, _, _ = ct.signal_matrix(self.long, "drop")
        self.assertEqual(p_drop.shape, (3, 3))

    def test_missing_level_raises(self):
        with self.assertRaises(ValueError):
            ct.signal_matrix(self.long, "jitter", noise_sd=0.7)

    def test_reports_run(self):
        rep = ct.perturbation_report(self.long)
        self.assertEqual(rep.index.tolist(),
                         [("base", 0.0), ("drop", 0.0), ("jitter", 0.1), ("jitter", 0.3)])
        dep = ct.feature_dependence(self.long)
        self.assertEqual(sorted(dep.index), ["f_A", "f_B"])
        self.assertTrue((dep["n"] == 3).all())

    def test_two_sample_test(self):
        rng = np.random.default_rng(0)
        same = ct.two_sample_test(rng.normal(size=(200, 3)), rng.normal(size=(200, 3)), 200)
        shifted = ct.two_sample_test(rng.normal(size=(200, 3)), rng.normal(1, 1, (200, 3)), 200)
        self.assertGreater(same["p_value"], 0.01)
        self.assertLess(shifted["p_value"], 0.01)
        self.assertGreater(shifted["auc"], 0.7)


class TestSelection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        size = y + rng.normal(0, 1, 500)
        self.y = y
        self.X = pd.DataFrame({
            "area": size,
            "area_copy": size + rng.normal(0, 0.01, 500),     # near-duplicate of area
            "texture": rng.normal(0, 1, 500),
            "weak_size": 0.5 * size + rng.normal(0, 1, 500),  # correlated, but not a copy
        })

    def test_drops_only_near_duplicates(self):
        kept, dropped = ct.drop_correlated(self.X, self.y, max_corr=0.95)
        self.assertEqual(len(dropped), 1)
        self.assertIn("texture", kept)
        self.assertIn("weak_size", kept)
        (gone, twin), = dropped.items()
        self.assertEqual({gone, twin}, {"area", "area_copy"})
        self.assertEqual(kept, [c for c in self.X.columns if c != gone])

    def test_keeps_more_label_related_copy(self):
        X = self.X.assign(area_copy=self.X["area"] - 0.5 * self.y)   # r ~ 0.98 with area, weaker label link
        kept, dropped = ct.drop_correlated(X, self.y, max_corr=0.9)
        self.assertEqual(dropped, {"area_copy": "area"})

    def test_threshold_one_keeps_everything(self):
        kept, dropped = ct.drop_correlated(self.X, self.y, max_corr=1.0)
        self.assertEqual(kept, list(self.X.columns))
        self.assertEqual(dropped, {})


class TestSemanticPrompts(unittest.TestCase):
    names = ["original_shape2D_Sphericity", "original_shape2D_MajorAxisLengthRelative",
             "original_shape2D_Elongation", "original_gldm_DependenceNonUniformity"]
    z = np.array([-1.2, 0.1, 1.9, 1.0])
    pct = np.array([12.0, 55.0, 97.0, 83.0])

    def test_rich_sentence(self):
        text = ct.render_prompt(self.names, self.z, value_format="rich", percentiles=self.pct)
        self.assertIn("The 'Sphericity' feature has a z-score of -1.20 and is at percentile 12 of the "
                      "training nodules: lower than typical, meaning a less round, more irregular outline.", text)
        self.assertIn("percentile 55 of the training nodules: close to typical.", text)        # no phrase when typical
        self.assertIn("meaning a more compact, less elongated shape", text)
        self.assertIn("Elongation (the ratio of the shortest to the longest axis", text)    # misleading name defined
        self.assertIn("'Dependence Non Uniformity' feature has a z-score of +1.00 and is at percentile 83 "
                      "of the training nodules: higher than typical.", text)                  # size-driven: no phrase

    def test_semantic_only_has_no_numbers(self):
        text = ct.render_prompt(self.names, self.z, value_format="semantic_only", percentiles=self.pct)
        self.assertIn("The 'Sphericity' feature is lower than typical, meaning a less round, more irregular outline.", text)
        self.assertNotIn("z-score", text)
        self.assertNotIn("percentile", text)

    def test_relative_sizes(self):
        from conformal_triage.semantics import meaning
        self.assertEqual(meaning("original_shape2D_MajorAxisLengthRelative", 95), "a longer nodule relative to the image size")
        self.assertEqual(meaning("original_shape2D_MajorAxisLength", 5), "a shorter nodule")
        self.assertNotIn("image size", meaning("original_shape2D_PerimeterSurfaceRatioRelative", 95))

    def test_bands(self):
        from conformal_triage.semantics import band
        self.assertEqual([band(p)[1] for p in (5, 10, 29.9, 50, 70, 89, 90, 100)],
                         ["much lower than typical", "lower than typical", "lower than typical", "close to typical",
                          "higher than typical", "higher than typical", "much higher than typical", "much higher than typical"])

    def test_needs_percentiles(self):
        with self.assertRaises(ValueError):
            ct.render_prompt(self.names, self.z, value_format="rich")


class TestMondrianPerClass(unittest.TestCase):
    def test_per_class_alpha_and_purity(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 400)
        p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
        scores = ct.softmax_scores(p)
        q_same = ct.calibrate_mondrian(scores, y, alpha=0.10)
        q_pair = ct.calibrate_mondrian(scores, y, alpha=(0.10, 0.05))
        self.assertEqual(q_same[0], q_pair[0])                 # benign threshold unchanged
        self.assertGreaterEqual(q_pair[1], q_same[1])          # stricter on cancers: wider "malignant"
        rep = ct.set_report(ct.prediction_sets(scores, q_pair), y)
        self.assertGreaterEqual(rep["coverage_malignant"], 0.95)
        sets = ct.prediction_sets(scores, q_pair)
        benign_only = sets[:, 0] & ~sets[:, 1]
        self.assertAlmostEqual(rep["cancer_share_in_benign_only"], y[benign_only].mean())


class TestClinicalFeatures(unittest.TestCase):
    @staticmethod
    def scene(h_axis, w_axis, inside=60.0, outside=120.0, lobes=False):
        yy, xx = np.mgrid[0:200, 0:200]
        mask = ((yy - 100) / h_axis) ** 2 + ((xx - 100) / w_axis) ** 2 <= 1
        if lobes:
            mask &= ~(((yy - 100) ** 2 + (xx - (100 + w_axis)) ** 2) <= (0.6 * w_axis) ** 2)
            mask &= ~(((yy - 100) ** 2 + (xx - (100 - w_axis)) ** 2) <= (0.6 * w_axis) ** 2)
        rng = np.random.default_rng(0)
        gray = np.where(mask, inside, outside) + rng.normal(0, 5, mask.shape)
        return np.clip(gray, 0, 255), mask

    def test_orientation(self):
        tall = ct.clinical_features(*self.scene(50, 30))
        wide = ct.clinical_features(*self.scene(30, 50))
        self.assertGreater(tall["original_clinical_TallerThanWideRatio"], 1.4)
        self.assertLess(wide["original_clinical_TallerThanWideRatio"], 0.7)

    def test_echogenicity(self):
        dark = ct.clinical_features(*self.scene(40, 40, inside=60, outside=120))
        bright = ct.clinical_features(*self.scene(40, 40, inside=150, outside=120))
        self.assertAlmostEqual(dark["original_clinical_EchogenicityRatio"], 0.5, delta=0.05)
        self.assertGreater(bright["original_clinical_EchogenicityRatio"], 1.15)

    def test_punctate_foci(self):
        gray, mask = self.scene(40, 40)
        base = ct.clinical_features(gray, mask)["original_clinical_PunctateFociDensity"]
        for y, x in [(90, 90), (110, 105), (95, 112), (105, 88)]:
            gray[y - 1:y + 2, x - 1:x + 2] = 250
        with_foci = ct.clinical_features(gray, mask)["original_clinical_PunctateFociDensity"]
        self.assertLess(base, 0.3)                 # speckle alone gives (almost) no foci
        self.assertAlmostEqual(with_foci * mask.sum() / 1000, 4, delta=0.6)   # four 3x3 foci

    def test_solidity_and_margin(self):
        smooth = ct.clinical_features(*self.scene(40, 40))
        lobed = ct.clinical_features(*self.scene(40, 40, lobes=True))
        self.assertGreater(smooth["original_clinical_Solidity"], 0.97)
        self.assertLess(lobed["original_clinical_Solidity"], smooth["original_clinical_Solidity"] - 0.03)
        gray, mask = self.scene(40, 40)
        from scipy import ndimage as ndi
        blurred = ct.clinical_features(ndi.gaussian_filter(gray, 4), mask)
        self.assertGreater(smooth["original_clinical_MarginSharpness"], blurred["original_clinical_MarginSharpness"])

    def test_solid_echogenicity_ignores_fluid(self):
        gray, mask = self.scene(40, 40, inside=60, outside=120)
        rows = np.mgrid[0:200, 0:200][0]
        gray[(rows < 100) & mask] = 10                          # half the nodule is fluid
        f = ct.clinical_features(gray, mask)
        self.assertLess(f["original_clinical_EchogenicityRatio"], 0.35)          # pulled down by fluid
        self.assertAlmostEqual(f["original_clinical_SolidEchogenicityRatio"], 0.5, delta=0.05)

    def test_anechoic_fraction(self):
        gray, mask = self.scene(40, 40, inside=110, outside=120)
        self.assertLess(ct.clinical_features(gray, mask)["original_clinical_AnechoicFraction"], 0.01)
        gray[(np.mgrid[0:200, 0:200][0] < 100) & mask] = 10      # upper half fluid-like
        frac = ct.clinical_features(gray, mask)["original_clinical_AnechoicFraction"]
        self.assertAlmostEqual(frac, 0.5, delta=0.06)


if __name__ == "__main__":
    unittest.main()
