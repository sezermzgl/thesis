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


if __name__ == "__main__":
    unittest.main()
