"""Conformal triage layer (Part D) on top of the radiomics -> LLM classifier."""

from .evaluation import (feasibility_report, feature_dependence, perturbation_report,
                         set_report, sets_to_decisions, triage_report, two_sample_test)
from .ltt import (BENIGN, MALIGNANT, REFER, TriageRule, binomial_pvalue, calibrate_triage,
                  fixed_sequence_test, vote_candidates)
from .prompts import build_perturbed_table, make_variants, render_prompt
from .split_cp import (calibrate_marginal, calibrate_mondrian, conformal_quantile,
                       prediction_sets, softmax_scores, vote_scores)
from .votes import count_votes, predict_malignant_proba, signal_matrix, vote_table
