"""Conformal triage layer (Part D) on top of the radiomics -> LLM classifier."""

from .evaluation import (feasibility_report, feature_dependence, set_report,
                         sets_to_decisions, triage_report)
from .ltt import (BENIGN, MALIGNANT, REFER, TriageRule, binomial_pvalue, calibrate_triage,
                  fixed_sequence_test, vote_candidates)
from .prompts import TEMPLATES, build_perturbed_table, make_variants, render_prompt
from .split_cp import (calibrate_marginal, calibrate_mondrian, conformal_quantile,
                       prediction_sets, softmax_scores, vote_scores)
from .votes import count_votes, predict_malignant_proba, probability_matrix, vote_table
