# general_k package
# Optimal FWER-controlling tests for K exchangeable hypotheses
#
# Main entry points:
#   coordinate_descent_fast  - find optimal dual multipliers mu*
#   compare_methods          - compare algorithm against Bonferroni, Holm, etc.
#   decisions                - compute optimal rejection decisions given mu*

from general_k.algorithm import coordinate_descent_fast, coordinate_descent
from general_k.simulations import compare_methods
from general_k.decision import decisions, optimal_l_star
from general_k.alternatives import (
    truncated_normal_g,
    mixture_normal_g,
    student_t_g,
    beta_g,
)
from general_k.baselines import bonferroni, holm, hochberg, hommel, romano_wolf

__version__ = "0.1.0"
__all__ = [
    "coordinate_descent_fast",
    "coordinate_descent",
    "compare_methods",
    "decisions",
    "optimal_l_star",
    "truncated_normal_g",
    "mixture_normal_g",
    "student_t_g",
    "beta_g",
    "bonferroni",
    "holm",
    "hochberg",
    "hommel",
    "romano_wolf",
]
