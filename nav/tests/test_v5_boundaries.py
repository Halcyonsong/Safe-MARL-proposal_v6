"""Guard: the V5 main path must not import QP or an expert u_ref."""

from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
MAIN_FILES = [
    NAV / 'algos/safe_policy/fixed_h.py',
    NAV / 'algos/safe_policy/qp_free.py',
    NAV / 'algos/safe_policy/training.py',
    NAV / 'scripts/train_qp_free.py',
    NAV / 'scripts/evaluate_qp_free.py',
]
FORBIDDEN = ('import osqp', 'import cvxpy', 'import jaxproxqp',
             'solve_agent_qp', 'qp_actions', 'u_ref(')


def test_main_path_has_no_qp_or_expert_uref_import():
    for path in MAIN_FILES:
        text = path.read_text()
        for token in FORBIDDEN:
            assert token not in text, f'{token} found in {path}'
