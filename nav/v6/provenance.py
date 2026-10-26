"""Independence guards for the standalone V6 pipeline.

V6 may reuse generic design and code, but it must not consume official h,
QP/u_ref/expert artifacts, or V5 weights/data.  These helpers reject the
legacy coupled paths early, before a run writes results.
"""

from __future__ import annotations

from pathlib import Path

# Path fragments that identify legacy coupled artifacts.  The guard is
# intentionally conservative: put independent V6 artifacts under
# ``outputs/v6/independent``.
FORBIDDEN_PATH_TOKENS = (
    'Safe-MARL-proposal_v4',
    '/v4',
    '/v5',
    'data_v5h0',
    'h_mech',
    'outputs/v6/h_data',
    'outputs/v6/official_h',
    'cbf.pkl',
)

# Separate list for official-h artifacts.
FORBIDDEN_OFFICIAL_TOKENS = (
    'cbf.pkl',
    'official_h',
)


def assert_independent_path(path: str | Path, *, kind: str = 'artifact') -> Path:
    """Raise if *path* looks like a V4/V5/official coupled artifact."""
    resolved = Path(path)
    text = str(resolved)
    lowered = text.lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token.lower() in lowered:
            raise ValueError(
                f'{kind} is not V6-independent: {resolved} '
                f'(matched {token!r}); use V6 self-collected data / '
                f'self-trained checkpoints under outputs/v6/independent')
    for token in FORBIDDEN_OFFICIAL_TOKENS:
        if token.lower() in lowered:
            raise ValueError(
                f'{kind} is an official-h artifact, which V6 forbids: '
                f'{resolved}')
    return resolved


def assert_independent_text(text: str, *, source: str = 'text') -> None:
    """Raise if a config/source blob contains a forbidden coupled path."""
    lowered = text.lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token.lower() in lowered:
            raise ValueError(
                f'{source} contains forbidden V4/V5/official coupling: '
                f'{token!r}')
