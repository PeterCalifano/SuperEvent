"""Repository-level pytest collection controls.

These files are library/utility modules with `test_*.py` names, not project
tests. Keep them excluded from automatic pytest discovery.
"""

collect_ignore = [
    "ts_generation/test_ts.py",
    "models/backbones/maxvit_backbone/layers/maxvit/layers/test_time_pool.py",
]
