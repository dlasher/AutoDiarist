"""Pin test imports to THIS checkout, ahead of any global editable install.

The shared interpreter may carry a PEP-660 editable install of this project
whose finder maps the top-level ``api``/``pipeline`` packages to a *different*
checkout (typically the main worktree).  ``python -m pytest`` happens to dodge
that because the cwd is first on ``sys.path``, but a bare ``pytest`` invocation
from a git worktree would otherwise import and test the other checkout's code
(observable as ``AttributeError`` on helpers that only exist on the branch
under test).

Inserting this conftest's directory at the head of ``sys.path`` makes the
regular PathFinder resolve ``api``/``pipeline`` to the checkout that contains
the tests, which is the only correct target for the suite.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
while _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
