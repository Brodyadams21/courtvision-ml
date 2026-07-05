"""Streamlit launcher — run from repo root: streamlit run dashboard/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from courtvision.dashboard.app import main  # noqa: E402

main()
