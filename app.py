"""
app.py — Streamlit Community Cloud entry point.

This thin shim adds the project root and steel_rag/ to sys.path,
then imports and re-exports the dashboard module so Streamlit Cloud
can find it at the repo root level.

Streamlit Cloud app file: app.py
"""

import sys
from pathlib import Path

# Ensure imports from steel_rag/ work when running from repo root
_ROOT = Path(__file__).parent
_STEEL_RAG = _ROOT / "steel_rag"

for _p in [str(_ROOT), str(_STEEL_RAG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import and run the dashboard (Streamlit re-runs this file on each interaction)
import dashboard  # noqa: F401  — triggers the Streamlit page rendering
