"""Streamlit entry point — multipage from the start (CLAUDE.md sections 3, 7).

Run with::

    streamlit run app/main.py

Multipage via ``st.navigation`` / ``st.Page`` is deliberate: in the sibling project the
migration from a single scrolling page was an avoidable refactor (section 7).

The navigation order is the checklist order of section 2 — macro, then market structure,
then the company, then execution. The panel presents the questions in that order and does
not let you skip to the bottom.

**Design constraints that are decisions, not omissions** (sections 2, 12): no
auto-refresh, no live tickers, no intraday charts, daily resolution only. The panel is
pull, not push. Nothing here rewards being opened more than once a week.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/main.py` puts app/ on sys.path, not the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402 — must follow the sys.path fix

from app.format import PUBLIC_NOTE, PUBLIC_PAGES  # noqa: E402
from core.config import load_settings  # noqa: E402

st.set_page_config(page_title="equitydash", page_icon="📊", layout="wide")

PAGES_DIR = Path(__file__).parent / "pages"

# Only the pages that exist are registered. A stub page that says "phase 3" is more
# honest than a page that renders an empty chart (section 12).
pages = [
    st.Page(PAGES_DIR / "today.py", title="Hoy", icon="🏠", default=True),
    st.Page(PAGES_DIR / "company.py", title="Empresa", icon="🏢"),
    st.Page(PAGES_DIR / "portfolio.py", title="Cartera", icon="🔬"),
]

settings = load_settings()
if settings.public_mode:
    # Public mode publishes the method, not the balance. The portfolio page ships, with
    # relative figures only (app/format.py); the fiscal layer never ships at all, because
    # its own configuration reveals the jurisdiction (section 11).
    pages = [p for p in pages if p.title in PUBLIC_PAGES]
    st.sidebar.caption(PUBLIC_NOTE)

st.navigation(pages).run()
