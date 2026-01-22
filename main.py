"""Starlette application."""

from __future__ import annotations

from starlette.applications import Starlette

app = Starlette(debug=True)
