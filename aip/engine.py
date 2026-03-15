"""Humanization engine – thin public wrapper around :mod:`aip.humanizer`.

This module is the primary entry point for callers that need text
humanization.  It re-exports the main :func:`humanize` function and the
:class:`HumanizeResult` dataclass for convenience.
"""

from __future__ import annotations

from .humanizer import HumanizeResult, humanize

__all__ = ["humanize", "HumanizeResult"]
